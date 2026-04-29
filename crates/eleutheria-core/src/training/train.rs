//! Training loop pro Core Memory.
//!
//! Produkční varianta single-iteration smoke testu z `smoke.rs`:
//! epochs × batches × gradient accumulation → AdamW step. Žádná
//! validace, žádný checkpoint resume — to přijde v alpha.12. Tenhle
//! modul je minimální loop, co umí odtrénovat malý korpus a reportovat
//! loss curve.
//!
//! **Gradient accumulation:** `grad_accum_steps` micro-batches se
//! akumulují (mean of losses), pak se volá `optimizer.step()`. Snižuje
//! efektivní batch VRAM — ale plné N-layer backward graph stále musí
//! projít jedním micro-batchem, takže accumulation samo o sobě neřeší
//! CUDA OOM z alpha.10 (6 GB nezvládne ani seq_len=2 micro-batch).
//! Řešení OOM = gradient checkpointing (alpha.12) nebo CPU fallback.

use anyhow::{Result, anyhow};
use candle_core::Tensor;
use candle_core::backprop::GradStore;
use candle_nn::ParamsAdamW;
use candle_nn::optim::Optimizer;

use crate::Sofie;
use crate::training::adamw_state::EleutheriaAdamW;
use crate::training::core_memory::CoreMemoryStack;
use crate::training::loss::cross_entropy_next_token;
use crate::training::optim_io::OptimizerArtifact;

/// Konfigurace training loopu.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Počet epoch (kolikrát se projede celý dataset).
    pub epochs: usize,
    /// Micro-batch size — počet sekvencí v jednom forward passu.
    pub batch_size: usize,
    /// Počet micro-batches, co se akumulují před `optimizer.step()`.
    /// Efektivní batch size = `batch_size * grad_accum_steps`.
    pub grad_accum_steps: usize,
    /// AdamW learning rate.
    pub learning_rate: f64,
    /// Max L2 norm pro gradient clipping. `None` = bez clippingu.
    pub grad_clip: Option<f64>,
    /// Seed pro shuffle datasetu per epoch (base value — epoch idx se
    /// přičte, aby každá epoch měla jiné pořadí).
    pub shuffle_seed: u64,
    /// Logování — po kolika optimizer stepech vypsat running loss.
    pub log_every_n_steps: usize,
    /// Použít gradient checkpointing (per-layer chunked backward, alpha.12).
    /// Sníží peak memory cca 10–20× za cenu ~2× delšího kroku — určeno
    /// pro CUDA, kde plný 24-vrstvý backward graf nesedí do VRAM (KI-005).
    pub checkpoint: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 1,
            batch_size: 1,
            grad_accum_steps: 1,
            learning_rate: 1e-3,
            grad_clip: Some(1.0),
            shuffle_seed: 0,
            log_every_n_steps: 10,
            checkpoint: false,
        }
    }
}

/// Souhrn jednoho tréninkového běhu.
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Celkový počet AdamW kroků napříč epochami.
    pub total_steps: usize,
    /// Celkový počet micro-batch forward/backward passů.
    pub total_micro_batches: usize,
    /// Loss na první kroku (baseline).
    pub initial_loss: f64,
    /// Loss na posledním kroku.
    pub final_loss: f64,
    /// Nejnižší loss v průběhu tréninku.
    pub best_loss: f64,
    /// Loss na konci každé epoch (mean přes epoch).
    pub loss_per_epoch: Vec<f64>,
    /// Wall-clock celkového času v ms.
    pub wall_time_ms: u128,
    /// True = loss klesl mezi první a poslední iterací. Minimální signál,
    /// že se Core Memory vůbec učí.
    pub loss_decreased: bool,
}

impl Sofie {
    /// Training loop — trénuje `CoreMemoryStack` na datasetu tokenů.
    ///
    /// - `stack`: multi-layer trainable Core Memory (in/out — Vars se
    ///   modifikují in-place přes AdamW)
    /// - `dataset`: tokenizovaný korpus, batches se shufflují per epoch
    /// - `config`: `TrainingConfig`
    /// - `resume_optim`: volitelný `OptimizerArtifact` pro restore m, v,
    ///   step_t (KI-007). Pokud `Some`, aplikuje se na čerstvý
    ///   `EleutheriaAdamW` před prvním stepem. Pokud `None`, AdamW startuje
    ///   s prázdným state (warmup overshoot, RN-006).
    ///
    /// Vrací `(TrainingResult, EleutheriaAdamW)` — caller (run_train) má
    /// optimizer pro `OptimizerArtifact::from_optimizer` save.
    ///
    /// Volá se z `run_train` v main.rs po načtení datasetu a CoreMemoryStack.
    ///
    /// **Halt na NaN:** pokud nějaký micro-batch produkuje NaN loss
    /// (backward instability), funkce vrací Err — trénink selže viditelně,
    /// nepoškozuje trained state.
    pub fn train_core_memory(
        &self,
        stack: &CoreMemoryStack,
        dataset: &crate::training::dataset::TokenDataset,
        config: &TrainingConfig,
        resume_optim: Option<&OptimizerArtifact>,
    ) -> Result<(TrainingResult, EleutheriaAdamW)> {
        let start = std::time::Instant::now();

        if config.epochs == 0 {
            return Err(anyhow!("epochs musí být > 0"));
        }
        if config.grad_accum_steps == 0 {
            return Err(anyhow!("grad_accum_steps musí být > 0"));
        }

        // AdamW optimizer — vezme všech 24 Vars najednou. Vlastní wrapper
        // s veřejným state pro persistence (alpha.16, KI-007).
        let vars = stack.vars_owned();
        let mut opt = EleutheriaAdamW::new(
            vars.clone(),
            ParamsAdamW {
                lr: config.learning_rate,
                ..ParamsAdamW::default()
            },
        )?;
        if let Some(art) = resume_optim {
            art.apply_to_optimizer(&mut opt)
                .map_err(|e| anyhow!("resume optimizer state: {e}"))?;
            tracing::info!(
                "AdamW state restored: step_t={}, prior HP lr={:.4e} β1={:.3} β2={:.4}",
                opt.step_t(),
                art.meta().lr,
                art.meta().beta1,
                art.meta().beta2,
            );
        }

        let mut total_steps = 0usize;
        let mut total_micro_batches = 0usize;
        let mut initial_loss: Option<f64> = None;
        let mut last_loss: f64 = f64::NAN;
        let mut best_loss: f64 = f64::INFINITY;
        let mut loss_per_epoch: Vec<f64> = Vec::with_capacity(config.epochs);

        for epoch in 0..config.epochs {
            let seed = config.shuffle_seed.wrapping_add(epoch as u64);
            let batches = dataset.iter_batches(config.batch_size, self.device_ref(), seed)?;
            let mut epoch_loss_sum = 0.0f64;
            let mut epoch_batch_count = 0usize;

            // Accumulator pro gradient (clone gradients přes N micro-batches)
            let mut accum_grads: Option<GradStore> = None;
            let mut accum_count = 0usize;
            let mut accum_loss_sum = 0.0f64;

            for batch in batches {
                // Forward + backward pro jeden micro-batch — checkpointed
                // varianta drop-uje per-layer activations, plný backward
                // držg všechno v autograd graphu (rychlejší, ale OOM rizika).
                let (loss_val, grads) = if config.checkpoint {
                    self.forward_backward_checkpointed(stack, &batch)?
                } else {
                    self.forward_backward_micro_batch(stack, &batch)?
                };
                total_micro_batches += 1;
                if !loss_val.is_finite() {
                    return Err(anyhow!(
                        "NaN/Inf loss na micro-batch {} (epoch {}) — training zastaven",
                        total_micro_batches,
                        epoch
                    ));
                }
                if initial_loss.is_none() {
                    initial_loss = Some(loss_val);
                }
                epoch_loss_sum += loss_val;
                epoch_batch_count += 1;
                accum_loss_sum += loss_val;

                // Akumuluj gradient (sum across micro-batches)
                accum_grads = Some(match accum_grads.take() {
                    None => grads,
                    Some(prev) => merge_grads(prev, grads, &vars)?,
                });
                accum_count += 1;

                // Pokud jsme dosáhli grad_accum_steps, aplikuj step
                if accum_count >= config.grad_accum_steps {
                    let mut final_grads = accum_grads.take().unwrap();

                    // Mean z akumulovaných gradientů (dělí se accum_count)
                    scale_grads(&mut final_grads, &vars, 1.0 / accum_count as f64)?;

                    // Gradient clipping (global L2 norm)
                    if let Some(max_norm) = config.grad_clip {
                        let var_refs: Vec<&candle_core::Var> = vars.iter().collect();
                        crate::training::clip::clip_grad_norm(
                            &mut final_grads,
                            &var_refs,
                            max_norm,
                        )?;
                    }

                    opt.step(&final_grads)?;
                    total_steps += 1;
                    let step_loss = accum_loss_sum / accum_count as f64;
                    last_loss = step_loss;
                    if step_loss < best_loss {
                        best_loss = step_loss;
                    }

                    if total_steps.is_multiple_of(config.log_every_n_steps) {
                        tracing::info!(
                            "step {} (epoch {}, micro-batch {}): loss={:.4}, best={:.4}",
                            total_steps,
                            epoch,
                            total_micro_batches,
                            step_loss,
                            best_loss
                        );
                    }

                    accum_count = 0;
                    accum_loss_sum = 0.0;
                }
            }

            // Pokud na konci epoch zbývají akumulované grady (méně než
            // grad_accum_steps), provedeme dodatečný step se škálováním
            // dle skutečného accum_count.
            if accum_count > 0
                && let Some(mut final_grads) = accum_grads.take()
            {
                scale_grads(&mut final_grads, &vars, 1.0 / accum_count as f64)?;
                if let Some(max_norm) = config.grad_clip {
                    let var_refs: Vec<&candle_core::Var> = vars.iter().collect();
                    crate::training::clip::clip_grad_norm(&mut final_grads, &var_refs, max_norm)?;
                }
                opt.step(&final_grads)?;
                total_steps += 1;
                let step_loss = accum_loss_sum / accum_count as f64;
                last_loss = step_loss;
                if step_loss < best_loss {
                    best_loss = step_loss;
                }
            }

            let epoch_mean = epoch_loss_sum / epoch_batch_count.max(1) as f64;
            loss_per_epoch.push(epoch_mean);
            tracing::info!(
                "epoch {} done: mean loss = {:.4}, total steps = {}",
                epoch,
                epoch_mean,
                total_steps
            );
        }

        let wall_time_ms = start.elapsed().as_millis();
        let initial = initial_loss.unwrap_or(f64::NAN);
        let loss_decreased = last_loss.is_finite() && initial.is_finite() && last_loss < initial;

        let result = TrainingResult {
            total_steps,
            total_micro_batches,
            initial_loss: initial,
            final_loss: last_loss,
            best_loss,
            loss_per_epoch,
            wall_time_ms,
            loss_decreased,
        };
        Ok((result, opt))
    }

    /// Interní: jeden forward + backward na micro-batch. Vrací
    /// `(loss_scalar, grad_store)`. Optimizer step se aplikuje později
    /// v loop (po akumulaci).
    fn forward_backward_micro_batch(
        &self,
        stack: &CoreMemoryStack,
        input_ids: &Tensor,
    ) -> Result<(f64, GradStore)> {
        // Čerstvý ModelState pro každý micro-batch — vliv na SSM state
        // z minulých batchí nechceme v backward grafu.
        let mut state = self.new_model_state()?;
        stack
            .inject_into_state(&mut state, self.dtype_ref())
            .map_err(|e| anyhow!("inject_into_state: {e}"))?;

        let logits = self.model_forward(input_ids, 0, &mut state)?;
        let loss = cross_entropy_next_token(&logits, input_ids)
            .map_err(|e| anyhow!("cross_entropy: {e}"))?;
        let loss_val: f64 = loss.to_scalar::<f32>()? as f64;
        let grads = loss.backward()?;
        Ok((loss_val, grads))
    }
}

/// Merge dvou gradient storů — element-wise sum pro každý Var.
/// `acc = acc + new` (in-place modifikace `acc`).
fn merge_grads(mut acc: GradStore, new: GradStore, vars: &[candle_core::Var]) -> Result<GradStore> {
    for var in vars {
        let tensor = var.as_tensor();
        let new_grad = match new.get(tensor) {
            Some(g) => g.clone(),
            None => continue,
        };
        let combined = match acc.get(tensor) {
            Some(prev) => (prev + &new_grad)?,
            None => new_grad,
        };
        acc.insert(tensor, combined);
    }
    Ok(acc)
}

/// Scale gradient pro každý Var faktorem `scale`.
fn scale_grads(grads: &mut GradStore, vars: &[candle_core::Var], scale: f64) -> Result<()> {
    for var in vars {
        let tensor = var.as_tensor();
        if let Some(g) = grads.get(tensor) {
            let scaled = (g * scale)?;
            grads.insert(tensor, scaled);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_config_default_sensible() {
        let c = TrainingConfig::default();
        assert_eq!(c.epochs, 1);
        assert_eq!(c.batch_size, 1);
        assert_eq!(c.grad_accum_steps, 1);
        assert!(c.learning_rate > 0.0);
        assert_eq!(c.grad_clip, Some(1.0));
    }

    #[test]
    fn training_result_loss_decreased_detection() {
        let r = TrainingResult {
            total_steps: 10,
            total_micro_batches: 10,
            initial_loss: 11.0,
            final_loss: 8.5,
            best_loss: 8.5,
            loss_per_epoch: vec![9.0, 8.5],
            wall_time_ms: 1000,
            loss_decreased: true,
        };
        assert!(r.loss_decreased);

        let r2 = TrainingResult {
            loss_decreased: false,
            final_loss: 12.0,
            ..r.clone()
        };
        assert!(!r2.loss_decreased);
    }
}
