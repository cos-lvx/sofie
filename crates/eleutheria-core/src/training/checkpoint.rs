//! Gradient checkpointing pro Core Memory training (Fáze 5 alpha.12).
//!
//! Klasický gradient checkpointing redukuje peak memory během backward
//! tím, že nedrží všechny intermediate activations v autograd grafu.
//! Místo toho se vrstvy chunked-forwardují bez gradient-trackingu, jen
//! s detached snapshoty inputů. Při backward se každý chunk re-forwarduje
//! s autograd, dostane gradient cíl z následujícího chunku, a propaguje
//! gradient zpět přes synthetic loss.
//!
//! ## Synthetic loss trick
//!
//! Candle backward startuje od skalárního loss. Pro chunked backward
//! potřebujem propagovat libovolný gradient (tensor) skrz chunk —
//! konvertujeme to na skalár přes `synth = sum(output * grad_target)`.
//! Pak `synth.backward()` z chain rule vrátí korektní gradienty:
//!
//! - `d_synth / d_init_state[i] = grad_target * d_output / d_init_state[i]`
//! - `d_synth / d_x_in[i] = grad_target * d_output / d_x_in[i]`
//!
//! Druhý vzorec je gradient cíl pro chunk i-1.
//!
//! ## Per-layer chunking (alpha.12 baseline)
//!
//! Každá vrstva je samostatný chunk. To je nejjednodušší granularita —
//! peak memory během backward = activation graph **jedné** vrstvy plus
//! saved inputs (24× detached tensor). Compute overhead: každá vrstva
//! se forward-uje 2× (jednou no-grad pro snapshot, jednou s autograd v
//! backward).
//!
//! Optimalizace v alpha.13+: chunky po K vrstvách (compromise mezi
//! memory a compute), nebo komponentně (drop attention activations,
//! keep SSM scan).

use anyhow::{Result, anyhow};
use candle_core::backprop::GradStore;
use candle_core::{Tensor, Var};

use crate::Sofie;
use crate::falcon_h1::state::LayerState;
use crate::training::core_memory::CoreMemoryStack;
use crate::training::loss::cross_entropy_next_token;

impl Sofie {
    /// Per-layer chunked forward + backward s gradient checkpointing.
    ///
    /// Nahrazuje `forward_backward_micro_batch` (z `train.rs`) pro případy,
    /// kdy plný autograd graf přes 24 vrstev nesedí do paměti (CUDA OOM
    /// na 6 GB VRAM, KI-005).
    ///
    /// **Vrací:** `(loss_scalar, GradStore)` — kompatibilní formát s
    /// `forward_backward_micro_batch`. AdamW step volá caller (training loop).
    ///
    /// **Sekvence:**
    ///
    /// 1. **No-grad forward sweep** — embedding, pak per-layer forward
    ///    s detached input. Po každé vrstvě uložíme snapshot stavu (před
    ///    vrstvou) a detached input. Nakonec detached hidden po poslední
    ///    vrstvě.
    /// 2. **Final chunk backward** — re-forward final_norm + lm_head s
    ///    autograd, compute cross_entropy loss, `loss.backward()`. Z
    ///    GradStore vyjmeme gradient na poslední hidden — to je vstupní
    ///    gradient pro layer N-1.
    /// 3. **Reverse layer sweep** — pro vrstvu N-1, N-2, ..., 0:
    ///    - Restore snapshot stavu vrstvy
    ///    - Inject `init_state[i]` (Var s autograd) do `ssm_state`
    ///    - Re-forward vrstvy s saved input jako fresh Var leaf
    ///    - Synthetic loss = `sum(output * grad_target)`
    ///    - `synth.backward()` → gradient pro init_state[i] + gradient
    ///      pro saved input (= cíl pro chunk i-1)
    ///    - Akumulujeme gradient pro init_state[i] do globálního GradStore
    ///
    /// **Halt na NaN:** kterýkoli chunk může vrátit NaN gradient. Detekujeme
    /// při akumulaci a vracíme Err.
    pub fn forward_backward_checkpointed(
        &self,
        stack: &CoreMemoryStack,
        input_ids: &Tensor,
    ) -> Result<(f64, GradStore)> {
        let dtype = self.dtype_ref();
        let model = self.model_ref();
        let n_layers = model.num_layers();

        if stack.num_layers() != n_layers {
            return Err(anyhow!(
                "CoreMemoryStack má {} vrstev, model má {}",
                stack.num_layers(),
                n_layers
            ));
        }

        // === Phase 1: no-grad forward, save inputs + state snapshots ===
        let mut state = self.new_model_state()?;

        // Layer inputs: index 0 = embedding output, index i = vstup vrstvy i.
        // Délka po phase 1 = n_layers + 1 (embedding + n vrstev).
        let mut layer_inputs: Vec<Tensor> = Vec::with_capacity(n_layers + 1);

        // Snapshoty stavu PŘED každou vrstvou (pro restore v backward).
        let mut state_snapshots: Vec<LayerState> = Vec::with_capacity(n_layers);

        // Embedding (no autograd needed — embed_tokens není Var).
        let mut x = model.embed(input_ids)?.detach();
        layer_inputs.push(x.clone());

        for i in 0..n_layers {
            // Snapshot stavu vrstvy před injekcí + scan.
            state_snapshots.push(state.layers[i].snapshot()?);

            // Injekce init_state[i] (detached, no autograd v forward sweep).
            let init = stack.layers[i]
                .init_state
                .as_tensor()
                .detach()
                .to_dtype(dtype)?;
            state.layers[i].ssm_state = init;

            // Forward vrstvy bez autograd (input už je detached).
            x = model.forward_layer(i, &x, 0, &mut state)?.detach();
            layer_inputs.push(x.clone());
        }

        // === Phase 2: final chunk (norm + lm_head + cross_entropy) ===
        // Make `x` a fresh Var leaf, abychom dostali gradient na něj.
        let last_hidden_var = Var::from_tensor(&x)?;
        let logits = model.final_head(last_hidden_var.as_tensor())?;
        let loss = cross_entropy_next_token(&logits, input_ids)
            .map_err(|e| anyhow!("cross_entropy: {e}"))?;
        let loss_val: f64 = loss.to_scalar::<f32>()? as f64;
        if !loss_val.is_finite() {
            return Err(anyhow!("non-finite loss před backward: {loss_val}"));
        }
        let final_grads = loss.backward()?;
        let mut grad_target = final_grads
            .get(last_hidden_var.as_tensor())
            .ok_or_else(|| anyhow!("final chunk backward nevrátil gradient na last_hidden"))?
            .clone();

        // === Phase 3: reverse layer sweep ===
        // Akumulátor — GradStore inicializován z prvního chunku v loop,
        // pak doplňován `insert`. GradStore::new() je v Candle private,
        // proto re-use first chunk's store.
        let mut accum: Option<GradStore> = None;

        for i in (0..n_layers).rev() {
            // Restore stavu vrstvy (snapshot byl pořízen PŘED forward i).
            state.layers[i] = state_snapshots[i].snapshot()?;

            // Var pro vstup vrstvy (saved tensor → fresh Var leaf).
            let x_in_var = Var::from_tensor(&layer_inputs[i])?;

            // Inject init_state Var (autograd ON — to_dtype je tracked op).
            let init_tracked = stack.layers[i].init_state.as_tensor().to_dtype(dtype)?;
            state.layers[i].ssm_state = init_tracked;

            // Re-forward vrstvy s autograd.
            let x_out = model.forward_layer(i, x_in_var.as_tensor(), 0, &mut state)?;

            // Synthetic loss = sum(x_out * grad_target).
            //
            // grad_target může mít jiný dtype než x_out (loss.backward dává
            // F32 grady, vrstva běží v BF16 na CUDA). Cast na common dtype
            // — F32 pro precision, pak skalar je F32.
            let gt = grad_target.to_dtype(candle_core::DType::F32)?;
            let xo = x_out.to_dtype(candle_core::DType::F32)?;
            let synth = xo.mul(&gt)?.sum_all()?;
            let chunk_grads = synth.backward()?;

            // Gradient pro x_in (cíl pro chunk i-1).
            grad_target = chunk_grads
                .get(x_in_var.as_tensor())
                .ok_or_else(|| anyhow!("chunk {i} backward nevrátil gradient na x_in"))?
                .clone();

            // Gradient pro init_state[i] — akumulovat do `accum`.
            let var_tensor = stack.layers[i].init_state.as_tensor();
            let chunk_grad_for_var = chunk_grads.get(var_tensor).cloned();

            // Akumulátor: poprvé převezmem chunk_grads jako základ,
            // dál insertujem do něj.
            match accum.as_mut() {
                None => {
                    // První chunk — použij chunk_grads jako základ a smaž
                    // z něj všechno, co nemá patřit (např. grad na x_in_var).
                    let mut base = chunk_grads;
                    base.remove(x_in_var.as_tensor());
                    if let Some(ref g) = chunk_grad_for_var {
                        check_finite(g, &format!("grad init_state[{i}]"))?;
                    }
                    accum = Some(base);
                }
                Some(store) => {
                    if let Some(g) = chunk_grad_for_var {
                        check_finite(&g, &format!("grad init_state[{i}]"))?;
                        store.insert(var_tensor, g);
                    }
                }
            }
        }

        let final_store = accum.ok_or_else(|| {
            anyhow!("checkpointed backward: accum store nebyl inicializován (n_layers=0?)")
        })?;
        Ok((loss_val, final_store))
    }

    /// Reference na underlying `FalconH1Model` — interní accessor pro
    /// chunked checkpointing (potřebuje per-layer forward).
    fn model_ref(&self) -> &crate::falcon_h1::model::FalconH1Model {
        &self.model
    }
}

/// Validace: tensor neobsahuje NaN/Inf. Pokud ano, vrací Err s contextem.
fn check_finite(t: &Tensor, label: &str) -> Result<()> {
    let abs_max: f32 = t
        .abs()?
        .max_keepdim(0)?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()
        .unwrap_or(f32::NAN);
    if !abs_max.is_finite() {
        return Err(anyhow!("{label}: non-finite gradient (max_abs={abs_max})"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn dev_model_path() -> Option<PathBuf> {
        let p = PathBuf::from("/home/lvx/Models/falcon-h1-1.5b-instruct");
        if p.exists() { Some(p) } else { None }
    }

    /// Smoke test: chunked forward+backward na 1.5B CPU F32, krátký
    /// seq_len. Pokud loss konečné a accum obsahuje nenulové gradienty,
    /// implementace funguje. Skipuje se bez modelu.
    #[test]
    fn checkpointed_forward_backward_runs_on_short_seq() -> Result<()> {
        let Some(model_path) = dev_model_path() else {
            eprintln!("Skipping — no local model");
            return Ok(());
        };

        let sofie = Sofie::load(&model_path, false, None)?;
        let stack = CoreMemoryStack::randn_small(sofie.config(), sofie.device_ref())?;

        // Jednoduchý input: 4 tokeny, batch=1.
        let input_ids = Tensor::new(&[[1u32, 2, 3, 4]], sofie.device_ref())?;

        let (loss, grads) = sofie.forward_backward_checkpointed(&stack, &input_ids)?;
        assert!(loss.is_finite(), "loss must be finite, got {loss}");

        let mut nonzero_count = 0;
        for var in stack.vars() {
            if let Some(g) = grads.get(var.as_tensor()) {
                let abs_max: f32 = g.abs()?.flatten_all()?.max(0)?.to_scalar()?;
                if abs_max > 0.0 {
                    nonzero_count += 1;
                }
            }
        }
        assert!(
            nonzero_count > 0,
            "alespoň jedna vrstva musí mít nenulový gradient"
        );
        Ok(())
    }
}
