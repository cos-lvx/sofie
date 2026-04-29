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
//! ## Sub-layer chunking (alpha.13)
//!
//! Každá vrstva je rozdělena na **2 sub-chunky** podle topologie
//! Falcon-H1 hybrid Mamba+Attention:
//!
//! - **Chunk α (branches):** `x → res1 = x + ssm_branch + attention_branch`.
//!   Drží pre_norm + parallel SSM/attention. Při re-forward backward
//!   teče přes parallel branche + skip connection automaticky (autograd
//!   sčítá residual paths).
//! - **Chunk β (mlp):** `res1 → x_out = res1 + mlp(post_norm(res1))`.
//!   Drží post_norm + SwiGLU MLP + druhý residual.
//!
//! Memory peak per layer během backward = max(α activations, β activations)
//! místo sum. MLP intermediate (4608 × seq) byl dominantní v alpha.12,
//! teď nesoužije s SSM scan + attention QKV.
//!
//! Init_state[i] Var je tracked **jen v chunku α** (Mamba scan používá
//! init_state). Chunk β nepoužívá init_state, takže synthetic loss tam
//! nedopočítává `d_init_state`.

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

        // Diagnostický probe — povolen přes `ELEUTHERIA_CHECKPOINT_DEBUG=1`.
        // Volá `nvidia-smi` před každou fází + per N vrstev. Pomalé (~50ms
        // per call) ale poskytuje absolute reading volné GPU paměti, což
        // je k nezaplacení při debugu OOM regrese.
        let debug_mem = std::env::var("ELEUTHERIA_CHECKPOINT_DEBUG").is_ok();
        let log_gpu = |label: &str| {
            if !debug_mem {
                return;
            }
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args([
                    "--query-gpu=memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ])
                .output()
                && let Ok(s) = std::str::from_utf8(&output.stdout)
            {
                tracing::info!("checkpoint::{label} GPU: {}", s.trim());
            }
        };
        macro_rules! mem_probe {
            ($label:expr) => {
                if debug_mem {
                    log_gpu($label);
                }
            };
        }
        mem_probe!("phase 1 begin");

        // === Phase 1: no-grad forward, save inputs + sub-chunk outputs + state snapshots ===
        let mut state = self.new_model_state()?;

        // Layer inputs: index 0 = embedding output, index i = vstup vrstvy i.
        // Délka po phase 1 = n_layers + 1 (embedding + n vrstev).
        let mut layer_inputs: Vec<Tensor> = Vec::with_capacity(n_layers + 1);

        // Sub-chunk α output (residual_1) per vrstva — saved jako vstup pro chunk β.
        let mut layer_res1: Vec<Tensor> = Vec::with_capacity(n_layers);

        // Snapshoty stavu PŘED každou vrstvou (pro restore v backward).
        let mut state_snapshots: Vec<LayerState> = Vec::with_capacity(n_layers);

        // Embedding (no autograd needed — embed_tokens není Var).
        let mut x = model.embed(input_ids)?.detach();
        layer_inputs.push(x.clone());

        for i in 0..n_layers {
            if i.is_multiple_of(8) {
                mem_probe!(&format!("phase 1 layer {i}"));
            }
            // Snapshot stavu vrstvy před injekcí + scan.
            state_snapshots.push(state.layers[i].snapshot()?);

            // Injekce init_state[i] (detached, no autograd v forward sweep).
            let init = stack.layers[i]
                .init_state
                .as_tensor()
                .detach()
                .to_dtype(dtype)?;
            state.layers[i].ssm_state = init;

            // Sub-chunk α: x → res1 (Mamba scan + attention).
            let res1 = model.forward_layer_branches(i, &x, 0, &mut state)?.detach();
            layer_res1.push(res1.clone());

            // Sub-chunk β: res1 → x_out (post_norm + MLP + residual2).
            x = model.forward_layer_mlp(i, &res1, &mut state)?.detach();
            layer_inputs.push(x.clone());
        }
        mem_probe!("phase 1 done, entering phase 2");

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
        // loss.backward → grad_target. Pak DROPNI final_grads + loss +
        // last_hidden_var ihned (drží Arc references na intermediate tensory
        // lm_head + cross_entropy graph, ~700 MB na CUDA).
        let mut grad_target = {
            let final_grads = loss.backward()?;
            let g = final_grads
                .get(last_hidden_var.as_tensor())
                .ok_or_else(|| anyhow!("final chunk backward nevrátil gradient na last_hidden"))?
                .clone();
            drop(final_grads);
            g
        };
        drop(loss);
        drop(last_hidden_var);

        // === Phase 3: reverse layer sweep s sub-chunk granularitou ===
        // Pro každou vrstvu i (od N-1 dolů na 0):
        //   1. **β reverse:** Var leaf = saved res1, re-forward post_norm+MLP
        //      → x_out, synth = sum(x_out * grad_target), backward → d_res1.
        //   2. **α reverse:** Var leaf = saved x_in, init_state[i] tracked,
        //      re-forward pre_norm+SSM+attention → res1, synth = sum(res1 *
        //      d_res1), backward → d_x_in (target pro chunk i-1) + d_init_state.
        //
        // Helper `phase3_layer_reverse` drží intermediate tensors v lokálním
        // scope — Rust drop uvolní GPU memory mezi vrstvami (jinak Mamba
        // scan workspace akumuluje a OOM padá kolem vrstvy 7-8).
        //
        // GradStore::new() je private v Candle, takže `init_grads` shromažďuje
        // gradienty samostatně a inkrementálně se vkládají do `final_grads`
        // (existující GradStore z Phase 2 loss.backward, pro re-use).
        let mut init_grads: Vec<Option<Tensor>> = vec![None; n_layers];
        // Phase 3 první iterace produkuje GradStore z chunk α backward —
        // ten reusujeme jako akumulátor (Candle GradStore::new je private).
        let mut accum_store: Option<GradStore> = None;

        mem_probe!("phase 3 begin");
        // Reverse iteration: layer N-1 → 0. Konzumujeme saved tensors v
        // tomto pořadí — `mem::replace` umožňuje drop GPU storage Arc po
        // dokončení iterace (jinak vec drží reference po celou Phase 3).
        for i in (0..n_layers).rev() {
            mem_probe!(&format!("phase 3 layer {i}"));

            let saved_x_in = std::mem::replace(
                &mut layer_inputs[i],
                Tensor::zeros(0, dtype, self.device_ref())?,
            );
            let saved_res1 = std::mem::replace(
                &mut layer_res1[i],
                Tensor::zeros(0, dtype, self.device_ref())?,
            );
            let snapshot = std::mem::replace(
                &mut state_snapshots[i],
                LayerState::new(self.config(), dtype, self.device_ref())?,
            );

            let (new_grad_target, init_grad, store_for_accum) = phase3_layer_reverse(
                self,
                model,
                &mut state,
                &snapshot,
                &stack.layers[i].init_state,
                &saved_x_in,
                &saved_res1,
                i,
                &grad_target,
                dtype,
                accum_store.is_none(),
            )?;

            grad_target = new_grad_target;

            if let Some(g) = init_grad {
                check_finite(&g, &format!("grad init_state[{i}]"))?;
                init_grads[i] = Some(g);
            }

            // První iterace: store_for_accum obsahuje GradStore z chunk α
            // (po cleanup x_in_var entry). Use jako accum base.
            if let Some(s) = store_for_accum {
                accum_store = Some(s);
            }
        }

        let mut final_store = accum_store
            .ok_or_else(|| anyhow!("checkpointed backward: accum store nebyl inicializován"))?;
        for (i, layer_grad) in init_grads.into_iter().enumerate() {
            if let Some(g) = layer_grad {
                final_store.insert(stack.layers[i].init_state.as_tensor(), g);
            }
        }

        Ok((loss_val, final_store))
    }

    /// Reference na underlying `FalconH1Model` — interní accessor pro
    /// chunked checkpointing (potřebuje per-layer forward).
    fn model_ref(&self) -> &crate::falcon_h1::model::FalconH1Model {
        &self.model
    }
}

/// Phase 3 reverse pass pro **jednu** vrstvu — sub-chunk β následovaný α.
///
/// Drží všechny intermediate tensors (synth losses, GradStores, re-forward
/// activations) v lokálním scope. Po returnu se autograd grafy dropnou,
/// což na CUDA *kritické* — bez explicit drop scope Mamba scan workspace
/// akumuluje napříč iteracemi a peak memory roste lineárně s vrstvou indexem.
///
/// **Vrací:** `(d_x_in, init_state_grad, accum_seed)` — gradient pro vstup
/// vrstvy (target pro chunk i-1), opcionální gradient pro `init_state[i]`
/// a — pokud `produce_accum_seed = true` — clean GradStore z chunk α
/// (s odstraněným `x_in_var` entry) pro použití jako akumulátor v Phase 3.
#[allow(clippy::too_many_arguments)]
fn phase3_layer_reverse(
    sofie: &Sofie,
    model: &crate::falcon_h1::model::FalconH1Model,
    state: &mut crate::falcon_h1::state::ModelState,
    snapshot: &LayerState,
    init_state_var: &Var,
    saved_x_in: &Tensor,
    saved_res1: &Tensor,
    layer_idx: usize,
    grad_target: &Tensor,
    dtype: candle_core::DType,
    produce_accum_seed: bool,
) -> Result<(Tensor, Option<Tensor>, Option<GradStore>)> {
    let _ = sofie;

    // ----- Sub-chunk β: res1 → x_out -----
    let d_res1 = {
        state.layers[layer_idx] = snapshot.snapshot()?;

        let res1_var = Var::from_tensor(saved_res1)?;
        let x_out = model.forward_layer_mlp(layer_idx, res1_var.as_tensor(), state)?;

        let gt = grad_target.to_dtype(candle_core::DType::F32)?;
        let xo = x_out.to_dtype(candle_core::DType::F32)?;
        let synth_b = xo.mul(&gt)?.sum_all()?;
        let grads_b = synth_b.backward()?;

        grads_b
            .get(res1_var.as_tensor())
            .ok_or_else(|| {
                anyhow!("vrstva {layer_idx} chunk β backward nevrátil gradient na res1")
            })?
            .clone()
    };

    // ----- Sub-chunk α: x_in → res1 -----
    let result = {
        state.layers[layer_idx] = snapshot.snapshot()?;

        let x_in_var = Var::from_tensor(saved_x_in)?;
        let init_tracked = init_state_var.as_tensor().to_dtype(dtype)?;
        state.layers[layer_idx].ssm_state = init_tracked;

        let res1_recomputed =
            model.forward_layer_branches(layer_idx, x_in_var.as_tensor(), 0, state)?;

        let dr = d_res1.to_dtype(candle_core::DType::F32)?;
        let r1 = res1_recomputed.to_dtype(candle_core::DType::F32)?;
        let synth_a = r1.mul(&dr)?.sum_all()?;
        let mut grads_a = synth_a.backward()?;

        let d_x_in = grads_a
            .get(x_in_var.as_tensor())
            .ok_or_else(|| {
                anyhow!("vrstva {layer_idx} chunk α backward nevrátil gradient na x_in")
            })?
            .clone();
        let init_grad = grads_a.get(init_state_var.as_tensor()).cloned();

        let accum_seed = if produce_accum_seed {
            // Drop nepotřebné entries — `x_in_var` je ephemeral leaf, ne
            // trainable Var. Init_state grad pak `final_store.insert` vloží
            // nazpět při finalize.
            grads_a.remove(x_in_var.as_tensor());
            grads_a.remove(init_state_var.as_tensor());
            Some(grads_a)
        } else {
            None
        };

        (d_x_in, init_grad, accum_seed)
    };

    Ok(result)
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
