//! Smoke test pro Core Memory training — ověří, že autograd teče od loss
//! zpět k trainable `Var` reprezentující initial SSM state.
//!
//! **Co smoke test dělá:**
//! 1. Načte Falcon-H1 (frozen váhy, BF16 GPU nebo F32 CPU)
//! 2. Vytvoří `CoreMemory` s náhodnou inicializací (zero init by dal zero
//!    gradient na vstupu kvůli multiplikativní SSM rekurzi)
//! 3. Vloží trainable Var (upcast na runtime dtype) do `ModelState.layers[0].ssm_state`
//! 4. Forward pass na malém dummy vstupu (N tokenů)
//! 5. Dummy loss = mean squared logits (skalár)
//! 6. AdamW backward_step — vytvoří gradient, aplikuje ho na Var
//! 7. Reportuje: gradient L2 norm, loss před/po, wall time, delta init_state
//!
//! **Co smoke NEDĚLÁ:** žádný dataset, žádný epoch, jen jedno forward+backward+step.
//!
//! **Úspěch:** `gradient_norm > 0` znamená, že gradient dotekl k Var přes
//! celou forward sekvenci (token embedding → N vrstev attention/SSM → LM head
//! → loss → backward). To je důkaz, že plánovaný state tuning workflow
//! (`reference_candle_backprop.md`) bude fungovat.

use anyhow::{Result, anyhow};
use candle_core::Tensor;
use candle_nn::optim::Optimizer;
use candle_nn::{AdamW, ParamsAdamW};

use crate::Sofie;
use crate::training::core_memory::CoreMemory;

/// Výsledek smoke testu — metriky a diagnostika.
#[derive(Debug, Clone)]
pub struct SmokeTrainResult {
    /// L2 norma initial_state před krokem.
    pub init_state_norm_before: f64,
    /// L2 norma initial_state po kroku.
    pub init_state_norm_after: f64,
    /// Delta L2 norma (init_state_after - init_state_before).
    pub init_state_delta_norm: f64,
    /// L2 norma gradientu aplikovaného na init_state.
    pub gradient_norm: f64,
    /// Hodnota loss před krokem.
    pub loss_value: f64,
    /// Wall-clock čas celého cyklu (forward + backward + step) v ms.
    pub wall_time_ms: u128,
    /// Počet tokenů ve vstupu.
    pub seq_len: usize,
    /// Index trénované vrstvy.
    pub layer_idx: usize,
}

impl SmokeTrainResult {
    /// Smoke test prošel = gradient není zanedbatelný a init_state se reálně
    /// pohnul. Práh 1e-8 pro robustnost vůči numerické šumové podlaze.
    pub fn passed(&self) -> bool {
        self.gradient_norm > 1e-8 && self.init_state_delta_norm > 1e-8
    }
}

impl Sofie {
    /// Spustí smoke test pro Core Memory — jedna iterace forward + backward + step.
    /// Ověřuje, že autograd teče přes Falcon-H1 k trainable initial SSM state.
    ///
    /// - `seq_len` — počet tokenů ve vstupu (použij malé číslo, ~10, pro minimum
    ///   peak VRAM)
    /// - `layer_idx` — index Mamba-2 vrstvy, jejíž initial state trénujeme
    ///   (pro alpha.1 obvykle 0)
    /// - `learning_rate` — AdamW lr (pro smoke stačí 1.0 — viz RWKV doporučení)
    pub fn smoke_train_core_memory(
        &self,
        seq_len: usize,
        layer_idx: usize,
        learning_rate: f64,
    ) -> Result<SmokeTrainResult> {
        self.smoke_train_core_memory_impl(seq_len, layer_idx, learning_rate, None)
    }

    /// Varianta smoke testu, která **zastaví forward po vrstvě `cut_at_layer`**
    /// a loss počítá z hidden stream místo z logits. Umožňuje izolovat backward
    /// path na `[layer_idx ..= cut_at_layer]` vrstev, užitečné pro hledání
    /// konkrétní op, která v backward produkuje NaN/Inf.
    ///
    /// `cut_at_layer` musí být ≥ `layer_idx` (jinak gradient nedoteče k init_state).
    pub fn smoke_train_core_memory_cut(
        &self,
        seq_len: usize,
        layer_idx: usize,
        learning_rate: f64,
        cut_at_layer: usize,
    ) -> Result<SmokeTrainResult> {
        if cut_at_layer < layer_idx {
            return Err(anyhow!(
                "cut_at_layer={} musí být >= layer_idx={}",
                cut_at_layer,
                layer_idx
            ));
        }
        self.smoke_train_core_memory_impl(seq_len, layer_idx, learning_rate, Some(cut_at_layer))
    }

    fn smoke_train_core_memory_impl(
        &self,
        seq_len: usize,
        layer_idx: usize,
        learning_rate: f64,
        cut_at_layer: Option<usize>,
    ) -> Result<SmokeTrainResult> {
        let start = std::time::Instant::now();

        // 1) Vytvoř trainable Core Memory s malou náhodnou inicializací.
        //    Nulová inicializace by dala zero gradient (h=0, dB⊗x → 0·...).
        let core = CoreMemory::randn_small(self.config(), self.device_ref(), layer_idx)
            .map_err(|e| anyhow!("CoreMemory::randn_small: {e}"))?;

        // 2) Dummy input — token IDs 1..=seq_len (garantovaně valid pro vocab > seq_len).
        let input_ids: Vec<u32> = (1..=seq_len as u32).collect();
        let input_tensor = Tensor::new(input_ids.as_slice(), self.device_ref())?.unsqueeze(0)?;

        // 3) Čerstvý ModelState, override layer_idx.ssm_state trainable Var (upcast na dtype).
        let mut state = self.new_model_state()?;
        let trained_tensor = core.init_state.as_tensor().to_dtype(self.dtype_ref())?;
        state.layers[layer_idx].ssm_state = trained_tensor;

        // L2 norma před krokem (scalar, F32).
        let init_state_norm_before = tensor_l2_norm(core.init_state.as_tensor())?;

        // 4) Forward pass — plný (logits) nebo cut (hidden).
        let activation = match cut_at_layer {
            None => self.model_forward(&input_tensor, 0, &mut state)?,
            Some(cut) => self.model_forward_up_to_layer(&input_tensor, 0, &mut state, cut)?,
        };

        // 5) Dummy loss — jeden element aktivace (pozice [0, 0, 0]).
        //    Gradient = 1 na ten element, 0 jinde. Minimální fan-in.
        let act_f32 = activation.to_dtype(candle_core::DType::F32)?;
        let loss = act_f32
            .narrow(0, 0, 1)?
            .narrow(1, 0, 1)?
            .narrow(2, 0, 1)?
            .sum_all()?;
        let loss_value: f64 = loss.to_scalar::<f32>()? as f64;
        if !loss_value.is_finite() {
            return Err(anyhow!(
                "loss je {} před backward — forward pass je numericky nestabilní",
                loss_value
            ));
        }

        // 6) AdamW backward_step — backward() + step() v jednom.
        let mut opt = AdamW::new(
            vec![core.init_state.clone()],
            ParamsAdamW {
                lr: learning_rate,
                ..ParamsAdamW::default()
            },
        )?;

        // Backward store pro diagnostiku gradientu (separate call — AdamW konzumuje loss).
        let grads = loss.backward()?;
        let grad_tensor = grads.get(core.init_state.as_tensor()).ok_or_else(|| {
            anyhow!("gradient pro init_state_var nebyl vytvořen — autograd neprotékl")
        })?;
        let gradient_norm = tensor_l2_norm(grad_tensor)?;

        // Numerická stabilita — NaN/Inf v gradientu znamená, že něco přeteklo
        // v backward pass (viz alpha.1 kde sqr.mean akumulovala do Inf→NaN).
        if !gradient_norm.is_finite() {
            return Err(anyhow!(
                "gradient L2 norm je {:.3e} (NaN/Inf) — numerická exploze v backward pass. \
                 Zkus menší `learning_rate`, kratší `seq_len`, nebo upcast problematických ops na F32.",
                gradient_norm
            ));
        }

        // Optimalizační krok — použije gradienty z backward() výše.
        opt.step(&grads)?;

        // L2 norma po kroku + delta.
        let init_state_norm_after = tensor_l2_norm(core.init_state.as_tensor())?;
        let init_state_delta_norm = (init_state_norm_after - init_state_norm_before).abs();

        let wall_time_ms = start.elapsed().as_millis();

        Ok(SmokeTrainResult {
            init_state_norm_before,
            init_state_norm_after,
            init_state_delta_norm,
            gradient_norm,
            loss_value,
            wall_time_ms,
            seq_len,
            layer_idx,
        })
    }
}

/// L2 norma tensoru (sqrt sum of squares), jako f64 pro reporting.
fn tensor_l2_norm(t: &Tensor) -> Result<f64> {
    let t_f32 = t.to_dtype(candle_core::DType::F32)?;
    let sum_sq: f32 = t_f32.sqr()?.sum_all()?.to_scalar()?;
    Ok((sum_sq as f64).sqrt())
}
