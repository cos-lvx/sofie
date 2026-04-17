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
use crate::falcon_h1::layer::LayerStop;
use crate::training::core_memory::{CoreMemory, CoreMemoryStack};
use crate::training::loss::cross_entropy_next_token;
use crate::training::trace::{self, TraceEntry};

/// Výsledek smoke testu — metriky a diagnostika.
#[derive(Debug, Clone)]
pub struct SmokeTrainResult {
    /// L2 norma initial_state před krokem.
    pub init_state_norm_before: f64,
    /// L2 norma initial_state po kroku.
    pub init_state_norm_after: f64,
    /// Delta L2 norma (init_state_after - init_state_before).
    pub init_state_delta_norm: f64,
    /// L2 norma gradientu aplikovaného na init_state (po případném clippingu).
    pub gradient_norm: f64,
    /// Pre-clip L2 norma gradientu (pro monitoring). Pokud gradient nebyl
    /// clippován, rovná se `gradient_norm`.
    pub pre_clip_gradient_norm: f64,
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
        self.smoke_train_core_memory_impl(
            seq_len,
            layer_idx,
            learning_rate,
            None,
            None,
            LayerStop::Full,
            false,
        )
        .map(|(r, _)| r)
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
        self.smoke_train_core_memory_impl(
            seq_len,
            layer_idx,
            learning_rate,
            Some(cut_at_layer),
            None,
            LayerStop::Full,
            false,
        )
        .map(|(r, _)| r)
    }

    /// Smoke test s **gradient clippingem** — standardní Mamba-2 recept
    /// (`max_grad_norm=1.0`). Pokud je Peri-LN massive activations root cause
    /// NaN gradientu v backward, clipping ho odblokuje.
    pub fn smoke_train_core_memory_clipped(
        &self,
        seq_len: usize,
        layer_idx: usize,
        learning_rate: f64,
        cut_at_layer: Option<usize>,
        max_grad_norm: f64,
    ) -> Result<SmokeTrainResult> {
        if let Some(cut) = cut_at_layer
            && cut < layer_idx
        {
            return Err(anyhow!(
                "cut_at_layer={} musí být >= layer_idx={}",
                cut,
                layer_idx
            ));
        }
        self.smoke_train_core_memory_impl(
            seq_len,
            layer_idx,
            learning_rate,
            cut_at_layer,
            Some(max_grad_norm),
            LayerStop::Full,
            false,
        )
        .map(|(r, _)| r)
    }

    /// Smoke test se sub-layer cut-at-component na poslední trénované vrstvě.
    /// Umožňuje bisect uvnitř jedné vrstvy — např. `--cut-at-layer 23
    /// --cut-at-component after-ssm` zastaví forward po SSM branch na layer 22
    /// (index 23 znamená vrstvy 0..=23, sub-stop se vztahuje k té poslední).
    ///
    /// Volitelně `enable_trace=true` aktivuje forward tensor stats sink a
    /// vrátí seznam entries vedle `SmokeTrainResult`.
    #[allow(clippy::too_many_arguments)]
    pub fn smoke_train_core_memory_component(
        &self,
        seq_len: usize,
        layer_idx: usize,
        learning_rate: f64,
        cut_at_layer: Option<usize>,
        max_grad_norm: Option<f64>,
        stop: LayerStop,
        enable_trace: bool,
    ) -> Result<(SmokeTrainResult, Option<Vec<TraceEntry>>)> {
        if let Some(cut) = cut_at_layer
            && cut < layer_idx
        {
            return Err(anyhow!(
                "cut_at_layer={} musí být >= layer_idx={}",
                cut,
                layer_idx
            ));
        }
        self.smoke_train_core_memory_impl(
            seq_len,
            layer_idx,
            learning_rate,
            cut_at_layer,
            max_grad_norm,
            stop,
            enable_trace,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn smoke_train_core_memory_impl(
        &self,
        seq_len: usize,
        layer_idx: usize,
        learning_rate: f64,
        cut_at_layer: Option<usize>,
        max_grad_norm: Option<f64>,
        stop: LayerStop,
        enable_trace: bool,
    ) -> Result<(SmokeTrainResult, Option<Vec<TraceEntry>>)> {
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

        // Aktivuj trace sink jen pokud caller chce. Forward pass naplní,
        // po forward ho odebereme — backward už neinstrumentujeme.
        if enable_trace {
            trace::start();
        }

        // 4) Forward pass — plný (logits) / cut bez sub-stop / cut s sub-stop.
        let activation = match (cut_at_layer, stop) {
            (None, _) => self.model_forward(&input_tensor, 0, &mut state)?,
            (Some(cut), LayerStop::Full) => {
                self.model_forward_up_to_layer(&input_tensor, 0, &mut state, cut)?
            }
            (Some(cut), sub_stop) => self.model_forward_up_to_layer_with_stop(
                &input_tensor,
                0,
                &mut state,
                cut,
                sub_stop,
            )?,
        };

        let trace_entries = if enable_trace { trace::finish() } else { None };

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
        let mut grads = loss.backward()?;
        let pre_clip_gradient_norm = match grads.get(core.init_state.as_tensor()) {
            Some(g) => tensor_l2_norm(g)?,
            None => {
                return Err(anyhow!(
                    "gradient pro init_state_var nebyl vytvořen — autograd neprotékl"
                ));
            }
        };

        // Volitelný gradient clipping (standardní Mamba-2 recept max_grad_norm=1.0).
        if let Some(max_norm) = max_grad_norm
            && pre_clip_gradient_norm.is_finite()
        {
            crate::training::clip::clip_grad_norm(&mut grads, &[&core.init_state], max_norm)?;
        }

        let grad_tensor = grads.get(core.init_state.as_tensor()).unwrap();
        let gradient_norm = tensor_l2_norm(grad_tensor)?;

        // NaN/Inf v gradientu NENÍ Err — vrátíme SmokeTrainResult s NaN
        // hodnotami a passed()=false. To umožňuje sweep přes mnoho konfigurací
        // i když některé selhávají (analýza patternu).
        let (init_state_norm_after, init_state_delta_norm) = if gradient_norm.is_finite() {
            opt.step(&grads)?;
            let n_after = tensor_l2_norm(core.init_state.as_tensor())?;
            (n_after, (n_after - init_state_norm_before).abs())
        } else {
            // Skip optimizer step — aplikovat NaN gradient by rozbilo Var.
            (f64::NAN, f64::NAN)
        };

        let wall_time_ms = start.elapsed().as_millis();

        let result = SmokeTrainResult {
            init_state_norm_before,
            init_state_norm_after,
            init_state_delta_norm,
            gradient_norm,
            pre_clip_gradient_norm,
            loss_value,
            wall_time_ms,
            seq_len,
            layer_idx,
        };
        Ok((result, trace_entries))
    }
}

// ---------------------------------------------------------------------------
// Helper — zpětně kompatibilní konstrukce SmokeTrainResult bez pre_clip (pro
// testy a starší callers, pokud by se přidali)
// ---------------------------------------------------------------------------

/// L2 norma tensoru (sqrt sum of squares), jako f64 pro reporting.
fn tensor_l2_norm(t: &Tensor) -> Result<f64> {
    let t_f32 = t.to_dtype(candle_core::DType::F32)?;
    let sum_sq: f32 = t_f32.sqr()?.sum_all()?.to_scalar()?;
    Ok((sum_sq as f64).sqrt())
}

// ---------------------------------------------------------------------------
// Diagnostický sweep — pro binary search NaN zdroje v backward
// ---------------------------------------------------------------------------

impl Sofie {
    /// Změř L2 normu hidden stream po každé vrstvě (pro analýzu forward
    /// amplifikace). Vrací `Vec<f64>` délky `num_hidden_layers` —
    /// `result[i]` je norm aktivace po vrstvě i.
    ///
    /// Inicializuje fresh state (nulové SSM), spouští forward_up_to_layer
    /// v cyklu. Drahé (N²/2 ops celkem) ale čistě diagnostické.
    pub fn measure_forward_hidden_norms(&self, seq_len: usize) -> Result<Vec<f64>> {
        let input_ids: Vec<u32> = (1..=seq_len as u32).collect();
        let input_tensor = Tensor::new(input_ids.as_slice(), self.device_ref())?.unsqueeze(0)?;

        let num_layers = self.config().num_hidden_layers;
        let mut norms = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            let mut state = self.new_model_state()?;
            let hidden = self.model_forward_up_to_layer(&input_tensor, 0, &mut state, layer)?;
            norms.push(tensor_l2_norm(&hidden)?);
        }
        Ok(norms)
    }

    /// Sweep smoke testu přes rozsah `cut_at_layer` — pro fixní `layer_idx`
    /// spustí `smoke_train_core_memory_cut` postupně pro cut od `layer_idx`
    /// do `num_hidden_layers - 1`. Plus jeden běh bez cut (plný forward).
    ///
    /// Vrací vector `(cut_description, SmokeTrainResult)`. Results s NaN
    /// gradientem mají `passed()=false`, neselhávají sweep jako celek —
    /// umožňuje kompletní tabulku i přes některé fail body.
    pub fn smoke_sweep(
        &self,
        seq_len: usize,
        layer_idx: usize,
        learning_rate: f64,
    ) -> Result<Vec<(String, SmokeTrainResult)>> {
        let num_layers = self.config().num_hidden_layers;
        let mut out = Vec::new();

        for cut in layer_idx..num_layers {
            let desc = format!("cut={}", cut);
            let result =
                self.smoke_train_core_memory_cut(seq_len, layer_idx, learning_rate, cut)?;
            out.push((desc, result));
        }

        // Finální plný forward (cut=None, přes lm_head)
        let full = self.smoke_train_core_memory(seq_len, layer_idx, learning_rate)?;
        out.push(("cut=full (logits)".to_string(), full));

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Multi-layer smoke test (alpha.10+) — Vec<Var> pro všech N vrstev,
// cross-entropy loss na next-token prediction
// ---------------------------------------------------------------------------

/// Výsledek multi-layer smoke testu — agregát + per-layer gradient norms.
#[derive(Debug, Clone)]
pub struct MultiLayerSmokeResult {
    /// Hodnota cross-entropy loss před krokem (v ideálním případě blízko
    /// `ln(vocab_size)` pro random init, klesá k 0 pro perfect fit).
    pub loss_value: f64,
    /// Agregovaná L2 norma gradientu napříč všemi vrstvami
    /// (sqrt of sum of squared grad norms — globální L2).
    pub total_gradient_norm: f64,
    /// Pre-clip varianta `total_gradient_norm`.
    pub pre_clip_total_gradient_norm: f64,
    /// Per-vrstva L2 norm gradientu — `[layer_0_grad, layer_1_grad, ...]`.
    /// Užitečné pro analýzu: distribuuje se gradient rovnoměrně, nebo
    /// je koncentrovaný v některých vrstvách?
    pub per_layer_gradient_norms: Vec<f64>,
    /// Per-layer L2 norm init_state před krokem.
    pub per_layer_init_norms_before: Vec<f64>,
    /// Per-layer L2 norm init_state po kroku.
    pub per_layer_init_norms_after: Vec<f64>,
    /// Wall-clock čas celého cyklu (ms).
    pub wall_time_ms: u128,
    /// Seq_len vstupu.
    pub seq_len: usize,
    /// Počet trénovaných vrstev.
    pub num_layers: usize,
}

impl MultiLayerSmokeResult {
    /// Pass = všech per-layer gradientů je finite a alespoň průměrně
    /// nenulové. Jedna vrstva se zero gradientem je OK (může se stát
    /// pro některé pozice při krátkém seq_len), ale většina musí hořet.
    pub fn passed(&self) -> bool {
        let finite = self.total_gradient_norm.is_finite();
        let non_trivial_count = self
            .per_layer_gradient_norms
            .iter()
            .filter(|&&n| n.is_finite() && n > 1e-10)
            .count();
        finite && non_trivial_count * 2 >= self.num_layers // alespoň polovina
    }
}

impl Sofie {
    /// Multi-layer smoke test — trénuje `init_state` všech Mamba-2 vrstev
    /// najednou s cross-entropy loss na next-token prediction.
    ///
    /// **Rozdíly oproti single-layer `smoke_train_core_memory`:**
    /// - Trénovaný: `CoreMemoryStack` (Vec<Var>) místo jednoho Var
    /// - Loss: cross-entropy na next-token (realistic LM objective)
    ///   místo dummy single-element
    /// - Output: per-layer gradient norms (analýza distribuce)
    ///
    /// **Požadavek:** `seq_len >= 2` (cross-entropy potřebuje alespoň
    /// jeden next-token target).
    ///
    /// **VRAM pozor:** Pro 1.5B model s `seq_len=4` už jsme blízko 6 GB
    /// limitu (RTX 4050). Pro větší seq_len použij gradient accumulation
    /// (alpha.11+).
    pub fn smoke_train_core_memory_multilayer(
        &self,
        seq_len: usize,
        learning_rate: f64,
        max_grad_norm: Option<f64>,
    ) -> Result<MultiLayerSmokeResult> {
        if seq_len < 2 {
            return Err(anyhow!(
                "multi-layer smoke vyžaduje seq_len >= 2 pro next-token loss"
            ));
        }

        let start = std::time::Instant::now();
        let num_layers = self.config().num_hidden_layers;

        // 1) Vytvoř trainable Core Memory stack s malou náhodnou init
        //    (zero init by dal zero grad pro SSM rekurzi).
        let stack = CoreMemoryStack::randn_small(self.config(), self.device_ref())
            .map_err(|e| anyhow!("CoreMemoryStack::randn_small: {e}"))?;

        // 2) Dummy input sekvence — token IDs 1..=seq_len (valid pro vocab > seq_len)
        let input_ids: Vec<u32> = (1..=seq_len as u32).collect();
        let input_tensor = Tensor::new(input_ids.as_slice(), self.device_ref())?.unsqueeze(0)?;

        // 3) Čerstvý ModelState, injektuj všechny trainable init_states
        let mut state = self.new_model_state()?;
        stack
            .inject_into_state(&mut state, self.dtype_ref())
            .map_err(|e| anyhow!("inject_into_state: {e}"))?;

        // Baseline norms před krokem (per layer)
        let per_layer_init_norms_before: Vec<f64> = stack
            .layers
            .iter()
            .map(|c| tensor_l2_norm(c.init_state.as_tensor()))
            .collect::<Result<Vec<_>>>()?;

        // 4) Plný forward pass → logits [batch, seq_len, vocab]
        let logits = self.model_forward(&input_tensor, 0, &mut state)?;

        // 5) Cross-entropy next-token loss
        let loss = cross_entropy_next_token(&logits, &input_tensor)
            .map_err(|e| anyhow!("cross_entropy: {e}"))?;
        let loss_value: f64 = loss.to_scalar::<f32>()? as f64;
        if !loss_value.is_finite() {
            return Err(anyhow!(
                "loss je {loss_value} před backward — forward je nestabilní"
            ));
        }

        // 6) Backward — gradient store pro diagnostiku
        let mut grads = loss.backward()?;

        // Per-layer gradient norms + total
        let mut per_layer_gradient_norms: Vec<f64> = Vec::with_capacity(num_layers);
        let mut pre_clip_sum_sq = 0.0f64;
        for core in &stack.layers {
            let grad_norm = match grads.get(core.init_state.as_tensor()) {
                Some(g) => tensor_l2_norm(g)?,
                None => 0.0, // chybějící grad = žádný gradient signal
            };
            per_layer_gradient_norms.push(grad_norm);
            if grad_norm.is_finite() {
                pre_clip_sum_sq += grad_norm * grad_norm;
            } else {
                pre_clip_sum_sq = f64::NAN;
            }
        }
        let pre_clip_total_gradient_norm = if pre_clip_sum_sq.is_nan() {
            f64::NAN
        } else {
            pre_clip_sum_sq.sqrt()
        };

        // Volitelný gradient clipping (global L2 norm napříč všemi Vars)
        let vars_owned = stack.vars_owned();
        if let Some(max_norm) = max_grad_norm
            && pre_clip_total_gradient_norm.is_finite()
        {
            let var_refs: Vec<&candle_core::Var> = vars_owned.iter().collect();
            crate::training::clip::clip_grad_norm(&mut grads, &var_refs, max_norm)?;
        }

        // Post-clip total
        let total_gradient_norm = if pre_clip_total_gradient_norm.is_finite() {
            let mut sq = 0.0f64;
            for core in &stack.layers {
                if let Some(g) = grads.get(core.init_state.as_tensor()) {
                    let n = tensor_l2_norm(g)?;
                    sq += n * n;
                }
            }
            sq.sqrt()
        } else {
            f64::NAN
        };

        // 7) Optimizer step — pouze pokud gradient finite
        if total_gradient_norm.is_finite() {
            let mut opt = AdamW::new(
                vars_owned.clone(),
                ParamsAdamW {
                    lr: learning_rate,
                    ..ParamsAdamW::default()
                },
            )?;
            opt.step(&grads)?;
        }

        // Post-step norms (i když nebyl step — pak se rovnají before)
        let per_layer_init_norms_after: Vec<f64> = stack
            .layers
            .iter()
            .map(|c| tensor_l2_norm(c.init_state.as_tensor()))
            .collect::<Result<Vec<_>>>()?;

        let wall_time_ms = start.elapsed().as_millis();

        Ok(MultiLayerSmokeResult {
            loss_value,
            total_gradient_norm,
            pre_clip_total_gradient_norm,
            per_layer_gradient_norms,
            per_layer_init_norms_before,
            per_layer_init_norms_after,
            wall_time_ms,
            seq_len,
            num_layers,
        })
    }
}
