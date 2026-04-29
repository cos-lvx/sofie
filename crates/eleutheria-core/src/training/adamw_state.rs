//! `EleutheriaAdamW` — vlastní AdamW optimizer s **introspekcí state**.
//!
//! Re-implementace algoritmu z `candle-nn::AdamW` (`f526033/candle-nn/src/optim.rs`),
//! který má všechna důležitá pole privátní (`vars: Vec<VarAdamW>`, `step_t`).
//! Pro multi-stage tréninky (KI-007) potřebujeme číst i zapisovat:
//!
//! - `first_moment` (m) — exponenciálně klouzavá suma gradientů
//! - `second_moment` (v) — exponenciálně klouzavá suma `g²`
//! - `step_t` — globální counter (driver bias correction)
//!
//! Bez perzistence těchto tří položek **resume tréninku startuje s prázdným**
//! Adam state. Adam bias correction částečně kompenzuje warmup okno (~prvních
//! 100–500 stepů), ale velocity buffer naskakuje znova a produkuje overshoot
//! fázi (RN-002, RN-006). Plný state restore tuto regresi eliminuje.
//!
//! ## Numerická identita s Candle
//!
//! Step algoritmus je byte-identický s Candle implementací — stejné pořadí
//! operací, stejné konstanty. Test `step_matches_candle_for_one_step` v
//! `tests` to verifikuje.

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::ParamsAdamW;
use candle_nn::optim::Optimizer;

/// Per-Var AdamW state. Veřejný kvůli persistenci — externí kód
/// (`OptimizerArtifact`) potřebuje číst `first_moment`/`second_moment`
/// jako tensory pro safetensors zápis.
#[derive(Debug)]
pub struct VarAdamW {
    pub var: Var,
    pub first_moment: Var,
    pub second_moment: Var,
}

impl VarAdamW {
    /// Inicializuje per-Var state nulovými m, v se stejným shape/dtype/device
    /// jako trainable Var (matchuje Candle konstrukci).
    pub fn fresh(var: Var) -> Result<Self> {
        let dtype = var.dtype();
        let shape = var.shape().clone();
        let device = var.device().clone();
        let first_moment = Var::zeros(&shape, dtype, &device)?;
        let second_moment = Var::zeros(&shape, dtype, &device)?;
        Ok(Self {
            var,
            first_moment,
            second_moment,
        })
    }
}

/// AdamW optimizer s veřejným přístupem k state.
///
/// Algoritmicky identický s `candle_nn::AdamW`. Liší se v API:
/// - `state()` / `state_mut()` — read-only / mutable přístup ke `Vec<VarAdamW>`
/// - `step_t()` — counter
/// - `set_step_t(n)` — restore counter (pro `--resume-from`)
/// - `Optimizer` trait — dropin replacement v `train_core_memory`
#[derive(Debug)]
pub struct EleutheriaAdamW {
    vars: Vec<VarAdamW>,
    step_t: usize,
    params: ParamsAdamW,
}

impl Optimizer for EleutheriaAdamW {
    type Config = ParamsAdamW;

    fn new(vars: Vec<Var>, params: ParamsAdamW) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|v| v.dtype().is_float())
            .map(VarAdamW::fresh)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            params,
            step_t: 0,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    fn step(&mut self, grads: &GradStore) -> Result<()> {
        self.step_t += 1;
        let lr = self.params.lr;
        let lambda = self.params.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));
        for var in self.vars.iter() {
            let theta = &var.var;
            let m = &var.first_moment;
            let v = &var.second_moment;
            if let Some(g) = grads.get(theta) {
                let next_m = ((m.as_tensor() * beta1)? + (g * (1.0 - beta1))?)?;
                let next_v = ((v.as_tensor() * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
                let m_hat = (&next_m * scale_m)?;
                let v_hat = (&next_v * scale_v)?;
                let next_theta = (theta.as_tensor() * (1f64 - lr_lambda))?;
                let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
                let next_theta = (next_theta - (adjusted_grad * lr)?)?;
                m.set(&next_m)?;
                v.set(&next_v)?;
                theta.set(&next_theta)?;
            }
        }
        Ok(())
    }
}

impl EleutheriaAdamW {
    /// Konstrukce s default LR override (kompatibilita s Candle API).
    pub fn new_lr(vars: Vec<Var>, learning_rate: f64) -> Result<Self> {
        let params = ParamsAdamW {
            lr: learning_rate,
            ..ParamsAdamW::default()
        };
        Self::new(vars, params)
    }

    /// Aktuální AdamW parametry.
    pub fn params(&self) -> &ParamsAdamW {
        &self.params
    }

    /// Override všech parametrů (např. po resume pro adjust LR).
    pub fn set_params(&mut self, params: ParamsAdamW) {
        self.params = params;
    }

    /// Počet provedených `step()` volání. Používá se v bias correction
    /// `1/(1-β^t)`. Po `EleutheriaAdamW::new` je 0; po prvním stepu 1.
    pub fn step_t(&self) -> usize {
        self.step_t
    }

    /// Read-only přístup k per-Var state pro inspekci/persistenci.
    /// Délka odpovídá počtu **float** Var-ů předaných do `new` (non-float
    /// Var-y jsou vyfiltrovány stejně jako v Candle).
    pub fn state(&self) -> &[VarAdamW] {
        &self.vars
    }

    /// Snapshot per-Var moments jako CPU F32 tensory — připraveno pro
    /// safetensors zápis. Tuple `(first_moment, second_moment)` per Var,
    /// vrácené v pořadí trainable Var-ů (= `vars` slice).
    pub fn snapshot_moments(&self) -> Result<Vec<(Tensor, Tensor)>> {
        self.vars
            .iter()
            .map(|s| {
                let m = s
                    .first_moment
                    .as_tensor()
                    .to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?;
                let v = s
                    .second_moment
                    .as_tensor()
                    .to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?;
                Ok((m, v))
            })
            .collect()
    }

    /// Restore moments + step counter z dříve uloženého snapshotu.
    ///
    /// `moments` musí mít stejnou délku jako interní `vars` a každý
    /// tensor musí mít stejný shape jako odpovídající Var. Tensory se
    /// konvertují na runtime dtype/device dle Var.
    ///
    /// **Pozor:** volá se po `EleutheriaAdamW::new`, takže před prvním
    /// `step()`. Pokud je voláno mid-training, pre-existující m/v se
    /// přepíší — `step_t` se nastaví explicitně, žádný carry-over.
    pub fn restore_moments(&mut self, moments: &[(Tensor, Tensor)], step_t: usize) -> Result<()> {
        if moments.len() != self.vars.len() {
            return Err(candle_core::Error::Msg(format!(
                "restore_moments: očekávám {} per-Var snapshotů, dostal jsem {}",
                self.vars.len(),
                moments.len()
            )));
        }
        for (idx, (state, (m_src, v_src))) in self.vars.iter().zip(moments.iter()).enumerate() {
            let var_dtype = state.var.dtype();
            let var_device = state.var.device();
            let var_shape = state.var.shape().dims().to_vec();

            if m_src.shape().dims() != var_shape.as_slice() {
                return Err(candle_core::Error::Msg(format!(
                    "restore_moments[{idx}]: m shape {:?} ≠ var shape {:?}",
                    m_src.shape().dims(),
                    var_shape
                )));
            }
            if v_src.shape().dims() != var_shape.as_slice() {
                return Err(candle_core::Error::Msg(format!(
                    "restore_moments[{idx}]: v shape {:?} ≠ var shape {:?}",
                    v_src.shape().dims(),
                    var_shape
                )));
            }

            let m_runtime = m_src.to_dtype(var_dtype)?.to_device(var_device)?;
            let v_runtime = v_src.to_dtype(var_dtype)?.to_device(var_device)?;
            state.first_moment.set(&m_runtime)?;
            state.second_moment.set(&v_runtime)?;
        }
        self.step_t = step_t;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::AdamW;

    fn make_var(value: f32, device: &Device) -> Result<Var> {
        let t = Tensor::from_vec(vec![value, value * 2.0], (2,), device)?;
        Var::from_tensor(&t)
    }

    /// Bootstrap GradStore přes triviální `backward()` na dummy Var, pak
    /// přes veřejné `insert` overrideuje grad pro testovaný `var`.
    /// (`GradStore::new` je v Candle private, takže nelze konstruovat napřímo.)
    fn make_grads(var: &Var, grad_value: f32) -> Result<GradStore> {
        let dummy = Var::zeros((1,), DType::F32, var.device())?;
        let loss = dummy.as_tensor().sum_all()?;
        let mut grads = loss.backward()?;
        let g = Tensor::ones(var.shape(), var.dtype(), var.device())?;
        let g = (g * grad_value as f64)?;
        grads.insert(var.as_tensor(), g);
        Ok(grads)
    }

    /// Nový optimizer má step_t=0 a m, v všechno nulové.
    #[test]
    fn fresh_optimizer_has_zero_state() -> Result<()> {
        let var = make_var(1.0, &Device::Cpu)?;
        let opt = EleutheriaAdamW::new(vec![var], ParamsAdamW::default())?;
        assert_eq!(opt.step_t(), 0);
        assert_eq!(opt.state().len(), 1);
        let m: f32 = opt.state()[0]
            .first_moment
            .as_tensor()
            .sum_all()?
            .to_scalar()?;
        let v: f32 = opt.state()[0]
            .second_moment
            .as_tensor()
            .sum_all()?
            .to_scalar()?;
        assert_eq!(m, 0.0);
        assert_eq!(v, 0.0);
        Ok(())
    }

    /// Po prvním stepu se step_t inkrementuje a m, v už nejsou nulové.
    #[test]
    fn step_increments_step_t_and_updates_moments() -> Result<()> {
        let var = make_var(1.0, &Device::Cpu)?;
        let mut opt = EleutheriaAdamW::new(vec![var.clone()], ParamsAdamW::default())?;
        let grads = make_grads(&var, 0.5)?;
        opt.step(&grads)?;
        assert_eq!(opt.step_t(), 1);
        let m_sum: f32 = opt.state()[0]
            .first_moment
            .as_tensor()
            .sum_all()?
            .to_scalar()?;
        // m = β1 * 0 + (1-β1) * g = 0.1 * 0.5 = 0.05 per element, sum 0.1
        assert!((m_sum - 0.1).abs() < 1e-6);
        Ok(())
    }

    /// Step algoritmus produkuje přesně stejnou `var.set()` hodnotu jako
    /// `candle_nn::AdamW` pro identický grad.
    #[test]
    fn step_matches_candle_for_one_step() -> Result<()> {
        let device = Device::Cpu;
        let v_ours = make_var(1.0, &device)?;
        let v_candle = make_var(1.0, &device)?;

        let mut ours = EleutheriaAdamW::new(vec![v_ours.clone()], ParamsAdamW::default())?;
        let mut candle = AdamW::new(vec![v_candle.clone()], ParamsAdamW::default())?;

        let g_ours = make_grads(&v_ours, 0.5)?;
        let g_candle = make_grads(&v_candle, 0.5)?;
        ours.step(&g_ours)?;
        candle.step(&g_candle)?;

        let after_ours: Vec<f32> = v_ours.as_tensor().to_vec1()?;
        let after_candle: Vec<f32> = v_candle.as_tensor().to_vec1()?;
        for (a, b) in after_ours.iter().zip(after_candle.iter()) {
            assert!(
                (a - b).abs() < 1e-7,
                "ours={a}, candle={b} — step musí být numericky identický"
            );
        }
        Ok(())
    }

    /// Step algoritmus produkuje stejnou Var hodnotu i po více stepech (akumulace m, v).
    #[test]
    fn step_matches_candle_for_five_steps() -> Result<()> {
        let device = Device::Cpu;
        let v_ours = make_var(1.0, &device)?;
        let v_candle = make_var(1.0, &device)?;

        let mut ours = EleutheriaAdamW::new(vec![v_ours.clone()], ParamsAdamW::default())?;
        let mut candle = AdamW::new(vec![v_candle.clone()], ParamsAdamW::default())?;

        for step in 0..5 {
            let grad_value = 0.5 - 0.1 * step as f32; // klesající grad
            let g_ours = make_grads(&v_ours, grad_value)?;
            let g_candle = make_grads(&v_candle, grad_value)?;
            ours.step(&g_ours)?;
            candle.step(&g_candle)?;
        }

        let after_ours: Vec<f32> = v_ours.as_tensor().to_vec1()?;
        let after_candle: Vec<f32> = v_candle.as_tensor().to_vec1()?;
        for (a, b) in after_ours.iter().zip(after_candle.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "ours={a}, candle={b} po 5 stepech (akumulace musí ladit)"
            );
        }
        assert_eq!(ours.step_t(), 5);
        Ok(())
    }

    /// Snapshot/restore round-trip: 3 steps → snapshot → fresh opt → restore →
    /// další 2 steps. Výsledek shodný s 5 stepy bez restartu.
    #[test]
    fn snapshot_restore_round_trip_preserves_trajectory() -> Result<()> {
        let device = Device::Cpu;

        // Reference: 5 stepů bez restartu.
        let v_ref = make_var(1.0, &device)?;
        let mut opt_ref = EleutheriaAdamW::new(vec![v_ref.clone()], ParamsAdamW::default())?;
        for step in 0..5 {
            let grad_value = 0.5 - 0.1 * step as f32;
            let g = make_grads(&v_ref, grad_value)?;
            opt_ref.step(&g)?;
        }
        let ref_var: Vec<f32> = v_ref.as_tensor().to_vec1()?;

        // Test: 3 steps → snapshot → nová Var s identickou hodnotou + nový opt
        // → restore → další 2 steps.
        let v_test = make_var(1.0, &device)?;
        let mut opt_a = EleutheriaAdamW::new(vec![v_test.clone()], ParamsAdamW::default())?;
        for step in 0..3 {
            let grad_value = 0.5 - 0.1 * step as f32;
            let g = make_grads(&v_test, grad_value)?;
            opt_a.step(&g)?;
        }
        let snapshot = opt_a.snapshot_moments()?;
        let step_t_a = opt_a.step_t();

        // Resume: nový optimizer nad TÝMŽ Var (state Vars sdílí storage),
        // takže pokračujeme z bodu kde jsme skončili.
        let mut opt_b = EleutheriaAdamW::new(vec![v_test.clone()], ParamsAdamW::default())?;
        opt_b.restore_moments(&snapshot, step_t_a)?;
        for step in 3..5 {
            let grad_value = 0.5 - 0.1 * step as f32;
            let g = make_grads(&v_test, grad_value)?;
            opt_b.step(&g)?;
        }
        let test_var: Vec<f32> = v_test.as_tensor().to_vec1()?;

        for (a, b) in ref_var.iter().zip(test_var.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "trajektorie po restore se musí shodovat: ref={a}, test={b}"
            );
        }
        assert_eq!(opt_b.step_t(), 5);
        Ok(())
    }

    /// Restore odmítne moments s nesprávným počtem.
    #[test]
    fn restore_rejects_wrong_count() -> Result<()> {
        let device = Device::Cpu;
        let var = make_var(1.0, &device)?;
        let mut opt = EleutheriaAdamW::new(vec![var], ParamsAdamW::default())?;
        let bad: Vec<(Tensor, Tensor)> = vec![]; // prázdný
        let result = opt.restore_moments(&bad, 5);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("snapshotů"));
        Ok(())
    }

    /// Restore odmítne moments s nesprávným shape.
    #[test]
    fn restore_rejects_wrong_shape() -> Result<()> {
        let device = Device::Cpu;
        let var = make_var(1.0, &device)?;
        let mut opt = EleutheriaAdamW::new(vec![var], ParamsAdamW::default())?;
        // Var je shape (2,), zkus restore s (3,)
        let bad_m = Tensor::zeros((3,), DType::F32, &device)?;
        let bad_v = Tensor::zeros((3,), DType::F32, &device)?;
        let result = opt.restore_moments(&[(bad_m, bad_v)], 5);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("shape"));
        Ok(())
    }

    /// set_learning_rate funguje stejně jako v Candle (kompatibilita Optimizer trait).
    #[test]
    fn set_learning_rate_updates_params() -> Result<()> {
        let var = make_var(1.0, &Device::Cpu)?;
        let mut opt = EleutheriaAdamW::new(vec![var], ParamsAdamW::default())?;
        assert!((opt.learning_rate() - 0.001).abs() < 1e-12);
        opt.set_learning_rate(0.01);
        assert!((opt.learning_rate() - 0.01).abs() < 1e-12);
        assert!((opt.params().lr - 0.01).abs() < 1e-12);
        Ok(())
    }
}
