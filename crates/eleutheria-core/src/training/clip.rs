//! Gradient clipping pro numerickou stabilitu.
//!
//! Candle nemá built-in `clip_grad_norm` v candle-nn (2026-04). Tohle je
//! vlastní implementace matching PyTorch `torch.nn.utils.clip_grad_norm_`:
//!
//! 1. Spočítá globální L2 normu přes všechny gradienty: `total_norm = sqrt(Σ grad_i²)`
//! 2. Pokud `total_norm > max_norm`, škáluje **všechny** gradienty faktorem
//!    `max_norm / total_norm` (uniformní škálování, zachovává směr)
//! 3. Vrací původní (pre-clip) total_norm pro monitoring
//!
//! Gradient clipping je **standardní recept** pro Mamba-2 / Falcon-H1 training
//! (HF Transformers issue #32570, `max_grad_norm=1.0`). Pro Pre-LN architektury
//! s massive activations je to nutnost, ne volitelnost.

use anyhow::{Result, anyhow};
use candle_core::{DType, Tensor, Var, backprop::GradStore};

/// Globální L2 norma gradientů přes množinu `Var`. Pokud je větší než
/// `max_norm`, škáluje **všechny** gradienty in-place v `GradStore` tak,
/// že nová globální norma je přesně `max_norm`. Vrací původní pre-clip
/// normu (pro logging a diagnostiku).
///
/// Pokud některý gradient chybí (Var není v grad store), přeskočí ho.
/// Pokud globální norma je NaN/Inf, vrátí Err — clipping nemá smysl,
/// gradienty jsou nepoužitelné.
pub fn clip_grad_norm(grads: &mut GradStore, vars: &[&Var], max_norm: f64) -> Result<f64> {
    // 1) Spočítej sum of squares napříč všemi gradienty.
    let mut sum_sq: f64 = 0.0;
    for var in vars {
        let grad_clone = grads.get(var.as_tensor()).cloned();
        if let Some(g) = grad_clone {
            let g_f32 = g.to_dtype(DType::F32)?;
            let ssq: f32 = g_f32.sqr()?.sum_all()?.to_scalar()?;
            sum_sq += ssq as f64;
        }
    }
    let total_norm = sum_sq.sqrt();

    if !total_norm.is_finite() {
        return Err(anyhow!(
            "global gradient L2 norm je {:.3e} (NaN/Inf) — clipping nedává smysl, gradienty jsou nepoužitelné",
            total_norm
        ));
    }

    // 2) Pokud pod prahem, nic nedělej.
    if total_norm <= max_norm {
        return Ok(total_norm);
    }

    // 3) Jinak škáluj všechny gradienty dolů.
    let scale = (max_norm / total_norm) as f32;

    for var in vars {
        let grad_clone = grads.get(var.as_tensor()).cloned();
        if let Some(g) = grad_clone {
            let scale_t = Tensor::new(scale, g.device())?.to_dtype(g.dtype())?;
            let scaled = g.broadcast_mul(&scale_t)?;
            grads.insert(var.as_tensor(), scaled);
        }
    }

    Ok(total_norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    #[test]
    fn clip_below_threshold_does_nothing() {
        // Vytvoř Var a manuálně nainstaluj gradient menší než threshold.
        let var = Var::randn_f64(0.0, 0.01, (10,), DType::F32, &Device::Cpu).unwrap();
        let loss = var.as_tensor().sum_all().unwrap();
        let mut grads = loss.backward().unwrap();

        let pre_norm = clip_grad_norm(&mut grads, &[&var], 100.0).unwrap();
        let grad = grads.get(var.as_tensor()).unwrap();
        let post_sum_sq: f32 = grad
            .to_dtype(DType::F32)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        let post_norm = (post_sum_sq as f64).sqrt();

        // Pre-norm a post-norm musí být stejné (pod prahem).
        assert!((pre_norm - post_norm).abs() < 1e-6);
    }

    #[test]
    fn clip_above_threshold_scales_down() {
        let var = Var::randn_f64(0.0, 1.0, (100,), DType::F32, &Device::Cpu).unwrap();
        // Loss = 10 * sum(var) → gradient bude 10 * ones
        let scale = Tensor::new(10.0f32, &Device::Cpu).unwrap();
        let loss = var
            .as_tensor()
            .broadcast_mul(&scale)
            .unwrap()
            .sum_all()
            .unwrap();
        let mut grads = loss.backward().unwrap();

        let pre_norm = clip_grad_norm(&mut grads, &[&var], 1.0).unwrap();

        // pre-norm by mělo být sqrt(100 * 10²) = 100
        assert!((pre_norm - 100.0).abs() < 1.0);

        let grad = grads.get(var.as_tensor()).unwrap();
        let post_sum_sq: f32 = grad
            .to_dtype(DType::F32)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        let post_norm = (post_sum_sq as f64).sqrt();
        // Post-norm má být přesně max_norm (s malou číselnou tolerancí)
        assert!((post_norm - 1.0).abs() < 0.01, "post_norm = {}", post_norm);
    }
}
