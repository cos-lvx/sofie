//! Normalizační vrstvy pro Falcon-H1.
//! RMSNorm (pre-layer norm) a RMSNormGated (SSM branch s gating).

use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::VarBuilder;

/// RMSNorm — Root Mean Square Layer Normalization.
/// Jednodušší než LayerNorm: žádné mean-centering, jen škálování.
/// output = weight * x / sqrt(mean(x²) + eps)
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x shape: [batch, seq_len, hidden_size]
        // 1. x² po prvcích
        let x_sq = x.sqr()?;
        // 2. mean(x²) přes poslední dimenzi, keepdim pro broadcasting
        let mean_sq = x_sq.mean_keepdim(D::Minus1)?;
        // 3. 1/sqrt(mean + eps)
        let scale = (mean_sq + self.eps)?.sqrt()?.recip()?;
        // 4. normalizuj a vynásob naučenou váhou
        x.broadcast_mul(&scale)?.broadcast_mul(&self.weight)
    }
}

/// RMSNormGated — RMSNorm s gating mechanismem pro Mamba-2 SSM branch.
///
/// norm_before_gate = false (Falcon-H1): output = rms_norm(x) * silu(gate)
/// norm_before_gate = true:              output = rms_norm(x * silu(gate))
pub struct RmsNormGated {
    weight: Tensor,
    eps: f64,
    norm_before_gate: bool,
}

impl RmsNormGated {
    pub fn load(
        size: usize,
        eps: f64,
        norm_before_gate: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps, norm_before_gate })
    }
}

impl RmsNormGated {
    /// Forward s gate signálem.
    /// x: SSM výstup, gate: gate projekce (obě [batch, seq, d_ssm])
    pub fn forward(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        if self.norm_before_gate {
            // Varianta A: norm(x * silu(gate))
            let gated = x.broadcast_mul(&silu(gate)?)?;
            self.rms_norm(&gated)
        } else {
            // Varianta B (Falcon-H1): rms_norm(x) * silu(gate)
            let normed = self.rms_norm(x)?;
            normed.broadcast_mul(&silu(gate)?)
        }
    }

    fn rms_norm(&self, x: &Tensor) -> Result<Tensor> {
        let x_sq = x.sqr()?;
        let mean_sq = x_sq.mean_keepdim(D::Minus1)?;
        let scale = (mean_sq + self.eps)?.sqrt()?.recip()?;
        x.broadcast_mul(&scale)?.broadcast_mul(&self.weight)
    }
}

/// Silu aktivace: silu(x) = x * sigmoid(x)
/// Smooth gate — propouští kladné, tlumí záporné.
fn silu(x: &Tensor) -> Result<Tensor> {
    let sigmoid = x.neg()?.exp()?.affine(1.0, 1.0)?.recip()?;
    x.broadcast_mul(&sigmoid)
}


