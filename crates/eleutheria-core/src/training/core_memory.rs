//! `CoreMemory` — trainable initial SSM state pro jednu Mamba-2 vrstvu.
//!
//! Drží `candle_nn::Var` v F32 (pro numerickou stabilitu během backpropu)
//! se tvarem `[n_heads, headdim, d_state]`. Při forward pass se kopíruje
//! do `ModelState.layers[layer_idx].ssm_state` po upcastu na runtime dtype.
//! Gradient teče zpět přes `to_dtype` (autograd-aware v Candle) až k `Var`.
//!
//! Ve v0.5.0-alpha.1 trénujeme pouze vrstvu 0 — smoke test, ne produkce.

use candle_core::{DType, Device, Result, Var};

use crate::falcon_h1::config::FalconH1Config;

/// Trainable initial SSM state pro jednu vrstvu.
pub struct CoreMemory {
    /// Trainable tensor, F32, shape `[n_heads, headdim, d_state]`.
    pub init_state: Var,
    /// Index vrstvy, kterou tato Core Memory inicializuje.
    pub layer_idx: usize,
    /// Referenční tvary pro validaci proti modelu.
    pub n_heads: usize,
    pub headdim: usize,
    pub d_state: usize,
}

impl CoreMemory {
    /// Vytvoří Core Memory s nulovou inicializací (matching `ModelState::new`).
    pub fn zeros(config: &FalconH1Config, device: &Device, layer_idx: usize) -> Result<Self> {
        Self::validate_layer_idx(config, layer_idx)?;
        let shape = Self::state_shape(config);
        let init_state = Var::zeros(shape, DType::F32, device)?;
        Ok(Self::wrap(init_state, config, layer_idx))
    }

    /// Vytvoří Core Memory s malou náhodnou inicializací (pro experimenty,
    /// kde nulový start nevyvolá gradient — SSM rekurze s nulovým stavem
    /// a multiplikativní update `h' = dA·h + dB⊗x` má při `h=0` a `x=0`
    /// také nulovou derivaci vůči `h`; pro gradient signal chceme malé
    /// počáteční odchylky).
    pub fn randn_small(config: &FalconH1Config, device: &Device, layer_idx: usize) -> Result<Self> {
        Self::validate_layer_idx(config, layer_idx)?;
        let shape = Self::state_shape(config);
        let init_state = Var::randn_f64(0.0, 0.01, shape, DType::F32, device)?;
        Ok(Self::wrap(init_state, config, layer_idx))
    }

    fn validate_layer_idx(config: &FalconH1Config, layer_idx: usize) -> Result<()> {
        if layer_idx >= config.num_hidden_layers {
            return Err(candle_core::Error::Msg(format!(
                "layer_idx={} je mimo rozsah (num_hidden_layers={})",
                layer_idx, config.num_hidden_layers
            )));
        }
        Ok(())
    }

    fn state_shape(config: &FalconH1Config) -> (usize, usize, usize) {
        (
            config.mamba_n_heads,
            config.mamba_d_head,
            config.mamba_d_state,
        )
    }

    fn wrap(init_state: Var, config: &FalconH1Config, layer_idx: usize) -> Self {
        Self {
            init_state,
            layer_idx,
            n_heads: config.mamba_n_heads,
            headdim: config.mamba_d_head,
            d_state: config.mamba_d_state,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_config() -> FalconH1Config {
        FalconH1Config {
            vocab_size: 100,
            hidden_size: 32,
            num_hidden_layers: 4,
            intermediate_size: 64,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 16,
            mamba_d_state: 8,
            mamba_n_heads: 2,
            mamba_d_head: 16,
            mamba_d_ssm: 32,
            mamba_d_conv: 4,
            mamba_expand: 2,
            mamba_n_groups: 1,
            mamba_chunk_size: 256,
            mamba_conv_bias: true,
            mamba_proj_bias: false,
            mamba_norm_before_gate: false,
            mamba_rms_norm: true,
            mamba_use_mlp: true,
            embedding_multiplier: 5.66,
            lm_head_multiplier: 0.0195,
            ssm_in_multiplier: 0.4167,
            ssm_out_multiplier: 0.1179,
            ssm_multipliers: vec![0.2946],
            attention_in_multiplier: 1.0,
            attention_out_multiplier: 0.1042,
            key_multiplier: 1.0,
            mlp_multipliers: vec![0.2946],
            rms_norm_eps: 1e-5,
            eos_token_id: Some(11),
            rope_theta: 1e11,
            tie_word_embeddings: false,
            max_position_embeddings: 1000,
        }
    }

    #[test]
    fn zeros_creates_var_with_correct_shape() {
        let config = dummy_config();
        let cm = CoreMemory::zeros(&config, &Device::Cpu, 0).unwrap();
        assert_eq!(cm.layer_idx, 0);
        assert_eq!(cm.n_heads, 2);
        assert_eq!(cm.headdim, 16);
        assert_eq!(cm.d_state, 8);
        let tensor = cm.init_state.as_tensor();
        assert_eq!(tensor.dims(), &[2, 16, 8]);
        assert_eq!(tensor.dtype(), DType::F32);
        // Zero init: sum must be 0.
        let sum: f32 = tensor.sum_all().unwrap().to_scalar().unwrap();
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn randn_small_creates_nonzero_var() {
        let config = dummy_config();
        let cm = CoreMemory::randn_small(&config, &Device::Cpu, 1).unwrap();
        assert_eq!(cm.layer_idx, 1);
        let tensor = cm.init_state.as_tensor();
        // Random init: variance > 0 (not all zeros).
        let sum_sq: f32 = tensor
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(sum_sq > 0.0, "randn init should produce non-zero values");
    }

    #[test]
    fn invalid_layer_idx_rejected() {
        let config = dummy_config();
        let result = CoreMemory::zeros(&config, &Device::Cpu, 99);
        assert!(result.is_err());
    }
}
