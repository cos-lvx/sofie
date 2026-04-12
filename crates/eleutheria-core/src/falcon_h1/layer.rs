//! Falcon-H1 Decoder Layer — parallel hybrid (SSM + Attention + MLP).
//! Obě branch (SSM, Attention) dostávají stejný vstup po pre_norm,
//! jejich výstupy se SČÍTAJÍ s residualem. Pak post_norm → MLP → druhý residual.

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

use super::attention::Attention;
use super::mixer::Mixer;
use super::norm::RmsNorm;
use super::state::LayerState;

/// Jeden decoder layer Falcon-H1.
pub struct FalconH1Layer {
    /// Pre-normalizace sdílená pro SSM i Attention branch
    pre_norm: RmsNorm,
    /// SSM branch (Mamba-2)
    mixer: Mixer,
    /// Attention branch (GQA)
    attention: Attention,
    /// Post-normalizace před MLP
    post_norm: RmsNorm,
    /// MLP: SwiGLU — gate a up projekce
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    /// muP: škálování attention výstupu
    attention_out_multiplier: f64,
    /// muP: škálování SSM výstupu
    ssm_out_multiplier: f64,
    /// muP: MLP multipliery [gate/up scaling, down scaling]
    mlp_multipliers: Vec<f64>,
}

impl FalconH1Layer {
    pub fn load(
        config: &super::config::FalconH1Config,
        vb: VarBuilder,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let hidden = config.hidden_size; // 3072
        let intermediate = config.intermediate_size; // 12288

        // Normy
        let pre_norm = RmsNorm::load(hidden, config.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_norm = RmsNorm::load(hidden, config.rms_norm_eps, vb.pp("pre_ff_layernorm"))?;

        // Parallel branches
        let mixer = Mixer::load(config, vb.pp("mamba"), device)?;
        let attention = Attention::load(config, vb.pp("self_attn"), device)?;

        // MLP: SwiGLU
        let gate_proj = linear_no_bias(hidden, intermediate, vb.pp("feed_forward.gate_proj"))?;
        let up_proj = linear_no_bias(hidden, intermediate, vb.pp("feed_forward.up_proj"))?;
        let down_proj = linear_no_bias(intermediate, hidden, vb.pp("feed_forward.down_proj"))?;

        Ok(Self {
            pre_norm,
            mixer,
            attention,
            post_norm,
            gate_proj,
            up_proj,
            down_proj,
            attention_out_multiplier: config.attention_out_multiplier,
            ssm_out_multiplier: config.ssm_out_multiplier,
            mlp_multipliers: config.mlp_multipliers.clone(),
        })
    }
}

impl FalconH1Layer {
    /// Forward pass jednoho decoder layeru.
    /// x: [batch, seq_len, hidden_size]
    /// pos: pozice prvního tokenu (pro RoPE a KV cache)
    /// state: LayerState — modifikuje se in-place (SSM state, conv state, KV cache)
    pub fn forward(&self, x: &Tensor, pos: usize, state: &mut LayerState) -> Result<Tensor> {
        // === 1. Pre-norm (sdílená pro obě branch) ===
        let normed = self.pre_norm.forward(x)?; // [b, s, 3072]

        // === 2. Parallel branches + muP scaling ===
        // Prefill mód (seq_len > 1): parallel conv1d + sekvenční SSM scan = přesné výsledky
        // Decode mód (seq_len = 1): rekurentní krok token po tokenu
        let ssm_out = if normed.dim(1)? > 1 {
            self.mixer.forward_prefill(&normed, state)?
        } else {
            self.mixer.forward(&normed, state)?
        }; // [b, s, 3072]
        let ssm_scale = Tensor::new(&[self.ssm_out_multiplier as f32], ssm_out.device())?
            .to_dtype(ssm_out.dtype())?;
        let ssm_out = ssm_out.broadcast_mul(&ssm_scale)?;

        let attn_out = self.attention.forward(&normed, pos, state)?; // [b, s, 3072]
        let attn_scale = Tensor::new(&[self.attention_out_multiplier as f32], attn_out.device())?
            .to_dtype(attn_out.dtype())?;
        let attn_out = attn_out.broadcast_mul(&attn_scale)?;

        // === 3. Residual + obě branch (sčítání) ===
        let x = (x + ssm_out + attn_out)?; // [b, s, 3072]

        // === 4. Post-norm → MLP (SwiGLU) ===
        let normed = self.post_norm.forward(&x)?;

        let gate = self.gate_proj.forward(&normed)?; // [b, s, 12288]
        let up = self.up_proj.forward(&normed)?; // [b, s, 12288]

        // muP: škálování gate projekce (up se neškáluje)
        let mlp_in_scale = Tensor::new(&[self.mlp_multipliers[0] as f32], gate.device())?
            .to_dtype(gate.dtype())?;
        let gate = gate.broadcast_mul(&mlp_in_scale)?;

        // SwiGLU: silu(gate) * up — cast přes F32 pro numerickou stabilitu
        let gate_activated = {
            let g = gate.to_dtype(candle_core::DType::F32)?;
            let result = candle_nn::ops::silu(&g)?;
            result.to_dtype(normed.dtype())?
        };
        let mlp_out = (gate_activated * up)?; // [b, s, 12288]
        let mlp_out = self.down_proj.forward(&mlp_out)?; // [b, s, 3072]

        // muP: škálování down projekce
        let mlp_out_scale = Tensor::new(&[self.mlp_multipliers[1] as f32], mlp_out.device())?
            .to_dtype(mlp_out.dtype())?;
        let mlp_out = mlp_out.broadcast_mul(&mlp_out_scale)?;

        // === 5. Druhý residual ===
        let x = (x + mlp_out)?;

        Ok(x)
    }
}
