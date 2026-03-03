//! Falcon-H1 Decoder Layer — parallel hybrid (SSM + Attention + MLP).
//! Obě branch (SSM, Attention) dostávají stejný vstup po pre_norm,
//! jejich výstupy se SČÍTAJÍ s residualem. Pak post_norm → MLP → druhý residual.

use candle_core::{Result, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

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
}

impl FalconH1Layer {
    pub fn load(
        config: &super::config::FalconH1Config,
        vb: VarBuilder,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let hidden = config.hidden_size;          // 3072
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
        let normed = self.pre_norm.forward(x)?;  // [b, s, 3072]

        // === 2. Parallel branches ===
        let ssm_out = self.mixer.forward(&normed, state)?;       // [b, s, 3072]
        let attn_out = self.attention.forward(&normed, pos, state)?; // [b, s, 3072]

        // === 3. Residual + obě branch (sčítání) ===
        let x = (x + ssm_out + attn_out)?;  // [b, s, 3072]

        // === 4. Post-norm → MLP (SwiGLU) ===
        let normed = self.post_norm.forward(&x)?;

        let gate = self.gate_proj.forward(&normed)?;  // [b, s, 12288]
        let up = self.up_proj.forward(&normed)?;       // [b, s, 12288]

        // SwiGLU: silu(gate) * up
        let gate = candle_nn::ops::silu(&gate)?;
        let mlp_out = (gate * up)?;                    // [b, s, 12288]
        let mlp_out = self.down_proj.forward(&mlp_out)?; // [b, s, 3072]

        // === 5. Druhý residual ===
        let x = (x + mlp_out)?;

        Ok(x)
    }
}