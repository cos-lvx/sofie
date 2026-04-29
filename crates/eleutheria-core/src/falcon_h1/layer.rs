//! Falcon-H1 Decoder Layer — parallel hybrid (SSM + Attention + MLP).
//! Obě branch (SSM, Attention) dostávají stejný vstup po pre_norm,
//! jejich výstupy se SČÍTAJÍ s residualem. Pak post_norm → MLP → druhý residual.

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

use super::attention::Attention;
use super::mixer::Mixer;
use super::norm::RmsNorm;
use super::state::LayerState;
use crate::training::trace;

/// Sub-layer cut point — umožňuje zastavit `FalconH1Layer::forward` uprostřed
/// a vrátit hidden stream z konkrétní mezibodě. Slouží pro binary search
/// lokalizaci op, jejíž backward v BUG-010 produkuje NaN.
///
/// Pořadí odpovídá forward flow: `pre_norm → ssm_branch → attn_branch →
/// residual1 → post_norm → mlp_gate → mlp_silu_mul → mlp_down → residual2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerStop {
    /// Po `pre_norm` — před rozvětvením do SSM/attention.
    AfterPreNorm,
    /// Po SSM branch (včetně muP scaling), před attention.
    AfterSsmBranch,
    /// Po attention branch (včetně muP scaling), před residual sum.
    AfterAttnBranch,
    /// Po `x + ssm_out + attn_out` — před post_norm.
    AfterResidual1,
    /// Po `post_norm` — před MLP.
    AfterPostNorm,
    /// Po `gate_proj + muP scale` — před silu.
    AfterMlpGate,
    /// Po `silu(gate) * up` — před down_proj.
    AfterMlpSiluMul,
    /// Po `down_proj + muP scale` — před druhý residual.
    AfterMlpDown,
    /// Kompletní layer (stejné jako `forward`).
    Full,
}

impl LayerStop {
    /// Popisek pro reporting.
    pub fn label(&self) -> &'static str {
        match self {
            Self::AfterPreNorm => "after-pre-norm",
            Self::AfterSsmBranch => "after-ssm",
            Self::AfterAttnBranch => "after-attn",
            Self::AfterResidual1 => "after-residual-1",
            Self::AfterPostNorm => "after-post-norm",
            Self::AfterMlpGate => "after-mlp-gate",
            Self::AfterMlpSiluMul => "after-mlp-silu-mul",
            Self::AfterMlpDown => "after-mlp-down",
            Self::Full => "full",
        }
    }
}

impl std::str::FromStr for LayerStop {
    type Err = String;

    /// Parser pro CLI flag (`--cut-at-component`).
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "pre-norm" | "after-pre-norm" => Ok(Self::AfterPreNorm),
            "ssm" | "after-ssm" => Ok(Self::AfterSsmBranch),
            "attn" | "attention" | "after-attn" => Ok(Self::AfterAttnBranch),
            "residual1" | "after-residual-1" => Ok(Self::AfterResidual1),
            "post-norm" | "after-post-norm" => Ok(Self::AfterPostNorm),
            "mlp-gate" | "after-mlp-gate" => Ok(Self::AfterMlpGate),
            "mlp-silu-mul" | "after-mlp-silu-mul" => Ok(Self::AfterMlpSiluMul),
            "mlp-down" | "after-mlp-down" => Ok(Self::AfterMlpDown),
            "full" | "after-residual-2" => Ok(Self::Full),
            other => Err(format!("neznámá LayerStop varianta: {other}")),
        }
    }
}

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
        self.forward_until(x, pos, state, LayerStop::Full)
    }

    /// Sub-layer chunk α — `x → residual_1` (pre_norm + parallel SSM/attention
    /// branches + první residual sum). Používané pro sub-layer gradient
    /// checkpointing (alpha.13). Modifikuje state (SSM scan, KV cache).
    ///
    /// Output: `x + ssm(pre_norm(x)) + attention(pre_norm(x))` — vstup pro
    /// chunk β. Když gradient teče zpět přes residual sum, automaticky se
    /// pokrývají obě parallel branche + skip connection.
    pub fn forward_chunk_branches(
        &self,
        x: &Tensor,
        pos: usize,
        state: &mut LayerState,
    ) -> Result<Tensor> {
        let normed = self.pre_norm.forward(x)?;

        let ssm_out = if normed.dim(1)? > 1 {
            self.mixer.forward_prefill(&normed, state)?
        } else {
            self.mixer.forward(&normed, state)?
        };
        let ssm_scale = Tensor::new(&[self.ssm_out_multiplier as f32], ssm_out.device())?
            .to_dtype(ssm_out.dtype())?;
        let ssm_out = ssm_out.broadcast_mul(&ssm_scale)?;

        let attn_out = self.attention.forward(&normed, pos, state)?;
        let attn_scale = Tensor::new(&[self.attention_out_multiplier as f32], attn_out.device())?
            .to_dtype(attn_out.dtype())?;
        let attn_out = attn_out.broadcast_mul(&attn_scale)?;

        let res1 = (x + ssm_out + attn_out)?;
        Ok(res1)
    }

    /// Sub-layer chunk β — `residual_1 → x_out` (post_norm + MLP + druhý
    /// residual). Doplněk k `forward_chunk_branches` pro sub-layer
    /// checkpointing (alpha.13). Vrací finální output vrstvy.
    ///
    /// `forward_chunk_branches(x, pos, state)` + `forward_chunk_mlp(res1, state)`
    /// = `forward(x, pos, state)`.
    pub fn forward_chunk_mlp(&self, res1: &Tensor, _state: &mut LayerState) -> Result<Tensor> {
        let normed = self.post_norm.forward(res1)?;
        let gate = self.gate_proj.forward(&normed)?;
        let up = self.up_proj.forward(&normed)?;

        let mlp_in_scale = Tensor::new(&[self.mlp_multipliers[0] as f32], gate.device())?
            .to_dtype(gate.dtype())?;
        let gate = gate.broadcast_mul(&mlp_in_scale)?;

        let gate_activated = {
            let g = gate.to_dtype(candle_core::DType::F32)?;
            let result = candle_nn::ops::silu(&g)?;
            result.to_dtype(normed.dtype())?
        };
        let mlp_out = (gate_activated * up)?;
        let mlp_out = self.down_proj.forward(&mlp_out)?;

        let mlp_out_scale = Tensor::new(&[self.mlp_multipliers[1] as f32], mlp_out.device())?
            .to_dtype(mlp_out.dtype())?;
        let mlp_out = mlp_out.broadcast_mul(&mlp_out_scale)?;

        let x_out = (res1 + mlp_out)?;
        Ok(x_out)
    }

    /// Forward pass s volitelným sub-layer cut bodem. Vrací hidden stream
    /// z požadované mezibodě. Pro `LayerStop::Full` je chování totožné
    /// s `forward`.
    ///
    /// Slouží pro binary search lokalizaci op zodpovědné za NaN gradient
    /// (BUG-010) — loss na hidden z konkrétní mezibodě izoluje backward path.
    pub fn forward_until(
        &self,
        x: &Tensor,
        pos: usize,
        state: &mut LayerState,
        stop: LayerStop,
    ) -> Result<Tensor> {
        // === 1. Pre-norm (sdílená pro obě branch) ===
        let normed = self.pre_norm.forward(x)?; // [b, s, 3072]
        trace::probe(&normed, "layer.pre_norm_out")?;
        if stop == LayerStop::AfterPreNorm {
            return Ok(normed);
        }

        // === 2. Parallel branches + muP scaling ===
        // Prefill mód (seq_len > 1): parallel conv1d + sekvenční SSM scan = přesné výsledky
        // Decode mód (seq_len = 1): rekurentní krok token po tokenu
        let ssm_out = if normed.dim(1)? > 1 {
            self.mixer.forward_prefill(&normed, state)?
        } else {
            self.mixer.forward(&normed, state)?
        }; // [b, s, 3072]
        trace::probe(&ssm_out, "layer.mixer_out_raw")?;
        let ssm_scale = Tensor::new(&[self.ssm_out_multiplier as f32], ssm_out.device())?
            .to_dtype(ssm_out.dtype())?;
        let ssm_out = ssm_out.broadcast_mul(&ssm_scale)?;
        trace::probe(&ssm_out, "layer.ssm_out_scaled")?;
        if stop == LayerStop::AfterSsmBranch {
            return Ok(ssm_out);
        }

        let attn_out = self.attention.forward(&normed, pos, state)?; // [b, s, 3072]
        trace::probe(&attn_out, "layer.attn_out_raw")?;
        let attn_scale = Tensor::new(&[self.attention_out_multiplier as f32], attn_out.device())?
            .to_dtype(attn_out.dtype())?;
        let attn_out = attn_out.broadcast_mul(&attn_scale)?;
        trace::probe(&attn_out, "layer.attn_out_scaled")?;
        if stop == LayerStop::AfterAttnBranch {
            return Ok(attn_out);
        }

        // === 3. Residual + obě branch (sčítání) ===
        let x = (x + ssm_out + attn_out)?; // [b, s, 3072]
        trace::probe(&x, "layer.residual_1")?;
        if stop == LayerStop::AfterResidual1 {
            return Ok(x);
        }

        // === 4. Post-norm → MLP (SwiGLU) ===
        let normed = self.post_norm.forward(&x)?;
        trace::probe(&normed, "layer.post_norm_out")?;
        if stop == LayerStop::AfterPostNorm {
            return Ok(normed);
        }

        let gate = self.gate_proj.forward(&normed)?; // [b, s, 12288]
        let up = self.up_proj.forward(&normed)?; // [b, s, 12288]
        trace::probe(&gate, "layer.mlp.gate_raw")?;
        trace::probe(&up, "layer.mlp.up")?;

        // muP: škálování gate projekce (up se neškáluje)
        let mlp_in_scale = Tensor::new(&[self.mlp_multipliers[0] as f32], gate.device())?
            .to_dtype(gate.dtype())?;
        let gate = gate.broadcast_mul(&mlp_in_scale)?;
        trace::probe(&gate, "layer.mlp.gate_scaled")?;
        if stop == LayerStop::AfterMlpGate {
            return Ok(gate);
        }

        // SwiGLU: silu(gate) * up — cast přes F32 pro numerickou stabilitu
        let gate_activated = {
            let g = gate.to_dtype(candle_core::DType::F32)?;
            let result = candle_nn::ops::silu(&g)?;
            result.to_dtype(normed.dtype())?
        };
        trace::probe(&gate_activated, "layer.mlp.silu_gate")?;
        let mlp_out = (gate_activated * up)?; // [b, s, 12288]
        trace::probe(&mlp_out, "layer.mlp.silu_gate_times_up")?;
        if stop == LayerStop::AfterMlpSiluMul {
            return Ok(mlp_out);
        }

        let mlp_out = self.down_proj.forward(&mlp_out)?; // [b, s, 3072]
        trace::probe(&mlp_out, "layer.mlp.down_raw")?;

        // muP: škálování down projekce
        let mlp_out_scale = Tensor::new(&[self.mlp_multipliers[1] as f32], mlp_out.device())?
            .to_dtype(mlp_out.dtype())?;
        let mlp_out = mlp_out.broadcast_mul(&mlp_out_scale)?;
        trace::probe(&mlp_out, "layer.mlp.down_scaled")?;
        if stop == LayerStop::AfterMlpDown {
            return Ok(mlp_out);
        }

        // === 5. Druhý residual ===
        let x = (x + mlp_out)?;
        trace::probe(&x, "layer.residual_2")?;

        Ok(x)
    }
}
