//! Falcon-H1 Model — kompletní inference pipeline.
//! Embedding → 44× decoder layer → final norm → lm_head → logity.

use candle_core::{Device, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};

use super::config::FalconH1Config;
use super::layer::FalconH1Layer;
use super::norm::RmsNorm;
use super::state::ModelState;

/// Kompletní Falcon-H1 model.
pub struct FalconH1Model {
    /// Token embedding: vocab_size → hidden_size
    embed_tokens: Embedding,
    /// 44 decoder layerů
    layers: Vec<FalconH1Layer>,
    /// Finální RMSNorm
    final_norm: RmsNorm,
    /// LM head: hidden_size → vocab_size (netie — vlastní váhy)
    lm_head: Linear,
    /// Konfigurace modelu (muP multipliery atd.)
    config: FalconH1Config,
}

impl FalconH1Model {
    pub fn load(config: &FalconH1Config, vb: VarBuilder, device: &Device) -> Result<Self> {
        // Token embedding
        let embed_tokens = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embed_tokens"),
        )?;

        // 44 decoder layerů
        let layers = (0..config.num_hidden_layers)
            .map(|i| FalconH1Layer::load(config, vb.pp(format!("model.layers.{i}")), device))
            .collect::<Result<Vec<_>>>()?;

        // Finální norma
        let final_norm = RmsNorm::load(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("model.final_layernorm"),
        )?;

        // LM head (vlastní váhy, ne tied)
        let lm_head =
            candle_nn::linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            config: config.clone(),
        })
    }
}

impl FalconH1Model {
    /// Forward pass celého modelu.
    /// input_ids: [batch, seq_len] — token IDs
    /// pos: pozice prvního tokenu v sekvenci
    /// state: ModelState — mutable stav všech 44 layerů
    /// Vrací logity: [batch, seq_len, vocab_size]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        pos: usize,
        state: &mut ModelState,
    ) -> Result<Tensor> {
        // === 1. Token embedding + muP scaling ===
        let mut x = self.embed_tokens.forward(input_ids)?; // [b, s, 3072]
        let emb_scale = Tensor::new(&[self.config.embedding_multiplier as f32], x.device())?
            .to_dtype(x.dtype())?;
        x = x.broadcast_mul(&emb_scale)?;

        // === 2. Průchod všemi 44 layery ===
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, pos, &mut state.layers[i])?;
        }

        // === 3. Finální norma ===
        x = self.final_norm.forward(&x)?;

        // === 4. LM head → logity + muP scaling ===
        let logits = self.lm_head.forward(&x)?; // [b, s, vocab_size]
        let lm_scale = Tensor::new(&[self.config.lm_head_multiplier as f32], logits.device())?
            .to_dtype(logits.dtype())?;
        let logits = logits.broadcast_mul(&lm_scale)?;

        Ok(logits)
    }
}
