//! Falcon-H1 Model — kompletní inference pipeline.
//! Embedding → 44× decoder layer → final norm → lm_head → logity.

use candle_core::{Device, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};

use super::config::FalconH1Config;
use super::layer::{FalconH1Layer, LayerStop};
use super::norm::RmsNorm;
use super::state::ModelState;
use crate::training::trace;

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
        let x = self.embed_and_layers(input_ids, pos, state, None, LayerStop::Full)?;
        self.final_head(&x)
    }

    /// Forward pouze do vrstvy `up_to_layer` (včetně) — vrací hidden
    /// stream **před** final_norm a lm_head. Slouží pro diagnostiku:
    /// loss na hidden izoluje backward path na `layer_idx..=up_to_layer`,
    /// takže lze binary-searchem lokalizovat zdroj NaN/Inf.
    pub fn forward_up_to_layer(
        &self,
        input_ids: &Tensor,
        pos: usize,
        state: &mut ModelState,
        up_to_layer: usize,
    ) -> Result<Tensor> {
        self.embed_and_layers(input_ids, pos, state, Some(up_to_layer), LayerStop::Full)
    }

    /// Varianta `forward_up_to_layer` se sub-layer cut na **poslední vrstvě**.
    /// Vrstvy `[0 .. up_to_layer]` běží plně, vrstva `up_to_layer` se zastaví
    /// na `stop` bodu a vrací odtud hidden. Pro `stop == LayerStop::Full` je
    /// chování totožné s `forward_up_to_layer`.
    ///
    /// Umožňuje bisect uvnitř jedné vrstvy pro lokalizaci op s NaN backward.
    pub fn forward_up_to_layer_with_stop(
        &self,
        input_ids: &Tensor,
        pos: usize,
        state: &mut ModelState,
        up_to_layer: usize,
        stop: LayerStop,
    ) -> Result<Tensor> {
        self.embed_and_layers(input_ids, pos, state, Some(up_to_layer), stop)
    }

    /// Počet decoder vrstev (= `config.num_hidden_layers`).
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Vrací embedding s muP scaling (viz `embed_and_layers` krok 1).
    /// Použito v chunked checkpointing — embedding je vstup pro chunk 0.
    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut x = self.embed_tokens.forward(input_ids)?;
        let emb_scale = Tensor::new(&[self.config.embedding_multiplier as f32], x.device())?
            .to_dtype(x.dtype())?;
        x = x.broadcast_mul(&emb_scale)?;
        Ok(x)
    }

    /// Forward pass jedné vrstvy — chunked checkpointing volá tuto metodu
    /// per layer s explicit input. Modifikuje `state.layers[layer_idx]`
    /// in-place (SSM scan, KV cache).
    pub fn forward_layer(
        &self,
        layer_idx: usize,
        x: &Tensor,
        pos: usize,
        state: &mut ModelState,
    ) -> Result<Tensor> {
        if layer_idx >= self.layers.len() {
            return Err(candle_core::Error::Msg(format!(
                "layer_idx={} mimo rozsah (num_layers={})",
                layer_idx,
                self.layers.len()
            )));
        }
        self.layers[layer_idx].forward(x, pos, &mut state.layers[layer_idx])
    }

    /// Sub-layer chunk α — `x → residual_1` (pre_norm + branches). Pro
    /// alpha.13 sub-layer checkpointing.
    pub fn forward_layer_branches(
        &self,
        layer_idx: usize,
        x: &Tensor,
        pos: usize,
        state: &mut ModelState,
    ) -> Result<Tensor> {
        if layer_idx >= self.layers.len() {
            return Err(candle_core::Error::Msg(format!(
                "layer_idx={} mimo rozsah (num_layers={})",
                layer_idx,
                self.layers.len()
            )));
        }
        self.layers[layer_idx].forward_chunk_branches(x, pos, &mut state.layers[layer_idx])
    }

    /// Sub-layer chunk β — `residual_1 → x_out` (post_norm + MLP + residual).
    /// Doplněk k `forward_layer_branches`.
    pub fn forward_layer_mlp(
        &self,
        layer_idx: usize,
        res1: &Tensor,
        state: &mut ModelState,
    ) -> Result<Tensor> {
        if layer_idx >= self.layers.len() {
            return Err(candle_core::Error::Msg(format!(
                "layer_idx={} mimo rozsah (num_layers={})",
                layer_idx,
                self.layers.len()
            )));
        }
        self.layers[layer_idx].forward_chunk_mlp(res1, &mut state.layers[layer_idx])
    }

    /// Final norm + lm_head + muP scaling — vrací logity.
    /// Pro chunked checkpointing: poslední chunk volá toto na hidden po
    /// vrstvě N-1.
    pub fn final_head(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.final_norm.forward(x)?;
        let logits = self.lm_head.forward(&x)?;
        let lm_scale = Tensor::new(&[self.config.lm_head_multiplier as f32], logits.device())?
            .to_dtype(logits.dtype())?;
        logits.broadcast_mul(&lm_scale)
    }

    fn embed_and_layers(
        &self,
        input_ids: &Tensor,
        pos: usize,
        state: &mut ModelState,
        up_to_layer: Option<usize>,
        last_layer_stop: LayerStop,
    ) -> Result<Tensor> {
        // === 1. Token embedding + muP scaling ===
        let mut x = self.embed_tokens.forward(input_ids)?;
        let emb_scale = Tensor::new(&[self.config.embedding_multiplier as f32], x.device())?
            .to_dtype(x.dtype())?;
        x = x.broadcast_mul(&emb_scale)?;
        trace::probe(&x, "model.embed_scaled")?;

        // === 2. Průchod decoder vrstvami (až do up_to_layer včetně, nebo všechny) ===
        let last = up_to_layer.unwrap_or(self.layers.len() - 1);
        for (i, layer) in self.layers.iter().enumerate() {
            if i > last {
                break;
            }
            let stop = if i == last {
                last_layer_stop
            } else {
                LayerStop::Full
            };
            x = layer.forward_until(&x, pos, &mut state.layers[i], stop)?;
            trace::probe(&x, &format!("model.after_layer_{i}"))?;
        }
        Ok(x)
    }
}
