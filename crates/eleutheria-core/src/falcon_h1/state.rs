//! Stavové struktury pro Falcon-H1 inference.
//! Každý layer si drží SSM state, conv state a KV cache.

use candle_core::{DType, Device, Result, Tensor};

/// Stav jednoho Falcon-H1 layeru mezi tokeny.
pub struct LayerState {
    /// SSM hidden state: [n_heads, headdim, d_state]
    /// Akumuluje informaci ze všech předchozích tokenů.
    pub ssm_state: Tensor,

    /// Conv1d sliding window: [d_inner, d_conv]
    /// d_inner = mamba_d_ssm + 2 * n_groups * d_state = 3584
    /// Drží posledních d_conv tokenů pro kauzální konvoluci.
    pub conv_state: Tensor,

    /// KV cache - klíče pro attention. Roste s každým tokenem.
    /// Shape: [batch, n_kv_heads, seq_len, head_dim]
    pub k_cache: Tensor,

    /// KV cache - hodnoty pro attention.
    /// Shape: [batch, n_kv_heads, seq_len, head_dim]
    pub v_cache: Tensor,
}

impl LayerState {
    /// Vytvoří prázdný stav (nuly) pro jeden layer.
    pub fn new(
        config: &super::config::FalconH1Config,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let d_inner = config.mamba_d_ssm + 2 * config.mamba_n_groups * config.mamba_d_state;

        Ok(Self {
            ssm_state: Tensor::zeros(
                (
                    config.mamba_n_heads,
                    config.mamba_d_head,
                    config.mamba_d_state,
                ),
                dtype,
                device,
            )?,
            conv_state: Tensor::zeros((d_inner, config.mamba_d_conv), dtype, device)?,
            k_cache: Tensor::zeros(
                (1, config.num_key_value_heads, 0, config.head_dim),
                dtype,
                device,
            )?,
            v_cache: Tensor::zeros(
                (1, config.num_key_value_heads, 0, config.head_dim),
                dtype,
                device,
            )?,
        })
    }
}

/// Stav celého modelu — všech 44 layerů.
pub struct ModelState {
    pub layers: Vec<LayerState>,
}

impl ModelState {
    /// Vytvoří prázdný stav pro celý model.
    pub fn new(
        config: &super::config::FalconH1Config,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|_| LayerState::new(config, dtype, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }
}
