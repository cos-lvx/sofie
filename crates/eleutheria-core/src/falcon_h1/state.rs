//! Stavové struktury pro Falcon-H1 inference.
//! Každý layer si drží SSM state, conv state a KV cache.

use candle_core::{Device, DType, Result, Tensor};

/// Stav jednoho Falcon-H1 layeru mezi tokeny.
pub struct LayerState {
    /// SSM hidden state: [n_heads, headdim, d_state]
    /// Akumuluje informaci ze všech předchozích tokenů.
    pub ssm_state: Tensor,

    /// Conv1d sliding window: [d_inner, d_conv-1]
    /// d_inner = mamba_d_ssm + 2 * n_groupps * d_state = 3584
    /// Drží posledních d_conv-1 = 3 tokenů pro casual konvoluci.
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
        n_heads: usize,
        headdim: usize,
        d_state: usize,
        d_inner: usize,
        d_conv: usize,
        n_kv_heads: usize,
        kv_head_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            // SSM state musí být FP32 - jinak numerická nestabilkita
            ssm_state: Tensor::zeros(
                (n_heads, headdim, d_state),
                DType::F32,
                device,
            )?,
            // Conv state: okno posledních d_conv-1 tokenů
            conv_state: Tensor::zeros(
                (d_inner, d_conv - 1),
                DType::F32,
                device,
            )?,
            // KV cache začíná prázdná (0 tokenů)
            k_cache: Tensor::zeros(
                (1, n_kv_heads, 0, kv_head_dim),
                DType::F32,
                device,
            )?,
            v_cache: Tensor::zeros(
                (1, n_kv_heads, 0, kv_head_dim),
                DType::F32,
                device,
            )?,
        })
    }
}

/// Stav celého modelu - všech 44 layerů
pub struct ModelState {
    pub layers: Vec<LayerState>,
}

impl ModelState {
    /// Vytvoří ptázdný stav pro celý model.
    pub fn new(config: &super::config::FalconH1Config, device: &Device) -> Result<Self> {
        let d_inner = config.mamba_d_ssm
            + 2 * config.mamba_n_groups * config.mamba_d_state;

        let layers = (0..config.num_hidden_layers)
            .map(|_| LayerState::new(
                config.mamba_n_heads,
                config.mamba_d_head,
                config.mamba_d_state,
                d_inner,
                config.mamba_d_conv,
                config.num_key_value_heads,
                config.head_dim,
                device,
            ))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }
}