//! Grouped Query Attention pro Falcon-H1.
//! 12 Q hlav, 2 KV hlav (GQA ratio 6:1), RoPE s theta=10^11.


use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

use super::state::LayerState;

/// Rotary Position Embeddings.
/// Kóduje pozici tokenu rotací párů dimenzí v Q a K.
/// Žádné naučené parametry — čistá geometrie.
pub struct RotaryEmbedding {
    /// Precomputed cos hodnoty: [max_seq, head_dim/2]
    cos_cache: Tensor,
    /// Precomputed sin hodnoty: [max_seq, head_dim/2]
    sin_cache: Tensor,
}

impl RotaryEmbedding {
    /// Vytvoří RoPE tabulky pro daný head_dim a theta.
    /// max_seq = kolik pozic předpočítat (256K pro Falcon-H1).
    pub fn new(head_dim: usize, theta: f64, max_seq: usize, device: &Device) -> Result<Self> {
        // Frekvence: theta^(-2i/d) pro i = 0, 1, ..., d/2 - 1
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf(i as f64 * 2.0 / head_dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?; // [half]

        // Pozice: 0, 1, 2, ..., max_seq-1
        let positions: Vec<f32> = (0..max_seq).map(|p| p as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?; // [max_seq]

        // Outer product: positions × inv_freq → [max_seq, half]
        let angles = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;

        let cos_cache = angles.cos()?;
        let sin_cache = angles.sin()?;

        Ok(Self { cos_cache, sin_cache })
    }

    /// Aplikuje rotaci na tensor [batch, n_heads, seq_len, head_dim] na dané pozici.
    pub fn apply(&self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let (_batch, _heads, seq_len, head_dim) = x.dims4()?;
        let half = head_dim / 2;

        // Vyber cos/sin pro pozice pos..pos+seq_len
        let cos = self.cos_cache.i(pos..pos + seq_len)?; // [seq_len, half]
        let sin = self.sin_cache.i(pos..pos + seq_len)?;

        // Reshape pro broadcasting: [1, 1, seq_len, half]
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        // Rozděl x na dvě poloviny
        let x1 = x.narrow(D::Minus1, 0, half)?;     // prvních half dimenzí
        let x2 = x.narrow(D::Minus1, half, half)?;          // druhých half dimenzí

        // Rotace: [x1*cos - x2*sin, x1*sin + x2*cos]
        let rotated_x1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let rotated_x2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;

        Tensor::cat(&[rotated_x1, rotated_x2], D::Minus1)
    }
}

/// Grouped Query Attention.
/// 12 Q hlav sdílí 2 KV hlavy (6:1 ratio).
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    rope: RotaryEmbedding,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    /// GQA ratio: kolik Q hlav sdílí jednu KV hlavu
    gqa_ratio: usize,
    /// muP key scaling (nahrazuje standardní 1/sqrt(d))
    key_multiplier: f64,
}

impl Attention {
    pub fn load(
        config: &super::config::FalconH1Config,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let hidden = config.hidden_size;        // 3072
        let n_q = config.num_attention_heads;   // 12
        let n_kv = config.num_key_value_heads;  // 2
        let hd = config.head_dim;               // 128

        let q_proj = linear_no_bias(hidden, n_q * hd, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden, n_kv * hd, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden, n_kv * hd, vb.pp("v_proj"))?;

        // RoPE: předpočítáme pro 4096 pozic, rozšíříme pokud bude potřeba
        let rope = RotaryEmbedding::new(hd, config.rope_theta, 4096, device)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            rope,
            n_q_heads: n_q,
            n_kv_heads: n_kv,
            head_dim: hd,
            gqa_ratio: n_q / n_kv,  // 12/2 = 6
            key_multiplier: config.key_multiplier,
        })
    }
}

impl Attention {
    /// Forward pass pro jeden nebo více tokenů.
    /// x: [batch, seq_len, hidden_size]
    /// pos: pozice prvního tokenu v sekvenci (pro RoPE a KV cache)
    /// state: LayerState s KV cache — modifikuje se in-place
    pub fn forward(&self, x: &Tensor, pos: usize, state: &mut LayerState) -> Result<Tensor> {
        let (batch, seq_len, _hidden) = x.dims3()?;

        // === 1. Projekce Q, K, V ===
        let q = self.q_proj.forward(x)?;  // [b, s, n_q * hd]
        let k = self.k_proj.forward(x)?;  // [b, s, n_kv * hd]
        let v = self.v_proj.forward(x)?;  // [b, s, n_kv * hd]

        // Reshape na hlavy: [b, s, n, hd] → [b, n, s, hd]
        let q = q.reshape((batch, seq_len, self.n_q_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k.reshape((batch, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v.reshape((batch, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // === 2. RoPE na Q a K ===
        let q = self.rope.apply(&q, pos)?;
        let k = self.rope.apply(&k, pos)?;

        // === 3. Aktualizace KV cache ===
        // Připojíme nové K, V k existujícímu cache
        let k = if state.k_cache.dim(0)? == 0 || state.k_cache.dims()[state.k_cache.dims().len() - 2] == 0 {
            k
        } else {
            Tensor::cat(&[&state.k_cache, &k], 2)?  // cat přes seq dim
        };
        let v = if state.v_cache.dim(0)? == 0 || state.v_cache.dims()[state.v_cache.dims().len() - 2] == 0 {
            v
        } else {
            Tensor::cat(&[&state.v_cache, &v], 2)?
        };

        // Ulož aktualizovaný cache
        state.k_cache = k.clone();
        state.v_cache = v.clone();

        // === 4. GQA expanze ===
        // Každou KV hlavu zopakujeme gqa_ratio krát (2 → 12)
        let k = self.expand_kv(&k)?;  // [b, n_q, full_seq, hd]
        let v = self.expand_kv(&v)?;

        // === 5. Attention skóre ===
        // Q @ K^T s muP škálováním
        let attn_weights = q.matmul(&k.transpose(2, 3)?)?;
        let attn_weights = (attn_weights * self.key_multiplier)?;

        // Kauzální maska: zakáže přístup k budoucím tokenům
        let full_seq = attn_weights.dim(D::Minus1)?;
        if seq_len > 1 {
            // Prefill: potřebujeme trojúhelníkovou masku
            let mask = Self::causal_mask(seq_len, full_seq, x.device())?;
            let attn_weights = attn_weights.broadcast_add(&mask)?;
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let output = attn_weights.matmul(&v)?;
            // [b, n_q, s, hd] → [b, s, n_q, hd] → [b, s, n_q * hd]
            let output = output.transpose(1, 2)?.reshape((batch, seq_len, self.n_q_heads * self.head_dim))?;
            Ok(output)
        } else {
            // Generování: seq_len=1, maska není potřeba
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let output = attn_weights.matmul(&v)?;
            let output = output.transpose(1, 2)?.reshape((batch, 1, self.n_q_heads * self.head_dim))?;
            Ok(output)
        }

    }

    /// Rozšíří KV hlavy pro GQA: [b, n_kv, s, hd] → [b, n_q, s, hd]
    fn expand_kv(&self, x: &Tensor) -> Result<Tensor> {
        if self.gqa_ratio == 1 {
            return Ok(x.clone());
        }
        // Přidáme dimenzi pro opakování, pak flatten
        let (b, n_kv, s, hd) = x.dims4()?;
        x.unsqueeze(2)?                              // [b, n_kv, 1, s, hd]
            .expand((b, n_kv, self.gqa_ratio, s, hd))?  // [b, n_kv, ratio, s, hd]
            .reshape((b, self.n_q_heads, s, hd))                // [b, n_q, s, hd]
    }

    /// Kauzální maska: -inf nad diagonálou.
    fn causal_mask(seq_len: usize, full_len: usize, device: &Device) -> Result<Tensor> {
        let offset = full_len - seq_len;
        // Vytvoříme matici [seq_len, full_len] kde:
        // mask[i][j] = 0.0 pokud j <= i + offset, jinak -inf
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..full_len).map(move |j| {
                    if j <= i + offset { 0.0 } else { f32::NEG_INFINITY }
                })
            })
            .collect();
        Tensor::new(mask.as_slice(), device)?
            .reshape((1, 1, seq_len, full_len)) // broadcast přes batch a heads
    }
}