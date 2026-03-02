//! Mamba-2 SSM Mixer pro Falcon-H1.
//! Pipeline: in_proj → conv1d → silu → SSM scan → RMSNormGated → out_proj
//! Rekurentní mód (token po tokenu) — žádný chunk SSD.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

use super::norm::RmsNormGated;
use super::state::LayerState;

/// Mamba-2 SSM mixer — rekurentní inference mód.
pub struct Mixer {
    /// Vstupní projekce: hidden → z + xBC + dt
    in_proj: Linear,
    /// Depthwise conv1d váhy: [d_inner, 1, d_conv]
    conv1d_weight: Tensor,
    /// Conv1d bias: [d_inner]
    conv1d_bias: Tensor,
    /// Bias pro dt (časový krok): [n_heads]
    dt_bias: Tensor,
    /// Log záporného A (decay rate): [n_heads]
    a_log: Tensor,
    /// D parametr (skip connection v SSM): [n_heads]
    d_param: Tensor,
    /// RMSNormGated pro výstup SSM branch
    norm: RmsNormGated,
    /// Výstupní projekce: d_ssm → hidden_size
    out_proj: Linear,

    // === Rozměry ===
    d_ssm: usize,      // 3072 = n_heads * headdim
    d_inner: usize,     // 3584 = d_ssm + 2 * n_groups * d_state
    n_heads: usize,     // 24
    headdim: usize,     // 128
    d_state: usize,     // 256
    n_groups: usize,    // 1
    d_conv: usize,      // 4
}

impl Mixer {
    pub fn load(
        config: &super::config::FalconH1Config,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let d_ssm = config.mamba_d_ssm;           // 3072
        let d_state = config.mamba_d_state;         // 256
        let n_groups = config.mamba_n_groups;        // 1
        let n_heads = config.mamba_n_heads;          // 24
        let headdim = config.mamba_d_head;           // 128
        let d_conv = config.mamba_d_conv;            // 4
        let d_inner = d_ssm + 2 * n_groups * d_state; // 3584

        // in_proj: hidden → z(d_ssm) + xBC(d_inner) + dt(n_heads)
        let d_in_proj = 2 * d_ssm + 2 * n_groups * d_state + n_heads; // 6680
        let in_proj = linear_no_bias(config.hidden_size, d_in_proj, vb.pp("in_proj"))?;

        // Conv1d: depthwise přes d_inner kanálů
        let conv1d_weight = vb.get((d_inner, 1, d_conv), "conv1d.weight")?;
        let conv1d_bias = vb.get(d_inner, "conv1d.bias")?;

        // SSM parametry
        let dt_bias = vb.get(n_heads, "dt_bias")?;
        let a_log = vb.get(n_heads, "A_log")?;
        let d_param = vb.get(n_heads, "D")?;

        // Gated norm na SSM výstupu
        let norm = RmsNormGated::load(
            d_ssm,
            config.rms_norm_eps,
            config.mamba_norm_before_gate,
            vb.pp("norm"),
        )?;

        // out_proj: d_ssm → hidden_size
        let out_proj = linear_no_bias(d_ssm, config.hidden_size, vb.pp("out_proj"))?;

        Ok(Self {
            in_proj,
            conv1d_weight,
            conv1d_bias,
            dt_bias,
            a_log,
            d_param,
            norm,
            out_proj,
            d_ssm,
            d_inner,
            n_heads,
            headdim,
            d_state,
            n_groups,
            d_conv,
        })
    }
}

impl Mixer {
    /// Forward pass — rekurentní mód (token po tokenu).
    /// x: [batch, 1, hidden_size]
    /// state: LayerState s conv_state a ssm_state
    pub fn forward(&self, x: &Tensor, state: &mut LayerState) -> Result<Tensor> {
        let (batch, _seq, _hidden) = x.dims3()?;
        // Squeeze seq dim: [b, 1, h] → [b, h]
        let x = x.squeeze(1)?;

        // === 1. Vstupní projekce ===
        let proj = self.in_proj.forward(&x)?;  // [b, 6680]

        // Split: z | xBC | dt
        let z = proj.narrow(D::Minus1, 0, self.d_ssm)?;                          // [b, 3072]
        let xbc = proj.narrow(D::Minus1, self.d_ssm, self.d_inner)?;             // [b, 3584]
        let dt = proj.narrow(D::Minus1, self.d_ssm + self.d_inner, self.n_heads)?; // [b, 24]

        // === 2. Kauzální konvoluce (přes sliding window) ===
        let xbc = self.conv1d_step(&xbc, state)?;  // [b, 3584]

        // === 3. Silu aktivace ===
        let xbc = silu(&xbc)?;

        // === 4. Split xBC → x, B, C ===
        let x_ssm = xbc.narrow(D::Minus1, 0, self.d_ssm)?;             // [b, 3072]
        let bc_offset = self.d_ssm;
        let group_state = self.n_groups * self.d_state;                   // 256
        let b_param = xbc.narrow(D::Minus1, bc_offset, group_state)?;    // [b, 256]
        let c_param = xbc.narrow(D::Minus1, bc_offset + group_state, group_state)?; // [b, 256]

        // === 5. SSM scan (rekurentní krok) ===
        let y = self.ssm_step(&x_ssm, &b_param, &c_param, &dt, state)?;  // [b, 3072]

        // === 6. Gated normalizace ===
        // y je SSM výstup, z je gate (obě [b, 3072])
        let y = self.norm.forward(&y.unsqueeze(1)?, &z.unsqueeze(1)?)?;
        let y = y.squeeze(1)?;  // [b, 3072]

        // === 7. Výstupní projekce ===
        let y = self.out_proj.forward(&y)?;  // [b, hidden_size]

        // Vrať s seq dimenzí: [b, 1, hidden]
        y.unsqueeze(1)
    }
}

impl Mixer {
    /// Kauzální conv1d krok: posune okno a aplikuje konvoluci.
    /// xbc: [batch, d_inner] — nový token
    /// state.conv_state: [d_inner, d_conv-1] — sliding window
    fn conv1d_step(&self, xbc: &Tensor, state: &mut LayerState) -> Result<Tensor> {
        // Aktualizace okna: zahoď nejstarší, přidej nový
        // conv_state: [d_inner, d_conv-1] = [3584, 3]
        // xbc: [batch, d_inner] — vezmeme první batch (batch=1 pro generování)

        let xbc_col = xbc.squeeze(0)?.unsqueeze(1)?; // [d_inner, 1]

        // Nové okno: [starý[:, 1:], nový]
        let old = state.conv_state.narrow(1, 1, self.d_conv - 2)?; // [d_inner, d_conv-2]
        let new_state = Tensor::cat(&[&old, &xbc_col], 1)?;       // [d_inner, d_conv-1]
        state.conv_state = new_state.clone();

        // Konvoluce: pro každý kanál, dot product [d_conv] vah s [d_conv] oknem
        // Okno pro konvoluci = [nový_state, xbc] = posledních d_conv tokenů
        // Ale conv_state drží d_conv-1 a xbc je aktuální → celkem d_conv
        let full_window = Tensor::cat(&[&new_state, &xbc_col], 1)?; // [d_inner, d_conv]

        // conv1d_weight: [d_inner, 1, d_conv] → squeeze na [d_inner, d_conv]
        let w = self.conv1d_weight.squeeze(1)?; // [d_inner, d_conv]

        // Depthwise: element-wise multiply + sum přes d_conv dimenzi
        let out = (full_window.broadcast_mul(&w))?.sum(1)?; // [d_inner]

        // Přidej bias
        let out = out.broadcast_add(&self.conv1d_bias)?;

        // Vrať s batch dimenzí: [1, d_inner]
        out.unsqueeze(0)
    }
}

impl Mixer {
    /// SSM rekurentní krok: h' = dA·h + dB⊗x, y = C·h + D·x
    /// x: [batch, d_ssm=3072]
    /// b_param, c_param: [batch, n_groups*d_state=256]
    /// dt: [batch, n_heads=24]
    /// state.ssm_state: [n_heads, headdim, d_state] = [24, 128, 256]
    fn ssm_step(
        &self,
        x: &Tensor,
        b_param: &Tensor,
        c_param: &Tensor,
        dt: &Tensor,
        state: &mut LayerState,
    ) -> Result<Tensor> {
        // === Diskretizace ===
        // dt: softplus(dt + dt_bias)
        let dt = dt.broadcast_add(&self.dt_bias)?;
        let dt = softplus(&dt)?;  // [b, n_heads]

        // A = -exp(A_log): záporný decay rate per head
        let a = self.a_log.exp()?.neg()?;  // [n_heads]

        // dA = exp(dt * A): [b, n_heads] → broadcast na [b, n_heads, 1, 1]
        let da = dt.broadcast_mul(&a)?.exp()?;        // [b, n_heads]
        let da = da.unsqueeze(2)?.unsqueeze(3)?;          // [b, n_heads, 1, 1]

        // === Reshape vstupů pro SSM ===
        // x: [b, d_ssm] → [b, n_heads, headdim, 1]
        let x_heads = x.reshape((1, self.n_heads, self.headdim))?
            .unsqueeze(3)?;                     // [b, n_heads, headdim, 1]

        // B: [b, 256] → [b, 1, 1, d_state] (broadcast přes heads a headdim)
        let b = b_param.reshape((1, self.n_groups, 1, self.d_state))?;

        // dB = dt * B: [b, n_heads, 1, 1] * [b, 1, 1, d_state] — ale potřebujeme dt ve správném tvaru
        let dt_expanded = dt.unsqueeze(2)?.unsqueeze(3)?; // [b, n_heads, 1, 1]
        let db = dt_expanded.broadcast_mul(&b)?;           // [b, n_heads, 1, d_state]

        // === State update: h' = dA·h + dB⊗x ===
        // h: [n_heads, headdim, d_state] → přidáme batch: [1, n_heads, headdim, d_state]
        let h = state.ssm_state.unsqueeze(0)?;

        // dA * h: [b, n_heads, 1, 1] broadcast * [1, n_heads, headdim, d_state]
        let h_decay = h.broadcast_mul(&da)?;

        // dB ⊗ x: outer product — [b, n_heads, headdim, 1] * [b, n_heads, 1, d_state]
        let bx = x_heads.broadcast_mul(&db)?;  // [b, n_heads, headdim, d_state]

        let h_new = (h_decay + bx)?;  // [b, n_heads, headdim, d_state]

        // Ulož nový stav (bez batch dimenze)
        state.ssm_state = h_new.squeeze(0)?;  // [n_heads, headdim, d_state]

        // === Výstup: y = C·h + D·x ===
        // C: [b, 256] → [b, 1, 1, d_state]
        let c = c_param.reshape((1, self.n_groups, 1, self.d_state))?;

        // C * h: [b, n_heads, headdim, d_state] * [b, 1, 1, d_state] → sum přes d_state
        let y = h_new.broadcast_mul(&c)?.sum(D::Minus1)?;  // [b, n_heads, headdim]

        // D * x: skip connection
        let x_skip = x.reshape((1, self.n_heads, self.headdim))?;
        let d = self.d_param.unsqueeze(0)?.unsqueeze(2)?;   // [1, n_heads, 1]
        let y = (y + x_skip.broadcast_mul(&d)?)?;            // [b, n_heads, headdim]

        // Flatten: [b, n_heads, headdim] → [b, d_ssm]
        y.reshape((1, self.d_ssm))
    }
}

/// Silu aktivace: silu(x) = x * sigmoid(x)
fn silu(x: &Tensor) -> Result<Tensor> {
    let sigmoid = x.neg()?.exp()?.affine(1.0, 1.0)?.recip()?;
    x.broadcast_mul(&sigmoid)
}

/// Softplus: softplus(x) = ln(1 + exp(x))
fn softplus(x: &Tensor) -> Result<Tensor> {
    x.exp()?.affine(1.0, 1.0)?.log()
}