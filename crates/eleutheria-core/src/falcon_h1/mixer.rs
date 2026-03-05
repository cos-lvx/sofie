//! Mamba-2 SSM Mixer pro Falcon-H1.
//! Pipeline: in_proj → conv1d → silu → SSM scan → RMSNormGated → out_proj
//! Rekurentní mód (token po tokenu) — žádný chunk SSD.

use candle_core::{DType, Device, Result, Tensor, D};
#[allow(unused_imports)]
use candle_core::IndexOp;
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

    // === muP multipliery ===
    /// muP: škálování vstupu před in_proj
    ssm_in_multiplier: f64,
    /// muP: předpočítaný vektor [d_in_proj] pro škálování segmentů z/x/B/C/dt po in_proj
    mup_vector: Tensor,
}

impl Mixer {
    pub fn load(
        config: &super::config::FalconH1Config,
        vb: VarBuilder,
        _device: &Device,
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

        // Předpočítej mup_vector: [1, d_in_proj]
        let ssm_multipliers = &config.ssm_multipliers;
        let gs = n_groups * d_state;
        let mut mup_vec = vec![1.0f32; d_in_proj];
        // z segment: [0..d_ssm]
        for i in 0..d_ssm { mup_vec[i] *= ssm_multipliers[0] as f32; }
        // x segment: [d_ssm..2*d_ssm]
        for i in d_ssm..2 * d_ssm { mup_vec[i] *= ssm_multipliers[1] as f32; }
        // B segment: [2*d_ssm..2*d_ssm+gs]
        for i in 2 * d_ssm..2 * d_ssm + gs { mup_vec[i] *= ssm_multipliers[2] as f32; }
        // C segment: [2*d_ssm+gs..2*d_ssm+2*gs]
        for i in 2 * d_ssm + gs..2 * d_ssm + 2 * gs { mup_vec[i] *= ssm_multipliers[3] as f32; }
        // dt segment: [2*d_ssm+2*gs..]
        for i in 2 * d_ssm + 2 * gs..d_in_proj { mup_vec[i] *= ssm_multipliers[4] as f32; }
        let mup_vector = Tensor::new(mup_vec.as_slice(), &candle_core::Device::Cpu)?;

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
            ssm_in_multiplier: config.ssm_in_multiplier,
            mup_vector,
        })
    }
}

impl Mixer {
    /// Forward pass — rekurentní mód (token po tokenu).
    /// x: [batch, 1, hidden_size]
    /// state: LayerState s conv_state a ssm_state
    pub fn forward(&self, x: &Tensor, state: &mut LayerState) -> Result<Tensor> {
        let (_batch, _seq, _hidden) = x.dims3()?;
        // Squeeze seq dim: [b, 1, h] → [b, h]
        let x = x.squeeze(1)?;

        // === 1. muP vstupní škálování → projekce → mup_vector ===
        let in_scale = Tensor::new(&[self.ssm_in_multiplier as f32], x.device())?
            .to_dtype(x.dtype())?;
        let x = x.broadcast_mul(&in_scale)?;
        let proj = self.in_proj.forward(&x)?;  // [b, 6680]

        // muP per-segment škálování (z/x/B/C/dt)
        let mup = self.mup_vector.to_device(proj.device())?.to_dtype(proj.dtype())?;
        let proj = proj.broadcast_mul(&mup)?;

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
    /// Kauzální conv1d krok: roll left + write new + convolve.
    /// xbc: [batch, d_inner] — nový token
    /// state.conv_state: [d_inner, d_conv] — sliding window
    fn conv1d_step(&self, xbc: &Tensor, state: &mut LayerState) -> Result<Tensor> {
        let xbc_col = xbc.squeeze(0)?.unsqueeze(1)?; // [d_inner, 1]

        // Roll left + write new (jako HF: conv_states.roll(-1); conv_states[:,:,-1] = new)
        let shifted = state.conv_state.narrow(1, 1, self.d_conv - 1)?; // [d_inner, d_conv-1]
        let new_state = Tensor::cat(&[&shifted, &xbc_col], 1)?;       // [d_inner, d_conv]
        state.conv_state = new_state.clone();

        // Convolve: element-wise multiply + sum over d_conv
        let w = self.conv1d_weight.squeeze(1)?; // [d_inner, d_conv]
        let out = new_state.broadcast_mul(&w)?.sum(1)?; // [d_inner]
        let out = out.broadcast_add(&self.conv1d_bias)?;

        out.unsqueeze(0) // [1, d_inner]
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

impl Mixer {
    /// Forward pass — parallel prefill mód (celá sekvence najednou).
    /// x: [batch, seq_len, hidden_size] kde seq_len > 1
    /// Klíčová výhoda: konvoluce a SSM scan proběhnou jednou přes celou sekvenci,
    /// místo N×24 průchodů s kumulací BF16 chyb.
    pub fn forward_prefill(&self, x: &Tensor, state: &mut LayerState) -> Result<Tensor> {
        let (batch, seq_len, _hidden) = x.dims3()?;

        // === 1. muP vstupní škálování → projekce → mup_vector ===
        let in_scale = Tensor::new(&[self.ssm_in_multiplier as f32], x.device())?
            .to_dtype(x.dtype())?;
        let x_scaled = x.broadcast_mul(&in_scale)?;  // [b, s, hidden]
        let proj = self.in_proj.forward(&x_scaled)?;  // [b, s, 6680]

        let mup = self.mup_vector.to_device(proj.device())?.to_dtype(proj.dtype())?;
        let proj = proj.broadcast_mul(&mup)?;

        // Split: z | xBC | dt
        let z = proj.narrow(D::Minus1, 0, self.d_ssm)?;                              // [b, s, 3072]
        let xbc = proj.narrow(D::Minus1, self.d_ssm, self.d_inner)?;                 // [b, s, 3584]
        let dt = proj.narrow(D::Minus1, self.d_ssm + self.d_inner, self.n_heads)?;   // [b, s, 24]

        // === 2. Kauzální conv1d na celé sekvenci ===
        // Transponuj: [b, s, d_inner] → [b, d_inner, s]
        let xbc_t = xbc.transpose(1, 2)?;

        // Ulož conv_state: poslední d_conv tokenů raw xBC (před konvolucí a silu)
        // Toto se použije pro následující decode fázi
        if seq_len >= self.d_conv {
            state.conv_state = xbc_t.narrow(2, seq_len - self.d_conv, self.d_conv)?
                .squeeze(0)?;  // [d_inner, d_conv]
        } else {
            let pad = Tensor::zeros(
                (batch, self.d_inner, self.d_conv - seq_len),
                xbc_t.dtype(), xbc_t.device(),
            )?;
            state.conv_state = Tensor::cat(&[&pad, &xbc_t], 2)?.squeeze(0)?;
        }

        // Kauzální padding: přidáme d_conv-1 nul na začátek (inicializace kontextu)
        let pad_zeros = Tensor::zeros(
            (batch, self.d_inner, self.d_conv - 1),
            xbc_t.dtype(), xbc_t.device(),
        )?;
        let xbc_padded = Tensor::cat(&[&pad_zeros, &xbc_t], 2)?;  // [b, d_inner, d_conv-1+s]

        // Depthwise conv1d: groups=d_inner, bez paddingu (ruční padding výše)
        let w = self.conv1d_weight.to_dtype(xbc_padded.dtype())?;
        let xbc_conv = xbc_padded.conv1d(&w, 0, 1, 1, self.d_inner)?;  // [b, d_inner, s]

        // Přidej bias ručně: [d_inner] → [1, d_inner, 1]
        let bias = self.conv1d_bias.to_dtype(xbc_conv.dtype())?
            .unsqueeze(0)?.unsqueeze(2)?;
        let xbc_conv = xbc_conv.broadcast_add(&bias)?;

        // Transponuj zpět: [b, s, d_inner]
        let xbc_conv = xbc_conv.transpose(1, 2)?;

        // === 3. SiLU aktivace ===
        let xbc_conv = silu(&xbc_conv)?;  // [b, s, d_inner]

        // === 4. Split xBC → x_ssm, B, C ===
        let x_ssm = xbc_conv.narrow(D::Minus1, 0, self.d_ssm)?;                   // [b, s, 3072]
        let group_state = self.n_groups * self.d_state;                              // 256
        let b_param = xbc_conv.narrow(D::Minus1, self.d_ssm, group_state)?;         // [b, s, 256]
        let c_param = xbc_conv.narrow(D::Minus1, self.d_ssm + group_state, group_state)?; // [b, s, 256]

        // === 5. Diskretizace dt pro celou sekvenci ===
        let dt_bias = self.dt_bias.to_dtype(DType::F32)?
            .unsqueeze(0)?.unsqueeze(0)?;  // [1, 1, n_heads]
        let dt_f32 = dt.to_dtype(DType::F32)?.broadcast_add(&dt_bias)?;  // [b, s, n_heads]
        let dt_f32 = softplus(&dt_f32)?;

        // A = -exp(A_log): decay rate per head [n_heads]
        let a_f32 = self.a_log.to_dtype(DType::F32)?.exp()?.neg()?;

        // dA = exp(dt * A): [b, s, n_heads]
        let a_bcast = a_f32.unsqueeze(0)?.unsqueeze(0)?;  // [1, 1, n_heads]
        let da_seq = dt_f32.broadcast_mul(&a_bcast)?.exp()?;  // [b, s, n_heads]

        // D parametr: [n_heads]
        let d_f32 = self.d_param.to_dtype(DType::F32)?;
        let d_bcast = d_f32.unsqueeze(0)?.unsqueeze(2)?;  // [1, n_heads, 1]

        // === 6. SSM scan — sekvenční smyčka přes seq_len (jedna vrstva) ===
        // Inicializuj h z aktuálního stavu: [n_heads, headdim, d_state] → [b, n_heads, headdim, d_state]
        let mut h = state.ssm_state.to_dtype(DType::F32)?.unsqueeze(0)?;

        let mut ys: Vec<Tensor> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Extrahuj token t (narrow je bezpečnější než i() pro usize indexování)
            let x_t = x_ssm.narrow(1, t, 1)?.squeeze(1)?.to_dtype(DType::F32)?;    // [b, d_ssm]
            let b_t = b_param.narrow(1, t, 1)?.squeeze(1)?.to_dtype(DType::F32)?;  // [b, group_state]
            let c_t = c_param.narrow(1, t, 1)?.squeeze(1)?.to_dtype(DType::F32)?;  // [b, group_state]
            let da_t = da_seq.narrow(1, t, 1)?.squeeze(1)?;                         // [b, n_heads]
            let dt_t = dt_f32.narrow(1, t, 1)?.squeeze(1)?;                         // [b, n_heads]

            // Reshape pro SSM operace
            let x_heads = x_t.reshape((batch, self.n_heads, self.headdim))?.unsqueeze(3)?;
            // [b, n_heads, headdim, 1]

            let b = b_t.reshape((batch, self.n_groups, 1, self.d_state))?;
            // [b, n_groups, 1, d_state]

            let dt_exp = dt_t.unsqueeze(2)?.unsqueeze(3)?;   // [b, n_heads, 1, 1]
            let da_exp = da_t.unsqueeze(2)?.unsqueeze(3)?;   // [b, n_heads, 1, 1]

            // dB = dt * B: [b, n_heads, 1, d_state]
            let db = dt_exp.broadcast_mul(&b)?;

            // h' = dA * h + dB ⊗ x
            let h_decay = h.broadcast_mul(&da_exp)?;
            let bx = x_heads.broadcast_mul(&db)?;       // outer: [b, n_heads, headdim, d_state]
            h = (h_decay + bx)?;                        // [b, n_heads, headdim, d_state]

            // y = C * h + D * x
            let c = c_t.reshape((batch, self.n_groups, 1, self.d_state))?;
            let y_t = h.broadcast_mul(&c)?.sum(D::Minus1)?;  // [b, n_heads, headdim]

            let x_skip = x_t.reshape((batch, self.n_heads, self.headdim))?;
            let y_t = (y_t + x_skip.broadcast_mul(&d_bcast)?)?;  // [b, n_heads, headdim]

            // Flatten + cast zpět na pracovní dtype
            let y_t = y_t.reshape((batch, self.d_ssm))?.to_dtype(x.dtype())?;
            ys.push(y_t);
        }

        // Ulož finální SSM stav (F32 → dtype stavu)
        let state_dtype = state.ssm_state.dtype();
        state.ssm_state = h.squeeze(0)?.to_dtype(state_dtype)?;  // [n_heads, headdim, d_state]

        // Stack výstupů do sekvence: list<[b, d_ssm]> → [b, s, d_ssm]
        let y = Tensor::stack(&ys, 1)?;

        // === 7. Gated normalizace ===
        // y a z jsou obě [b, s, d_ssm] — norm zpracuje celou sekvenci najednou
        let y = self.norm.forward(&y, &z)?;

        // === 8. Výstupní projekce ===
        let y = self.out_proj.forward(&y)?;  // [b, s, hidden]

        Ok(y)
    }
}

/// Silu aktivace: silu(x) = x * sigmoid(x)
fn silu(x: &Tensor) -> Result<Tensor> {
    let orig_dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let sigmoid = x.neg()?.exp()?.affine(1.0, 1.0)?.recip()?;
    x.broadcast_mul(&sigmoid)?.to_dtype(orig_dtype)
}

/// Softplus: softplus(x) = ln(1 + exp(x))
fn softplus(x: &Tensor) -> Result<Tensor> {
    let orig_dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    x.exp()?.affine(1.0, 1.0)?.log()?.to_dtype(orig_dtype)
}