//! Minimální reprodukce NaN gradientu z BUG-010.
//!
//! Izolované unit testy, které aplikují **jednu operaci** (RMSNorm, softplus,
//! exp, silu) na vstupy s různým dynamickým rozsahem (normal, tiny, extreme
//! mixed) a verifikují, zda backward produkuje finite gradient.
//!
//! Cíl: najít konkrétní op, která v Candle backward dává NaN pro realistický
//! input z Falcon-H1 (massive activations pattern: L0 ~1e-14, L2 ~1e2).
//!
//! **Metodika:** každý test vytvoří `Var` s vybraným dtype a hodnotami,
//! aplikuje operaci, vezme skalární loss (sum nebo sum of single element)
//! a volá `loss.backward()`. Potom kontroluje finitnost gradientu Var.
//!
//! **Nálezy** (po běhu `cargo test --package eleutheria-core training::repro`)
//! — viz test názvy s `_expect_pass` / `_expect_fail`.

#![cfg(test)]

use candle_core::{DType, Device, Result, Tensor, Var};

/// Pomocník: ověří, že gradient Var je finite (ne NaN, ne Inf).
fn assert_grad_finite(
    var: &Var,
    grads: &candle_core::backprop::GradStore,
    label: &str,
) -> Result<f64> {
    let g = grads
        .get(var.as_tensor())
        .unwrap_or_else(|| panic!("{}: gradient nenalezen v GradStore", label));
    let sum_sq: f32 = g.to_dtype(DType::F32)?.sqr()?.sum_all()?.to_scalar()?;
    let norm = (sum_sq as f64).sqrt();
    println!("  {}: gradient L2 norm = {:.6e}", label, norm);
    assert!(
        norm.is_finite(),
        "{}: gradient není finite: {}",
        label,
        norm
    );
    Ok(norm)
}

/// Naše RMSNorm signature (duplicate z `norm.rs` bez weight — čistě normalizační krok).
/// Používáme identitu pro weight (všechny 1) aby izolovala numerickou dynamiku samotné normy.
fn rms_norm_no_weight(x: &Tensor, eps: f64) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let x_sq = x_f32.sqr()?;
    let mean_sq = x_sq.mean_keepdim(candle_core::D::Minus1)?;
    let scale = (mean_sq + eps)?.sqrt()?.recip()?;
    x_f32.broadcast_mul(&scale)
}

// ---------------------------------------------------------------------------
// RMSNorm backward testy
// ---------------------------------------------------------------------------

#[test]
fn rmsnorm_backward_normal_input() -> Result<()> {
    // Standardní input ~ N(0, 1), eps=1e-5 (Falcon-H1 default)
    let var = Var::randn_f64(0.0, 1.0, (8, 32), DType::F32, &Device::Cpu)?;
    let out = rms_norm_no_weight(var.as_tensor(), 1e-5)?;
    let loss = out.sum_all()?;
    let grads = loss.backward()?;
    let norm = assert_grad_finite(&var, &grads, "normal input N(0,1)")?;
    assert!(norm > 0.0);
    Ok(())
}

#[test]
fn rmsnorm_backward_tiny_input() -> Result<()> {
    // Všechny hodnoty ~ 1e-7 (L1 post-layer output magnitude)
    let data = vec![1e-7f32; 256];
    let var = Var::from_slice(&data, (8, 32), &Device::Cpu)?;
    let out = rms_norm_no_weight(var.as_tensor(), 1e-5)?;
    let loss = out.sum_all()?;
    let grads = loss.backward()?;
    let norm = assert_grad_finite(&var, &grads, "tiny input 1e-7")?;
    println!("  tiny input gradient norm: {:.6e}", norm);
    Ok(())
}

#[test]
fn rmsnorm_backward_extreme_tiny_input() -> Result<()> {
    // Hodnoty ~ 1e-14 (L0 post-layer output)
    let data = vec![1e-14f32; 256];
    let var = Var::from_slice(&data, (8, 32), &Device::Cpu)?;
    let out = rms_norm_no_weight(var.as_tensor(), 1e-5)?;
    let loss = out.sum_all()?;
    let grads = loss.backward()?;
    let norm = assert_grad_finite(&var, &grads, "extreme tiny 1e-14")?;
    println!("  extreme tiny input gradient norm: {:.6e}", norm);
    Ok(())
}

#[test]
fn rmsnorm_backward_mixed_range() -> Result<()> {
    // Mix: polovina ~ 1e-7, polovina ~ 1e2 (simulace massive activations)
    let mut data = vec![1e-7f32; 128];
    data.extend(vec![1e2f32; 128]);
    let var = Var::from_slice(&data, (8, 32), &Device::Cpu)?;
    let out = rms_norm_no_weight(var.as_tensor(), 1e-5)?;
    let loss = out.sum_all()?;
    let grads = loss.backward()?;
    let norm = assert_grad_finite(&var, &grads, "mixed 1e-7 + 1e2")?;
    println!("  mixed range gradient norm: {:.6e}", norm);
    Ok(())
}

#[test]
fn rmsnorm_backward_massive_outliers() -> Result<()> {
    // 1 outlier ~ 1e4, ostatní ~ 1e-7 (Peri-LN massive activations pattern)
    let mut data = vec![1e-7f32; 255];
    data.push(1e4f32);
    let var = Var::from_slice(&data, (8, 32), &Device::Cpu)?;
    let out = rms_norm_no_weight(var.as_tensor(), 1e-5)?;
    let loss = out.sum_all()?;
    let grads = loss.backward()?;
    let norm = assert_grad_finite(&var, &grads, "massive outlier 1e4 vs 1e-7")?;
    println!("  massive outlier gradient norm: {:.6e}", norm);
    Ok(())
}

// ---------------------------------------------------------------------------
// softplus backward
// ---------------------------------------------------------------------------

/// Naše (naivní) softplus signature: log(1 + exp(x))
fn softplus(x: &Tensor) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    x_f32.exp()?.affine(1.0, 1.0)?.log()
}

/// Numericky stabilní softplus (stejná implementace jako v `mixer.rs` po alpha.7).
/// `softplus(x) = relu(x) + log(1 + exp(-|x|))` — mathematicky identické,
/// numericky bounded ve forward i backward.
fn softplus_stable(x: &Tensor) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let zero = Tensor::zeros_like(&x_f32)?;
    let relu_x = x_f32.maximum(&zero)?;
    let abs_x = x_f32.abs()?;
    let log1p = abs_x.neg()?.exp()?.affine(1.0, 1.0)?.log()?;
    relu_x + log1p
}

#[test]
fn softplus_backward_normal() -> Result<()> {
    let var = Var::randn_f64(0.0, 1.0, (64,), DType::F32, &Device::Cpu)?;
    let out = softplus(var.as_tensor())?;
    let loss = out.sum_all()?;
    let grads = loss.backward()?;
    assert_grad_finite(&var, &grads, "softplus normal")?;
    Ok(())
}

#[test]
fn softplus_backward_large_positive() -> Result<()> {
    // x = 50 → exp(50) ≈ 5e21, potenciál overflow v F32 (max 3.4e38)
    let data = vec![50.0f32; 64];
    let var = Var::from_slice(&data, (64,), &Device::Cpu)?;
    let out = softplus(var.as_tensor())?;
    let loss = out.sum_all()?;
    let grads = loss.backward()?;
    assert_grad_finite(&var, &grads, "softplus x=50")?;
    Ok(())
}

/// Dokumentovaný Candle limit: `softplus(x) = log(1 + exp(x))` naivní
/// implementace pro x ≥ 88 přetéká v F32 (exp(88) ≈ 1.7e38 blízko F32 max).
///
/// Forward: exp(100) = Inf v F32, log(Inf + 1) = Inf (loss is Inf, ne NaN)
/// Backward: propagace přes Inf dává NaN
///
/// **Relevance pro BUG-010:** SSM discretization v Mamba-2 používá
/// `dt = softplus(dt_raw + dt_bias)`. Pokud některá Falcon-H1 vrstva má
/// `dt_bias` + `dt_raw > 88`, softplus v naší implementaci dá Inf forward
/// a NaN backward. **Pravděpodobný primary root cause BUG-010.**
///
/// **Workaround:** numericky stabilní softplus:
/// ```rust,ignore
/// fn softplus_stable(x: &Tensor) -> Result<Tensor> {
///     // For x > 20, softplus(x) ≈ x (exponent je negligibly small)
///     // For x < -20, softplus(x) ≈ 0
///     // Between: standard log(1 + exp(x))
///     x.clamp(-20.0, 20.0)?.exp()?.affine(1.0, 1.0)?.log()
/// }
/// ```
#[test]
#[should_panic(expected = "gradient není finite")]
fn softplus_backward_extreme_positive_produces_nan() {
    let data = vec![100.0f32; 64];
    let var = Var::from_slice(&data, (64,), &Device::Cpu).unwrap();
    let out = softplus(var.as_tensor()).unwrap();
    let loss = out.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    assert_grad_finite(&var, &grads, "softplus x=100 (overflow zóna)").unwrap();
}

/// **Fix verification** — numericky stabilní softplus přežije x=100 bez NaN.
/// Forward i backward produkují finite hodnoty. To je oprava BUG-010 primary
/// root cause.
#[test]
fn softplus_stable_backward_extreme_positive_finite() -> Result<()> {
    let data = vec![100.0f32; 64];
    let var = Var::from_slice(&data, (64,), &Device::Cpu)?;
    let out = softplus_stable(var.as_tensor())?;

    // Forward correctness: softplus(100) ≈ 100 (error < 2e-44)
    let out_val: f32 = out.mean_all()?.to_scalar()?;
    assert!(
        (out_val - 100.0).abs() < 1e-3,
        "stable softplus(100) = {}, očekáváno ~100",
        out_val
    );

    let loss = out.sum_all()?;
    let grads = loss.backward()?;
    let norm = assert_grad_finite(&var, &grads, "stable softplus x=100")?;
    println!("  stable softplus(100) gradient: {:.6e}", norm);
    Ok(())
}

/// Math equivalence — stable == naive pro bezpečný rozsah.
#[test]
fn softplus_stable_matches_naive_on_safe_range() -> Result<()> {
    let data: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.3).collect(); // x in [-15, 15]
    let x = Tensor::from_slice(&data, (data.len(),), &Device::Cpu)?;

    let naive = softplus(&x)?;
    let stable = softplus_stable(&x)?;

    let naive_vec: Vec<f32> = naive.to_vec1()?;
    let stable_vec: Vec<f32> = stable.to_vec1()?;

    for (a, b) in naive_vec.iter().zip(stable_vec.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "softplus divergence: naive={}, stable={}",
            a,
            b
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// exp + neg (SSM discretization pattern: A = -exp(A_log))
// ---------------------------------------------------------------------------

#[test]
fn exp_neg_chain_backward() -> Result<()> {
    let var = Var::randn_f64(0.0, 1.0, (64,), DType::F32, &Device::Cpu)?;
    let out = var.as_tensor().exp()?.neg()?;
    let loss = out.sum_all()?;
    let grads = loss.backward()?;
    assert_grad_finite(&var, &grads, "exp().neg() chain")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// recip (1/x) — používané v RMSNorm a softmax backward
// ---------------------------------------------------------------------------

#[test]
fn recip_backward_normal() -> Result<()> {
    let var = Var::randn_f64(2.0, 0.5, (64,), DType::F32, &Device::Cpu)?; // shifted away from 0
    let out = var.as_tensor().recip()?;
    let loss = out.sum_all()?;
    let grads = loss.backward()?;
    assert_grad_finite(&var, &grads, "recip normal input")?;
    Ok(())
}

/// Dokumentovaný Candle limit: `recip` backward pro input ≈ 0 → Inf gradient.
///
/// Forward: 1/1e-10 = 1e10 (stále finite v F32)
/// Backward: -1/(1e-10)² = -1e20 (překračuje bezpečný range pro amplifikaci
/// přes další ops — snadno dojde k Inf v některém downstream kroku)
///
/// **Relevance pro BUG-010:** RMSNorm používá `recip(sqrt(mean_sq + eps))`.
/// Pokud někde v síti je `sqrt(mean_sq)` ≈ 0 (po eps korekci), `recip` je OK,
/// ale backward chain může amplifikovat. V naší konkrétní konfiguraci se zdá,
/// že RMSNorm samotná to zvládá (viz passing tests výše), ale kombinace s
/// jinými ops může být problém.
#[test]
#[should_panic(expected = "gradient není finite")]
fn recip_backward_near_zero_produces_inf() {
    let data = vec![1e-10f32; 64];
    let var = Var::from_slice(&data, (64,), &Device::Cpu).unwrap();
    let out = var.as_tensor().recip().unwrap();
    let loss = out.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    assert_grad_finite(&var, &grads, "recip near-zero 1e-10").unwrap();
}

// ---------------------------------------------------------------------------
// Kompletní RMSNorm + následná lineární vrstva (simulace backward path)
// ---------------------------------------------------------------------------

#[test]
fn rmsnorm_then_linear_backward() -> Result<()> {
    // Simuluj: RMSNorm → matmul. Replikuje pattern normed = pre_norm(x), pak linear.
    let var = Var::from_slice(&vec![1e-7f32; 256], (8, 32), &Device::Cpu)?;
    let weight = Tensor::randn(0.0f32, 0.1, (32, 32), &Device::Cpu)?;
    let normed = rms_norm_no_weight(var.as_tensor(), 1e-5)?;
    let out = normed.matmul(&weight)?;
    let loss = out.sum_all()?;
    let grads = loss.backward()?;
    let norm = assert_grad_finite(&var, &grads, "RMSNorm+matmul")?;
    println!("  RMSNorm+matmul gradient: {:.6e}", norm);
    Ok(())
}
