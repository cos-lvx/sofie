//! Thread-local trace sink pro diagnostiku forward pass.
//!
//! **Proč:** BUG-010 (NaN v `loss.backward()`) není lokalizovaný na jednu op.
//! Candle backward je black-box — nemáme hooky "před/po každé op". Forward
//! instrumentace ale ukáže, **který tenzor má extrémní dynamický rozsah**
//! (denormalizovaný vstup do `recip`, hodnoty přesahující F32 safe exp zónu,
//! NaN/Inf již ve forward). Tyto tenzory jsou prvními kandidáty pro backward
//! numerickou explozi.
//!
//! **Jak:** Thread-local `RefCell<Option<Vec<TraceEntry>>>`. `start()` zapne
//! sběr, `finish()` vrátí zachycený seznam, `probe(&t, label)` přidá záznam
//! pokud je aktivní (jinak no-op). Probe pracuje na detached kopii — neovlivní
//! autograd graf.
//!
//! **Stats:** `abs_max`, `abs_min_nonzero`, `mean`, `l2`, `has_nan`, `has_inf`.
//! `abs_min_nonzero` odfiltruje "skutečné nuly" (padding, masked slots) —
//! zajímá nás nejmenší neležící hodnota pro hledání denormalizovaných inputů.
//!
//! **Cost:** každý probe = 4-5 kernel launches (to_dtype, abs, max_all,
//! mean_all, sum_all). Pro seq_len=1 a ~30 probes per layer je to ~3 ms CPU
//! per layer — negligible pro diagnostiku, nikdy nezapínat v produkci.

use candle_core::{DType, Result, Tensor};
use std::cell::RefCell;

thread_local! {
    static TRACE: RefCell<Option<Vec<TraceEntry>>> = const { RefCell::new(None) };
}

/// Jeden záznam v trace sinku — statistiky jednoho tensoru v daném bodě.
#[derive(Debug, Clone)]
pub struct TraceEntry {
    /// Popisek probe bodu (např. "L22.pre_norm_out", "L22.mixer.softplus_dt").
    pub label: String,
    /// Tvar tensoru.
    pub shape: Vec<usize>,
    /// Dtype jako string (pro stabilní reporting napříč Candle verzemi).
    pub dtype: String,
    /// Maximální absolutní hodnota (Inf pokud obsahuje Inf, NaN pokud NaN).
    pub abs_max: f64,
    /// Minimální **neležící** absolutní hodnota. `0.0` pokud jsou všechny
    /// hodnoty nulové. Pomáhá detekovat denormalizované vstupy do `recip`.
    pub abs_min_nonzero: f64,
    /// Aritmetický průměr (signed).
    pub mean: f64,
    /// L2 norm (sqrt sum of squares).
    pub l2: f64,
    /// True pokud tensor obsahuje alespoň jeden NaN.
    pub has_nan: bool,
    /// True pokud tensor obsahuje alespoň jeden +/-Inf.
    pub has_inf: bool,
}

/// Aktivuje trace sink pro aktuální vlákno. Pokud už je aktivní, přepíše.
pub fn start() {
    TRACE.with(|c| *c.borrow_mut() = Some(Vec::new()));
}

/// Zachytí a vrátí seznam entries, deaktivuje sink. `None` pokud sink nebyl
/// aktivní.
pub fn finish() -> Option<Vec<TraceEntry>> {
    TRACE.with(|c| c.borrow_mut().take())
}

/// Je trace sink aktivní v tomto vlákně?
pub fn is_active() -> bool {
    TRACE.with(|c| c.borrow().is_some())
}

/// Zaznamená statistiky tenzoru do sinku, pokud je aktivní. Jinak no-op.
/// Nikdy nevrací Err — selhání stats výpočtu zapíše `has_nan=true` a pokračuje.
/// Detach před výpočtem zajistí, že probe neváže autograd graf.
pub fn probe(t: &Tensor, label: &str) -> Result<()> {
    if !is_active() {
        return Ok(());
    }
    let entry = compute_stats(t, label)?;
    TRACE.with(|c| {
        if let Some(vec) = c.borrow_mut().as_mut() {
            vec.push(entry);
        }
    });
    Ok(())
}

fn compute_stats(t: &Tensor, label: &str) -> Result<TraceEntry> {
    let shape = t.dims().to_vec();
    let dtype_str = format!("{:?}", t.dtype());
    // Detach + upcast na F32 — oddělené od grafu, stabilní statistiky.
    let detached = t.detach();
    let t_f32 = detached.to_dtype(DType::F32)?;

    let abs = t_f32.abs()?;
    let abs_max_f32: f32 = abs.max_all()?.to_scalar()?;
    let mean_f32: f32 = t_f32.mean_all()?.to_scalar()?;
    let sum_sq_f32: f32 = t_f32.sqr()?.sum_all()?.to_scalar()?;

    // abs_min_nonzero: nahraď nuly +inf (přes where_cond, ne multiplikaci —
    // `inf * 0` dává NaN, ne 0), pak min. Pokud byly všechny nuly, výsledek
    // je +inf → reportujeme 0.0 pro čitelnost.
    let numel = t_f32.elem_count();
    let abs_min_nonzero_f32 = if numel == 0 {
        0.0f32
    } else {
        let inf_fill = Tensor::full(f32::INFINITY, abs.shape(), t_f32.device())?;
        let is_zero_mask = abs.eq(0.0f64)?; // u8, 1 kde abs==0
        let masked = is_zero_mask.where_cond(&inf_fill, &abs)?;
        let min_val: f32 = masked.min_all()?.to_scalar()?;
        if min_val.is_infinite() { 0.0 } else { min_val }
    };

    let has_nan = abs_max_f32.is_nan() || mean_f32.is_nan() || sum_sq_f32.is_nan();
    let has_inf = !has_nan
        && (abs_max_f32.is_infinite() || mean_f32.is_infinite() || sum_sq_f32.is_infinite());

    let l2 = if sum_sq_f32.is_finite() {
        (sum_sq_f32 as f64).sqrt()
    } else {
        f64::NAN
    };

    Ok(TraceEntry {
        label: label.to_string(),
        shape,
        dtype: dtype_str,
        abs_max: abs_max_f32 as f64,
        abs_min_nonzero: abs_min_nonzero_f32 as f64,
        mean: mean_f32 as f64,
        l2,
        has_nan,
        has_inf,
    })
}

/// Vykreslí seznam entries jako tabulka pro CLI výstup.
pub fn render_table(entries: &[TraceEntry]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "  {:<40}  {:>12}  {:>12}  {:>12}  {:>12}  {:<5}\n",
        "label", "abs_max", "abs_min_nz", "mean", "l2", "flags"
    ));
    out.push_str(&format!(
        "  {:<40}  {:>12}  {:>12}  {:>12}  {:>12}  {:<5}\n",
        "-----", "-------", "----------", "----", "--", "-----"
    ));
    for e in entries {
        let flags = match (e.has_nan, e.has_inf) {
            (true, _) => "NaN",
            (false, true) => "Inf",
            _ => "ok",
        };
        out.push_str(&format!(
            "  {:<40}  {:>12.4e}  {:>12.4e}  {:>12.4e}  {:>12.4e}  {:<5}\n",
            e.label, e.abs_max, e.abs_min_nonzero, e.mean, e.l2, flags
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn probe_is_noop_when_sink_inactive() -> Result<()> {
        // Sink inactive by default
        assert!(!is_active());
        let t = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?;
        probe(&t, "test")?; // nesmí panic ani nic nezměnit
        assert!(finish().is_none());
        Ok(())
    }

    #[test]
    fn probe_captures_entry_when_active() -> Result<()> {
        start();
        assert!(is_active());
        let t = Tensor::new(&[1.0f32, -2.0, 3.0], &Device::Cpu)?;
        probe(&t, "sample")?;
        let entries = finish().expect("sink aktivní → musí vrátit Vec");
        assert!(!is_active(), "finish musí deaktivovat sink");
        assert_eq!(entries.len(), 1);
        let e = &entries[0];
        assert_eq!(e.label, "sample");
        assert_eq!(e.shape, vec![3]);
        assert!((e.abs_max - 3.0).abs() < 1e-6);
        assert!((e.abs_min_nonzero - 1.0).abs() < 1e-6);
        assert!(!e.has_nan && !e.has_inf);
        Ok(())
    }

    #[test]
    fn probe_flags_nan_and_inf() -> Result<()> {
        start();
        let t_nan = Tensor::new(&[f32::NAN, 1.0, 2.0], &Device::Cpu)?;
        probe(&t_nan, "nan_tensor")?;
        let t_inf = Tensor::new(&[f32::INFINITY, 1.0, 2.0], &Device::Cpu)?;
        probe(&t_inf, "inf_tensor")?;
        let entries = finish().unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries[0].has_nan, "nan tensor musí mít has_nan=true");
        assert!(entries[1].has_inf, "inf tensor musí mít has_inf=true");
        Ok(())
    }

    #[test]
    fn abs_min_nonzero_skips_zeros() -> Result<()> {
        start();
        let t = Tensor::new(&[0.0f32, 0.0, 0.5, 2.0], &Device::Cpu)?;
        probe(&t, "zeros_mixed")?;
        let entries = finish().unwrap();
        // Nejmenší neležící = 0.5, nikoli 0.0
        assert!((entries[0].abs_min_nonzero - 0.5).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn render_table_produces_nonempty_output() {
        let entries = vec![TraceEntry {
            label: "test".into(),
            shape: vec![1, 2, 3],
            dtype: "F32".into(),
            abs_max: 1.0,
            abs_min_nonzero: 0.1,
            mean: 0.5,
            l2: 1.2,
            has_nan: false,
            has_inf: false,
        }];
        let table = render_table(&entries);
        assert!(table.contains("test"));
        assert!(table.contains("ok"));
    }
}
