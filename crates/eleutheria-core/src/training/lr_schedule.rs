//! Learning rate schedule pro `EleutheriaAdamW`.
//!
//! Po RN-008 víme, že Phase 2 overshoot v multi-stage tréninku je
//! **dataset-driven**, ne Adam-driven (KI-008 původní hypotéza). Restored
//! Adam state nepomohl. Jediná spolehlivá cesta k eliminaci overshoot
//! je **LR warmup** — pomalé zvyšování LR z 0 na cílovou hodnotu, aby
//! velocity buffer Adamu nestihl naskočit na strong gradient prvních
//! kroků.
//!
//! ## Konvence
//!
//! - **Per-run step counter:** schedule počítá `current_step` v rámci
//!   aktuálního běhu (od 0 do `total_steps - 1`), **ne** globální
//!   `EleutheriaAdamW.step_t`. Důsledek: každý resume run prochází
//!   warmupem znovu. To je záměrné — schedule je tréninkový režim, ne
//!   globální kontinuita Adam state.
//! - **Linear warmup** matchuje konvenci HuggingFace Trainer:
//!   `lr(n) = target * n / warmup_steps` pro `n ∈ [0, warmup_steps]`.
//!   Step 0 dostane LR=0 (no update), step `warmup_steps` plný target.
//! - **Cosine decay** od `target_lr` po `min_lr` přes
//!   `[warmup_steps, total_steps)`. Standard `0.5 * (1 + cos(π * progress))`.
//!
//! ## Kdy použít co
//!
//! - **None / Constant** — debugging, reprodukce alpha.16 chování.
//! - **Warmup** — první volba pro multi-stage curriculum (cross-domain
//!   resume bez overshoot).
//! - **WarmupCosine** — dlouhé single-domain tréninky kde chceš v
//!   závěru jemnou konvergenci pod baseline.

use std::f64::consts::PI;

/// Druh LR schedule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LrScheduleKind {
    /// LR konstantní (alpha.16 chování). Pro debugging / reprodukci.
    Constant,
    /// Lineární warmup z 0 → `target_lr` přes `warmup_steps`, pak
    /// konstantně `target_lr` až do konce.
    Warmup,
    /// Lineární warmup, pak cosine decay z `target_lr` → `min_lr`.
    WarmupCosine,
}

/// Schedule pro per-step LR.
///
/// Postaví se před tréninkem (CLI ví `total_steps` z velikosti datasetu)
/// a `train_core_memory` volá `lr_at_step` před každým `optimizer.step()`.
#[derive(Debug, Clone)]
pub struct LrSchedule {
    /// Cílová learning rate (po warmupu, pre-decay).
    pub target_lr: f64,
    /// Počet warmup stepů (lineární ramp 0 → target_lr).
    /// Pokud 0, warmup se přeskočí.
    pub warmup_steps: usize,
    /// Celkový počet optimizer stepů v tomto běhu (ne kumulativně).
    pub total_steps: usize,
    /// Minimální LR pro cosine decay floor. Pokud == `target_lr` nebo
    /// `kind == Constant/Warmup`, decay se neaplikuje.
    pub min_lr: f64,
    /// Druh schedule.
    pub kind: LrScheduleKind,
}

impl LrSchedule {
    /// Konstantní LR — žádný warmup ani decay (alpha.16 chování).
    pub fn constant(target_lr: f64) -> Self {
        Self {
            target_lr,
            warmup_steps: 0,
            total_steps: 1,
            min_lr: target_lr,
            kind: LrScheduleKind::Constant,
        }
    }

    /// Pouze linear warmup (bez decay).
    pub fn warmup(target_lr: f64, warmup_steps: usize) -> Self {
        Self {
            target_lr,
            warmup_steps,
            total_steps: warmup_steps.saturating_add(1),
            min_lr: target_lr,
            kind: LrScheduleKind::Warmup,
        }
    }

    /// Linear warmup + cosine decay až k `min_lr`.
    pub fn warmup_cosine(
        target_lr: f64,
        warmup_steps: usize,
        total_steps: usize,
        min_lr: f64,
    ) -> Self {
        Self {
            target_lr,
            warmup_steps,
            total_steps: total_steps.max(warmup_steps + 1),
            min_lr,
            kind: LrScheduleKind::WarmupCosine,
        }
    }

    /// LR pro daný step (0-indexovaný v rámci aktuálního běhu).
    ///
    /// Konvence:
    /// - `step ∈ [0, warmup_steps)`: lineární ramp 0 → target. Step 0
    ///   dostane 0, step `warmup_steps - 1` dostane
    ///   `target * (warmup_steps - 1) / warmup_steps`.
    /// - `step == warmup_steps`: přesně `target_lr` (peak).
    /// - `step > warmup_steps`: konstantní (Warmup) nebo cosine decay
    ///   (WarmupCosine).
    /// - `step >= total_steps`: clamp na `min_lr` (WarmupCosine) nebo
    ///   `target_lr` (jinak) — bezpečnost při off-by-one.
    pub fn lr_at_step(&self, step: usize) -> f64 {
        match self.kind {
            LrScheduleKind::Constant => self.target_lr,
            LrScheduleKind::Warmup => {
                if self.warmup_steps == 0 || step >= self.warmup_steps {
                    self.target_lr
                } else {
                    self.target_lr * step as f64 / self.warmup_steps as f64
                }
            }
            LrScheduleKind::WarmupCosine => {
                if self.warmup_steps > 0 && step < self.warmup_steps {
                    self.target_lr * step as f64 / self.warmup_steps as f64
                } else if step >= self.total_steps {
                    self.min_lr
                } else {
                    let decay_steps = self.total_steps - self.warmup_steps;
                    let progress = (step - self.warmup_steps) as f64 / decay_steps as f64;
                    let progress = progress.min(1.0);
                    let cos_factor = 0.5 * (1.0 + (progress * PI).cos());
                    self.min_lr + (self.target_lr - self.min_lr) * cos_factor
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn constant_returns_target_at_every_step() {
        let s = LrSchedule::constant(1e-3);
        for step in [0, 1, 50, 1_000, 10_000] {
            assert!(close(s.lr_at_step(step), 1e-3, 1e-12));
        }
    }

    #[test]
    fn warmup_zero_steps_jumps_to_target() {
        let s = LrSchedule::warmup(1e-3, 0);
        // Při warmup_steps=0 nemá co rampovat — všechny stepy dostanou target.
        assert!(close(s.lr_at_step(0), 1e-3, 1e-12));
        assert!(close(s.lr_at_step(5), 1e-3, 1e-12));
    }

    #[test]
    fn warmup_step_zero_is_zero() {
        let s = LrSchedule::warmup(1e-3, 50);
        // HF konvence: step 0 dostane 0/N = 0 (no update), step N+ plný target.
        assert!(close(s.lr_at_step(0), 0.0, 1e-12));
    }

    #[test]
    fn warmup_linear_ramp_at_midpoint() {
        let s = LrSchedule::warmup(1e-3, 50);
        // step 25 → target * 25/50 = target/2.
        assert!(close(s.lr_at_step(25), 5e-4, 1e-12));
    }

    #[test]
    fn warmup_reaches_target_at_end() {
        let s = LrSchedule::warmup(1e-3, 50);
        // step == warmup_steps: plný target.
        assert!(close(s.lr_at_step(50), 1e-3, 1e-12));
        // step > warmup_steps: konstantně target.
        assert!(close(s.lr_at_step(100), 1e-3, 1e-12));
    }

    #[test]
    fn warmup_cosine_first_phase_matches_warmup() {
        let s_w = LrSchedule::warmup(1e-3, 50);
        let s_wc = LrSchedule::warmup_cosine(1e-3, 50, 200, 1e-5);
        for step in 0..50 {
            assert!(
                close(s_w.lr_at_step(step), s_wc.lr_at_step(step), 1e-12),
                "warmup phase se má lišit pro WarmupCosine"
            );
        }
    }

    #[test]
    fn warmup_cosine_peak_at_warmup_end() {
        let s = LrSchedule::warmup_cosine(1e-3, 50, 200, 1e-5);
        // step == warmup_steps: peak target_lr (cosine progress=0 → cos(0)=1
        // → factor=1.0 → lr = min + (target - min)*1 = target).
        assert!(close(s.lr_at_step(50), 1e-3, 1e-9));
    }

    #[test]
    fn warmup_cosine_midpoint_is_halfway() {
        let s = LrSchedule::warmup_cosine(1.0, 0, 100, 0.0);
        // Bez warmupu: step 50 (mid) → cos(π * 0.5) = 0 → factor = 0.5
        // → lr = 0 + (1 - 0) * 0.5 = 0.5.
        assert!(close(s.lr_at_step(50), 0.5, 1e-9));
    }

    #[test]
    fn warmup_cosine_decays_to_min_at_total() {
        let s = LrSchedule::warmup_cosine(1e-3, 50, 200, 1e-5);
        // step == total_steps: cos(π * 1) = -1 → factor = 0 → lr = min_lr.
        assert!(close(s.lr_at_step(200), 1e-5, 1e-9));
        // Beyond: clamp na min_lr.
        assert!(close(s.lr_at_step(500), 1e-5, 1e-12));
    }

    #[test]
    fn warmup_cosine_monotonic_descent_after_warmup() {
        let s = LrSchedule::warmup_cosine(1e-3, 10, 100, 1e-5);
        let mut prev = s.lr_at_step(10); // peak
        for step in 11..=100 {
            let lr = s.lr_at_step(step);
            assert!(
                lr <= prev + 1e-12,
                "step {step}: lr={lr} > prev={prev} (cosine má klesat monotonně)"
            );
            prev = lr;
        }
    }

    #[test]
    fn warmup_cosine_monotonic_ascent_during_warmup() {
        let s = LrSchedule::warmup_cosine(1e-3, 50, 200, 1e-5);
        let mut prev = s.lr_at_step(0);
        for step in 1..=50 {
            let lr = s.lr_at_step(step);
            assert!(
                lr >= prev - 1e-12,
                "step {step}: lr={lr} < prev={prev} (warmup má růst monotonně)"
            );
            prev = lr;
        }
    }

    /// Edge case: warmup_steps == total_steps (žádná decay fáze).
    /// Schedule clampuje total_steps na warmup+1, takže pokud uživatel
    /// zadá špatně, neudusí se to.
    #[test]
    fn warmup_cosine_clamps_total_to_at_least_warmup_plus_one() {
        let s = LrSchedule::warmup_cosine(1e-3, 100, 50, 1e-5); // total < warmup
        assert!(s.total_steps > s.warmup_steps);
    }
}
