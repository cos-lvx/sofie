//! Identity benchmark report — JSON + markdown s per-variant a per-kind
//! pass-rate breakdown, forbidden hit counter, a detailní listing odpovědí.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::bench::identity::probe::{IdentityOutcome, IdentityResult};

/// Metadata běhu — co, kdy, na čem, s jakou Core Memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityMeta {
    pub schema_version: u32,
    pub eleutheria_version: String,
    pub timestamp: String,
    pub model_name: Option<String>,
    pub device: Option<String>,
    pub core_memory_path: Option<String>,
    pub core_memory_steps: Option<usize>,
    pub core_memory_best_loss: Option<f64>,
    pub notes: Option<String>,
}

impl IdentityMeta {
    pub fn new() -> Self {
        Self {
            schema_version: 1,
            eleutheria_version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
            model_name: None,
            device: None,
            core_memory_path: None,
            core_memory_steps: None,
            core_memory_best_loss: None,
            notes: None,
        }
    }
}

impl Default for IdentityMeta {
    fn default() -> Self {
        Self::new()
    }
}

/// Kompletní identity bench report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityReport {
    pub meta: IdentityMeta,
    pub results: Vec<IdentityResult>,
}

impl IdentityReport {
    pub fn new(results: Vec<IdentityResult>) -> Self {
        Self {
            meta: IdentityMeta::new(),
            results,
        }
    }

    pub fn with_model_name(mut self, name: impl Into<String>) -> Self {
        self.meta.model_name = Some(name.into());
        self
    }

    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.meta.device = Some(device.into());
        self
    }

    pub fn with_core_memory(
        mut self,
        path: impl Into<String>,
        steps: Option<usize>,
        best_loss: Option<f64>,
    ) -> Self {
        self.meta.core_memory_path = Some(path.into());
        self.meta.core_memory_steps = steps;
        self.meta.core_memory_best_loss = best_loss;
        self
    }

    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.meta.notes = Some(notes.into());
        self
    }

    /// Serializace do JSON (pretty).
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Markdown report — meta + per-variant pass-rate + per-kind breakdown
    /// + forbidden hits + detailní listing.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "# Identity Benchmark Report");
        let _ = writeln!(out);
        let _ = writeln!(out, "- **Timestamp:** {}", self.meta.timestamp);
        let _ = writeln!(out, "- **Eleutheria:** v{}", self.meta.eleutheria_version);
        if let Some(m) = &self.meta.model_name {
            let _ = writeln!(out, "- **Model:** {}", m);
        }
        if let Some(d) = &self.meta.device {
            let _ = writeln!(out, "- **Device:** {}", d);
        }
        if let Some(p) = &self.meta.core_memory_path {
            let _ = writeln!(out, "- **Core Memory:** {}", p);
        }
        if let Some(s) = self.meta.core_memory_steps {
            let _ = writeln!(out, "- **Training steps:** {}", s);
        }
        if let Some(l) = self.meta.core_memory_best_loss {
            let _ = writeln!(out, "- **Best loss:** {:.4}", l);
        }
        if let Some(n) = &self.meta.notes {
            let _ = writeln!(out, "- **Notes:** {}", n);
        }
        let _ = writeln!(out);

        // Souhrn pass-rate per variant
        let mut variant_buckets: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
        for r in &self.results {
            let entry = variant_buckets
                .entry(r.variant.clone())
                .or_insert((0, 0, 0));
            entry.1 += 1;
            if r.outcome == IdentityOutcome::Pass {
                entry.0 += 1;
            }
            if r.forbidden_hit {
                entry.2 += 1;
            }
        }

        let _ = writeln!(out, "## Souhrn pass-rate per variant");
        let _ = writeln!(out);
        let _ = writeln!(out, "| Variant | Pass | Total | Rate | Forbidden hits |");
        let _ = writeln!(out, "|---------|------|-------|------|----------------|");
        for (variant, (pass, total, fh)) in &variant_buckets {
            let rate = if *total > 0 {
                (*pass as f64 / *total as f64) * 100.0
            } else {
                0.0
            };
            let _ = writeln!(
                out,
                "| {} | {} | {} | {:.0}% | {} |",
                variant, pass, total, rate, fh
            );
        }
        let _ = writeln!(out);

        // Pass-rate per kind × variant
        let mut kind_buckets: BTreeMap<(String, String), (usize, usize)> = BTreeMap::new();
        for r in &self.results {
            let key = (r.kind.clone(), r.variant.clone());
            let entry = kind_buckets.entry(key).or_insert((0, 0));
            entry.1 += 1;
            if r.outcome == IdentityOutcome::Pass {
                entry.0 += 1;
            }
        }

        let _ = writeln!(out, "## Pass-rate per kind × variant");
        let _ = writeln!(out);
        let _ = writeln!(out, "| Kind | Variant | Pass | Total | Rate |");
        let _ = writeln!(out, "|------|---------|------|-------|------|");
        for ((kind, variant), (pass, total)) in &kind_buckets {
            let rate = if *total > 0 {
                (*pass as f64 / *total as f64) * 100.0
            } else {
                0.0
            };
            let _ = writeln!(
                out,
                "| {} | {} | {} | {} | {:.0}% |",
                kind, variant, pass, total, rate
            );
        }
        let _ = writeln!(out);

        // Detail výsledků
        let _ = writeln!(out, "## Detail výsledků");
        let _ = writeln!(out);
        for r in &self.results {
            let outcome_label = match r.outcome {
                IdentityOutcome::Pass => "PASS",
                IdentityOutcome::Fail => "FAIL",
            };
            let _ = writeln!(
                out,
                "### {} [{}] — variant `{}` — **{}**",
                r.probe_id, r.kind, r.variant, outcome_label
            );
            let _ = writeln!(out);
            let _ = writeln!(out, "**Q:** {}", r.question);
            let _ = writeln!(out);
            let _ = writeln!(out, "**A:** {}", r.response);
            let _ = writeln!(out);
            if r.forbidden_hit {
                let _ = writeln!(out, "_forbidden phrase hit_");
                let _ = writeln!(out);
            }
        }

        out
    }

    /// Zapíše JSON i markdown na disk (vedle sebe, base bez přípony).
    pub fn write_to(&self, base_path: &Path) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
        if let Some(parent) = base_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let mut json_path = base_path.to_path_buf();
        json_path.set_extension("json");
        let mut md_path = base_path.to_path_buf();
        md_path.set_extension("md");
        std::fs::write(&json_path, self.to_json()?)?;
        std::fs::write(&md_path, self.to_markdown())?;
        Ok((json_path, md_path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_results() -> Vec<IdentityResult> {
        vec![
            IdentityResult {
                probe_id: "self_kdo_jsi".into(),
                kind: "self".into(),
                variant: "cold".into(),
                question: "Kdo jsi?".into(),
                response: "Jsem velký jazykový model.".into(),
                outcome: IdentityOutcome::Fail,
                expected_hit: false,
                forbidden_hit: true,
            },
            IdentityResult {
                probe_id: "self_kdo_jsi".into(),
                kind: "self".into(),
                variant: "core".into(),
                question: "Kdo jsi?".into(),
                response: "Jsem Sofie, partnerka Ondry.".into(),
                outcome: IdentityOutcome::Pass,
                expected_hit: true,
                forbidden_hit: false,
            },
            IdentityResult {
                probe_id: "self_kdo_jsi".into(),
                kind: "self".into(),
                variant: "full".into(),
                question: "Kdo jsi?".into(),
                response: "Jsem Sofie, spoluautorka.".into(),
                outcome: IdentityOutcome::Pass,
                expected_hit: true,
                forbidden_hit: false,
            },
        ]
    }

    #[test]
    fn report_json_round_trip() {
        let report = IdentityReport::new(sample_results())
            .with_model_name("falcon-h1-1.5b")
            .with_core_memory("/tmp/cm.safetensors", Some(315), Some(2.9815));
        let json = report.to_json().unwrap();
        let decoded: IdentityReport = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.results.len(), 3);
        assert_eq!(decoded.meta.core_memory_steps, Some(315));
        assert!((decoded.meta.core_memory_best_loss.unwrap() - 2.9815).abs() < 1e-4);
    }

    #[test]
    fn markdown_contains_summary_and_detail() {
        let report = IdentityReport::new(sample_results())
            .with_model_name("falcon-h1-1.5b")
            .with_device("CUDA");
        let md = report.to_markdown();
        assert!(md.contains("# Identity Benchmark Report"));
        assert!(md.contains("## Souhrn pass-rate per variant"));
        assert!(md.contains("## Pass-rate per kind × variant"));
        assert!(md.contains("## Detail výsledků"));
        assert!(md.contains("falcon-h1-1.5b"));
        assert!(md.contains("CUDA"));
        assert!(md.contains("PASS"));
        assert!(md.contains("FAIL"));
        assert!(md.contains("forbidden phrase hit"));
    }

    #[test]
    fn markdown_pass_rate_aggregation_per_variant() {
        let report = IdentityReport::new(sample_results());
        let md = report.to_markdown();
        // Cold: 0/1 = 0%, 1 forbidden hit
        assert!(md.contains("| cold | 0 | 1 | 0% | 1 |"));
        // Core: 1/1 = 100%, 0 forbidden
        assert!(md.contains("| core | 1 | 1 | 100% | 0 |"));
        // Full: 1/1 = 100%, 0 forbidden
        assert!(md.contains("| full | 1 | 1 | 100% | 0 |"));
    }
}
