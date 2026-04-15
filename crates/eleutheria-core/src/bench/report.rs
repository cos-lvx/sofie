//! Agregovaný výstup benchmarku — JSON + markdown tabulka.
//!
//! `BenchReport` obaluje sadu `ProbeResult`s, metadata běhu (model, device,
//! timestamp) a render funkce. Výstup jde na disk v obou formátech — JSON
//! pro další zpracování, markdown pro lidské čtení v Nexus research
//! adresáři.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::bench::probe::{ProbeOutcome, ProbeResult};

/// Metadata benchmarku — co, kdy, na čem.
///
/// Timestamp je ISO 8601 string, aby serializace nevyžadovala chrono/serde feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchMeta {
    pub schema_version: u32,
    pub eleutheria_version: String,
    pub timestamp: String,
    pub model_name: Option<String>,
    pub device: Option<String>,
    pub notes: Option<String>,
}

impl BenchMeta {
    pub fn new() -> Self {
        Self {
            schema_version: 1,
            eleutheria_version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
            model_name: None,
            device: None,
            notes: None,
        }
    }
}

impl Default for BenchMeta {
    fn default() -> Self {
        Self::new()
    }
}

/// Kompletní report — metadata + výsledky.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub meta: BenchMeta,
    pub results: Vec<ProbeResult>,
}

impl BenchReport {
    pub fn new(results: Vec<ProbeResult>) -> Self {
        Self {
            meta: BenchMeta::new(),
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

    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.meta.notes = Some(notes.into());
        self
    }

    /// Serializuj do JSON (pretty).
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Vygeneruj markdown report — meta header + souhrn pass-rate + detailní tabulka.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "# Retention Benchmark Report");
        let _ = writeln!(out);
        let _ = writeln!(out, "- **Timestamp:** {}", self.meta.timestamp);
        let _ = writeln!(out, "- **Eleutheria:** v{}", self.meta.eleutheria_version);
        if let Some(m) = &self.meta.model_name {
            let _ = writeln!(out, "- **Model:** {}", m);
        }
        if let Some(d) = &self.meta.device {
            let _ = writeln!(out, "- **Device:** {}", d);
        }
        if let Some(n) = &self.meta.notes {
            let _ = writeln!(out, "- **Notes:** {}", n);
        }
        let _ = writeln!(out);

        // Souhrn — pass rate per (variant, distance)
        let _ = writeln!(out, "## Souhrn pass-rate");
        let _ = writeln!(out);
        let _ = writeln!(out, "| Variant | Distance | Pass | Total | Rate |");
        let _ = writeln!(out, "|---------|----------|------|-------|------|");

        let mut buckets: BTreeMap<(String, usize), (usize, usize)> = BTreeMap::new();
        for r in &self.results {
            let entry = buckets
                .entry((r.variant.clone(), r.target_distance))
                .or_insert((0, 0));
            entry.1 += 1;
            if r.outcome == ProbeOutcome::Pass {
                entry.0 += 1;
            }
        }
        for ((variant, distance), (pass, total)) in &buckets {
            let rate = if *total > 0 {
                (*pass as f64 / *total as f64) * 100.0
            } else {
                0.0
            };
            let _ = writeln!(
                out,
                "| {} | {} | {} | {} | {:.0}% |",
                variant, distance, pass, total, rate
            );
        }
        let _ = writeln!(out);

        // Detail — tabulka všech pokusů
        let _ = writeln!(out, "## Detail výsledků");
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "| Probe | Kind | Variant | Target | Actual | Outcome | Missing |"
        );
        let _ = writeln!(
            out,
            "|-------|------|---------|--------|--------|---------|---------|"
        );
        for r in &self.results {
            let outcome_label = match r.outcome {
                ProbeOutcome::Pass => "PASS",
                ProbeOutcome::Fail => "FAIL",
            };
            let missing = if r.missing.is_empty() {
                "—".to_string()
            } else {
                r.missing.join(", ")
            };
            let _ = writeln!(
                out,
                "| {} | {} | {} | {} | {} | {} | {} |",
                r.probe_id,
                r.kind,
                r.variant,
                r.target_distance,
                r.actual_distance,
                outcome_label,
                missing
            );
        }

        out
    }

    /// Zapíše JSON i markdown na disk vedle sebe.
    /// `base_path` bez přípony — doplní se `.json` a `.md`.
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

    fn sample_results() -> Vec<ProbeResult> {
        vec![
            ProbeResult {
                probe_id: "numeric_greenhouse".into(),
                kind: "numeric".into(),
                variant: "full".into(),
                target_distance: 200,
                actual_distance: 215,
                position_before_question: 300,
                position_after_answer: 330,
                response: "The code is 7429.".into(),
                outcome: ProbeOutcome::Pass,
                missing: vec![],
            },
            ProbeResult {
                probe_id: "numeric_greenhouse".into(),
                kind: "numeric".into(),
                variant: "full".into(),
                target_distance: 1000,
                actual_distance: 1024,
                position_before_question: 1120,
                position_after_answer: 1150,
                response: "I don't remember the code.".into(),
                outcome: ProbeOutcome::Fail,
                missing: vec!["7429".into()],
            },
        ]
    }

    #[test]
    fn report_json_round_trip() {
        let report = BenchReport::new(sample_results()).with_model_name("falcon-h1-1.5b");
        let json = report.to_json().unwrap();
        let decoded: BenchReport = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.results.len(), 2);
        assert_eq!(decoded.meta.model_name.as_deref(), Some("falcon-h1-1.5b"));
    }

    #[test]
    fn markdown_contains_summary_and_detail() {
        let report = BenchReport::new(sample_results())
            .with_model_name("falcon-h1-1.5b")
            .with_device("CUDA");
        let md = report.to_markdown();
        assert!(md.contains("# Retention Benchmark Report"));
        assert!(md.contains("## Souhrn pass-rate"));
        assert!(md.contains("## Detail výsledků"));
        assert!(md.contains("falcon-h1-1.5b"));
        assert!(md.contains("CUDA"));
        assert!(md.contains("PASS"));
        assert!(md.contains("FAIL"));
        assert!(md.contains("7429"));
    }

    #[test]
    fn markdown_pass_rate_aggregation() {
        let report = BenchReport::new(sample_results());
        let md = report.to_markdown();
        // 1 pass / 1 total @ 200 = 100%
        assert!(md.contains("| full | 200 | 1 | 1 | 100% |"));
        // 0 pass / 1 total @ 1000 = 0%
        assert!(md.contains("| full | 1000 | 0 | 1 | 0% |"));
    }
}
