//! Varianty retention benchmarku.
//!
//! - `Full` — klasická session, SSM + KV cache + conv state přežívají mezi turny
//! - `SsmOnly` — po fact+filler se stav přefiltruje (`StateFilter::ssm_only()`):
//!   KV cache + conv state se zahodí, pozice se resetuje na 0, otázka běží
//!   přes plnou pipeline jako turn 1. Měří, kolik si SSM samostatně zachová.
//! - `Cold` — žádný kontext, otázka se klade na čerstvou session. Baseline.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchVariant {
    /// Plný state — standardní session behavior (SSM + KV + conv).
    Full,
    /// SSM-only — KV cache + conv state vyčištěny před otázkou.
    SsmOnly,
    /// Cold — otázka bez kontextu, baseline bez paměťového signálu.
    Cold,
}

impl BenchVariant {
    /// Krátký label pro reporty a logy.
    pub fn label(&self) -> &'static str {
        match self {
            BenchVariant::Full => "full",
            BenchVariant::SsmOnly => "ssm_only",
            BenchVariant::Cold => "cold",
        }
    }

    /// Všechny tři varianty v jednom poli — užitečné pro `--variant all`.
    pub fn all() -> &'static [BenchVariant] {
        &[
            BenchVariant::Full,
            BenchVariant::SsmOnly,
            BenchVariant::Cold,
        ]
    }
}

impl fmt::Display for BenchVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

impl FromStr for BenchVariant {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "full" => Ok(Self::Full),
            "ssm_only" | "ssm-only" | "ssm" => Ok(Self::SsmOnly),
            "cold" => Ok(Self::Cold),
            other => Err(format!(
                "neznámá varianta '{}' (očekávám: full | ssm_only | cold)",
                other
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_str_accepts_aliases() {
        assert_eq!(BenchVariant::from_str("full").unwrap(), BenchVariant::Full);
        assert_eq!(
            BenchVariant::from_str("ssm_only").unwrap(),
            BenchVariant::SsmOnly
        );
        assert_eq!(
            BenchVariant::from_str("ssm-only").unwrap(),
            BenchVariant::SsmOnly
        );
        assert_eq!(BenchVariant::from_str("cold").unwrap(), BenchVariant::Cold);
    }

    #[test]
    fn from_str_rejects_unknown() {
        assert!(BenchVariant::from_str("bogus").is_err());
    }

    #[test]
    fn all_returns_three_variants() {
        let all = BenchVariant::all();
        assert_eq!(all.len(), 3);
        assert!(all.contains(&BenchVariant::Full));
        assert!(all.contains(&BenchVariant::SsmOnly));
        assert!(all.contains(&BenchVariant::Cold));
    }

    #[test]
    fn label_round_trip() {
        for v in BenchVariant::all() {
            let label = v.label();
            let parsed = BenchVariant::from_str(label).unwrap();
            assert_eq!(parsed, *v);
        }
    }
}
