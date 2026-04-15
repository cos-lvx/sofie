//! Varianty retention benchmarku.
//!
//! - `Full` — klasická session, SSM + KV cache + conv state přežívají mezi turny
//! - `SsmOnly` — po fact+filler se stav přefiltruje na SSM-only (KV cache zruší),
//!   otázka běží s prázdným KV. Implementováno ve v0.4.2.
//! - `Cold` — žádný kontext, otázka se klade na čerstvý model. Baseline.
//!   Implementováno ve v0.4.2.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchVariant {
    /// Plný state — standardní session behavior.
    Full,
    /// SSM-only — KV cache vyčištěn před otázkou. (v0.4.2)
    SsmOnly,
    /// Cold — otázka bez kontextu, baseline. (v0.4.2)
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

    /// Je tato varianta v aktuální verzi implementována?
    pub fn is_implemented(&self) -> bool {
        matches!(self, BenchVariant::Full)
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
    fn only_full_is_implemented_in_v041() {
        assert!(BenchVariant::Full.is_implemented());
        assert!(!BenchVariant::SsmOnly.is_implemented());
        assert!(!BenchVariant::Cold.is_implemented());
    }

    #[test]
    fn label_round_trip() {
        for v in [
            BenchVariant::Full,
            BenchVariant::SsmOnly,
            BenchVariant::Cold,
        ] {
            let label = v.label();
            let parsed = BenchVariant::from_str(label).unwrap();
            assert_eq!(parsed, v);
        }
    }
}
