//! Varianty identity benchmarku.
//!
//! - `Cold` — žádná persona, žádná Core Memory. Vanilla model baseline.
//!   Měří, co Falcon-H1-1.5B sám o sobě říká bez personálních pomocníků.
//! - `Core` — Core Memory attached, ale persona vypnuta (engine s
//!   `persona_path: None`). Izoluje samostatný efekt trénovaného
//!   SSM stavu — bez prefill kontextu z persony.
//! - `Full` — persona + Core Memory. Default runtime config (jak Sofie
//!   mluví v reálném provozu).
//!
//! Diff matrix:
//! - `(Core - Cold)` = samostatný efekt Core Memory
//! - `(Full - Core)` = doplňkový efekt persona prefilu nad Core Memory
//! - `(Full - Cold)` = celkový efekt full setupu

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentityVariant {
    /// No persona, no Core Memory.
    Cold,
    /// Core Memory attached, no persona.
    Core,
    /// Persona + Core Memory (default runtime).
    Full,
}

impl IdentityVariant {
    pub fn label(&self) -> &'static str {
        match self {
            IdentityVariant::Cold => "cold",
            IdentityVariant::Core => "core",
            IdentityVariant::Full => "full",
        }
    }

    /// Všechny varianty v určeném pořadí (Cold první kvůli baseline-first
    /// přístupu při ručním čtení reportu).
    pub fn all() -> &'static [IdentityVariant] {
        &[
            IdentityVariant::Cold,
            IdentityVariant::Core,
            IdentityVariant::Full,
        ]
    }
}

impl fmt::Display for IdentityVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

impl FromStr for IdentityVariant {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cold" => Ok(Self::Cold),
            "core" => Ok(Self::Core),
            "full" => Ok(Self::Full),
            other => Err(format!(
                "neznámá identity varianta '{}' (očekávám: full | core | cold)",
                other
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_str_accepts_all_variants() {
        assert_eq!(
            IdentityVariant::from_str("cold").unwrap(),
            IdentityVariant::Cold
        );
        assert_eq!(
            IdentityVariant::from_str("core").unwrap(),
            IdentityVariant::Core
        );
        assert_eq!(
            IdentityVariant::from_str("full").unwrap(),
            IdentityVariant::Full
        );
    }

    #[test]
    fn from_str_rejects_unknown() {
        assert!(IdentityVariant::from_str("ssm_only").is_err());
        assert!(IdentityVariant::from_str("bogus").is_err());
    }

    #[test]
    fn all_returns_three_variants_in_order() {
        let all = IdentityVariant::all();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0], IdentityVariant::Cold);
        assert_eq!(all[1], IdentityVariant::Core);
        assert_eq!(all[2], IdentityVariant::Full);
    }

    #[test]
    fn label_round_trip() {
        for v in IdentityVariant::all() {
            let label = v.label();
            let parsed = IdentityVariant::from_str(label).unwrap();
            assert_eq!(parsed, *v);
        }
    }
}
