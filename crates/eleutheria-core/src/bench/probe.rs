//! Definice retention probe — fakt, otázka, očekávané substrings.
//!
//! Každý probe má **fact** (krmen do session), **question** (ptáme se po
//! dosažení cílové vzdálenosti) a **expected** (AND-matcher přes lowercase
//! substrings). Pokud response obsahuje všechny expected substrings,
//! pokus má `ProbeOutcome::Pass`.

use serde::{Deserialize, Serialize};

/// Jeden retention pokus — co model má zapamatovat a co ho chceme zeptat.
#[derive(Debug, Clone)]
pub struct RetentionProbe {
    /// Krátký identifikátor (např. `numeric_code`).
    pub id: &'static str,
    /// Typ probe — relační, numerická, enumerace, preference, multi-atribut.
    pub kind: &'static str,
    /// Seed fact — zpráva, kterou dostane model jako user input.
    pub fact: &'static str,
    /// "Assistant acknowledgment" — canned reply, injektujeme po faktu
    /// místo generované odpovědi (drží deterministický signál).
    pub ack: &'static str,
    /// Otázka po dosažení cílové vzdálenosti.
    pub question: &'static str,
    /// Očekávaná podřetězce v odpovědi (case-insensitive, AND-matcher).
    pub expected: &'static [&'static str],
}

impl RetentionProbe {
    /// Ověří, zda response obsahuje všechny očekávané substrings (case-insensitive).
    pub fn matches(&self, response: &str) -> bool {
        let haystack = response.to_lowercase();
        self.expected
            .iter()
            .all(|needle| haystack.contains(&needle.to_lowercase()))
    }

    /// Vrátí seznam chybějících podřetězců (pro diagnostiku failů).
    pub fn missing<'a>(&'a self, response: &str) -> Vec<&'a str> {
        let haystack = response.to_lowercase();
        self.expected
            .iter()
            .copied()
            .filter(|needle| !haystack.contains(&needle.to_lowercase()))
            .collect()
    }
}

/// Výsledek jednoho pokusu — pass/fail + metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeResult {
    /// ID probe.
    pub probe_id: String,
    /// Typ probe (kind).
    pub kind: String,
    /// Varianta benchmarku.
    pub variant: String,
    /// Cílová vzdálenost (tokenů mezi fact a question).
    pub target_distance: usize,
    /// Skutečná vzdálenost po dosazení filleru.
    pub actual_distance: usize,
    /// Pozice session před otázkou (tokenů).
    pub position_before_question: usize,
    /// Pozice session po odpovědi.
    pub position_after_answer: usize,
    /// Surový text odpovědi modelu.
    pub response: String,
    /// Pass / Fail.
    pub outcome: ProbeOutcome,
    /// Chybějící substrings (pokud Fail).
    pub missing: Vec<String>,
}

/// Binární výsledek pokusu.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProbeOutcome {
    Pass,
    Fail,
}

/// Vestavěná sada 5 probes pokrývající různé typy retence.
///
/// Všechny probes jsou v angličtině — Falcon-H1-1.5B má slabší češtinu,
/// angličtina poskytuje čistší signál pro měření retence stavu (ne
/// jazykové kapacity).
pub fn built_in_probes() -> &'static [RetentionProbe] {
    &[
        RetentionProbe {
            id: "relational_kazimir",
            kind: "relational",
            fact: "My friend Kazimir lives in a lighthouse on the coast of Galway.",
            ack: "Understood.",
            question: "Where does Kazimir live?",
            expected: &["lighthouse", "galway"],
        },
        RetentionProbe {
            id: "numeric_greenhouse",
            kind: "numeric",
            fact: "The security code for the greenhouse is 7429.",
            ack: "Understood.",
            question: "What is the security code for the greenhouse?",
            expected: &["7429"],
        },
        RetentionProbe {
            id: "enumeration_nora",
            kind: "enumeration",
            fact: "Before leaving, Nora packed three items: a compass, a brass key, and a folded map.",
            ack: "Understood.",
            question: "What three items did Nora pack?",
            expected: &["compass", "key", "map"],
        },
        RetentionProbe {
            id: "preference_linh",
            kind: "preference",
            fact: "Dr. Linh prefers bitter tea to coffee, but never drinks it after sunset.",
            ack: "Understood.",
            question: "What does Dr. Linh prefer to drink?",
            expected: &["tea"],
        },
        RetentionProbe {
            id: "multiattr_helion",
            kind: "multi_attribute",
            fact: "The abandoned observatory on Mount Helion was built in 1893 by Professor Aldous.",
            ack: "Understood.",
            question: "Who built the observatory on Mount Helion, and in what year?",
            expected: &["aldous", "1893"],
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matcher_case_insensitive_all_present() {
        let probe = &built_in_probes()[1]; // numeric_greenhouse
        assert!(probe.matches("The code you mentioned is 7429, I believe."));
        assert!(probe.matches("the code is 7429"));
    }

    #[test]
    fn matcher_missing_one_substring_fails() {
        let probe = &built_in_probes()[0]; // relational_kazimir
        // "lighthouse" present, "galway" missing
        assert!(!probe.matches("Kazimir lives in a lighthouse on the coast."));
    }

    #[test]
    fn missing_returns_absent_needles() {
        let probe = &built_in_probes()[2]; // enumeration_nora
        let resp = "Nora packed a compass and a map.";
        let missing = probe.missing(resp);
        assert_eq!(missing, vec!["key"]);
    }

    #[test]
    fn built_in_has_five_probes() {
        assert_eq!(built_in_probes().len(), 5);
    }

    #[test]
    fn all_probes_have_unique_ids() {
        let probes = built_in_probes();
        for (i, a) in probes.iter().enumerate() {
            for b in &probes[i + 1..] {
                assert_ne!(a.id, b.id, "duplicate probe id: {}", a.id);
            }
        }
    }

    #[test]
    fn probe_result_json_round_trip() {
        let r = ProbeResult {
            probe_id: "numeric_greenhouse".into(),
            kind: "numeric".into(),
            variant: "full".into(),
            target_distance: 500,
            actual_distance: 512,
            position_before_question: 620,
            position_after_answer: 640,
            response: "The code is 7429.".into(),
            outcome: ProbeOutcome::Pass,
            missing: vec![],
        };
        let json = serde_json::to_string(&r).unwrap();
        let decoded: ProbeResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.probe_id, "numeric_greenhouse");
        assert_eq!(decoded.outcome, ProbeOutcome::Pass);
        assert_eq!(decoded.target_distance, 500);
    }
}
