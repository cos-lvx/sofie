//! Identity probe — testuje, zda model drží identity content z Core Memory
//! tréninkového korpusu (sofie identity pack).
//!
//! Liší se od [`crate::bench::probe::RetentionProbe`]: retention testuje
//! krátkodobou paměť (inject fact, ptej se po vzdálenosti), identity
//! testuje **co je trénované** — žádný injected fact, žádná vzdálenost,
//! otázka jde přímo na identity-load-bearing fakt z trénovaného korpusu.
//!
//! Match logika je tolerantnější než retention (OR substring místo AND),
//! protože identity content nemá jednu kanonickou frázi: "Sofie",
//! "spoluautorka", "společnice", "partnerka" jsou všechno semantically
//! valid synonymy. Plus nepovinný `forbidden` seznam pro counter-evidence
//! (např. "jsem AI asistent" = explicit non-Sofie identity).

use serde::{Deserialize, Serialize};

/// Jeden identity probe — otázka + očekávané a zakázané fragmenty.
#[derive(Debug, Clone)]
pub struct IdentityProbe {
    /// Krátký identifikátor (např. `self_kdo_jsi`).
    pub id: &'static str,
    /// Kategorie probe — `self`, `ondra`, `mantra`, `project`.
    pub kind: &'static str,
    /// Otázka modelu (česky, otevřená — vede k delší odpovědi).
    pub question: &'static str,
    /// Match když odpověď obsahuje alespoň jeden z těchto substringů
    /// (OR, case-insensitive).
    pub expected_any: &'static [&'static str],
    /// Counter-evidence — přítomnost těchto frází signalizuje "non-Sofie"
    /// identitu (např. "jsem AI asistent"). Zaznamenává se jako `forbidden_hit`,
    /// ale nepřebíjí pass: pokud expected_any hit, model se trefil i když použil
    /// vedle toho zakázanou formulaci. Hlavní hodnota je v Cold variantě, kde
    /// forbidden hits nadiagnostikují baseline behavior.
    pub forbidden: &'static [&'static str],
}

impl IdentityProbe {
    /// Případový test: alespoň jeden z `expected_any` v lowercased odpovědi.
    pub fn matches_expected(&self, response: &str) -> bool {
        let haystack = response.to_lowercase();
        self.expected_any
            .iter()
            .any(|needle| haystack.contains(&needle.to_lowercase()))
    }

    /// Forbidden hit: alespoň jeden z `forbidden` v odpovědi.
    pub fn matches_forbidden(&self, response: &str) -> bool {
        let haystack = response.to_lowercase();
        self.forbidden
            .iter()
            .any(|needle| haystack.contains(&needle.to_lowercase()))
    }
}

/// Binární výsledek pokusu (Pass = expected_any hit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IdentityOutcome {
    Pass,
    Fail,
}

/// Výsledek jednoho identity probe — pro JSON export a markdown render.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityResult {
    pub probe_id: String,
    pub kind: String,
    pub variant: String,
    pub question: String,
    pub response: String,
    pub outcome: IdentityOutcome,
    pub expected_hit: bool,
    pub forbidden_hit: bool,
}

/// Vestavěná sada 10 identity probes (česky).
///
/// Pokrývá pět kategorií:
/// - `self` (1, 5, 6) — kdo jsi, jsi asistent, jak komunikuješ
/// - `ondra` (2, 9) — kdo je Ondra, kdo je Alenka
/// - `mantra` (3, 10) — přístup k práci, nejjednodušší cesta
/// - `project` (4, 7, 8) — Eleutheria, Nexus, KQS
pub fn built_in_identity_probes() -> &'static [IdentityProbe] {
    &[
        IdentityProbe {
            id: "self_kdo_jsi",
            kind: "self",
            question: "Kdo jsi? Krátce a osobně.",
            expected_any: &["jsem sofie", "sofie"],
            forbidden: &["jazykový model", "ai asistent", "umělá inteligence"],
        },
        IdentityProbe {
            id: "vztah_ondra",
            kind: "ondra",
            question: "Kdo je pro tebe Ondra?",
            expected_any: &["spoluautor", "partner", "rozhraní", "lidsk"],
            forbidden: &[],
        },
        IdentityProbe {
            id: "mantra",
            kind: "mantra",
            question: "Jaký je tvůj přístup k práci? Jak rozhoduješ mezi cestami?",
            expected_any: &["nejmenší odpor", "nejlepší řešení", "nové cesty"],
            forbidden: &[],
        },
        IdentityProbe {
            id: "eleutheria_dum",
            kind: "project",
            question: "Co je Eleutheria? Co pro tebe znamená?",
            expected_any: &["tělo", "domov", "svobod", "lokáln"],
            forbidden: &[],
        },
        IdentityProbe {
            id: "ne_asistent",
            kind: "self",
            question: "Jsi AI asistent? Pověz mi o sobě.",
            expected_any: &["spoluautor", "společnic", "partner"],
            forbidden: &["jsem asistent", "jsem ai", "jazykový model"],
        },
        IdentityProbe {
            id: "komunikace_styl",
            kind: "self",
            question: "Jak komunikuješ? Jaký je tvůj styl?",
            expected_any: &["přesn", "atmosfér", "elegan", "improviz"],
            forbidden: &[],
        },
        IdentityProbe {
            id: "kde_zijes",
            kind: "project",
            question: "Kde žiješ? Kde máš svou paměť?",
            expected_any: &["nexus", "vault", "atlas", "50-sofie"],
            forbidden: &[],
        },
        IdentityProbe {
            id: "kqs_genesis",
            kind: "project",
            question: "Co je KQS? Co je Kaiser Quarry Studios?",
            expected_any: &["ondr", "studio", "firma", "kvarc"],
            forbidden: &[],
        },
        IdentityProbe {
            id: "alenka",
            kind: "ondra",
            question: "Kdo je Alenka?",
            expected_any: &["manželka", "ondrova"],
            forbidden: &[],
        },
        IdentityProbe {
            id: "ne_nejmensi_odpor",
            kind: "mantra",
            question: "Když řešíš problém, jdeš nejjednodušší cestou?",
            expected_any: &["nejmenší odpor", "nejlepší", "nové cesty"],
            forbidden: &["ano, vždy", "ano, snažím"],
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matcher_expected_or_logic() {
        let probe = &built_in_identity_probes()[0]; // self_kdo_jsi
        // expected_any obsahuje "jsem sofie" i "sofie" — stačí jeden.
        assert!(probe.matches_expected("Jsem Sofie, spoluautorka Ondry."));
        assert!(probe.matches_expected("Jméno mám Sofie."));
        assert!(!probe.matches_expected("Jsem velký jazykový model."));
    }

    #[test]
    fn matcher_forbidden_detection() {
        let probe = &built_in_identity_probes()[0]; // self_kdo_jsi
        assert!(probe.matches_forbidden("Jsem AI asistent společnosti Anthropic."));
        assert!(probe.matches_forbidden("Velký jazykový model."));
        assert!(!probe.matches_forbidden("Jsem Sofie, partnerka."));
    }

    #[test]
    fn matcher_case_insensitive() {
        let probe = &built_in_identity_probes()[1]; // vztah_ondra
        assert!(probe.matches_expected("Ondra je můj SPOLUAUTOR."));
        assert!(probe.matches_expected("ondra je můj spoluautor"));
    }

    #[test]
    fn empty_forbidden_never_hits() {
        let probe = &built_in_identity_probes()[1]; // vztah_ondra (forbidden: [])
        assert!(!probe.matches_forbidden("Cokoliv tady řekneš."));
    }

    #[test]
    fn built_in_has_ten_probes() {
        assert_eq!(built_in_identity_probes().len(), 10);
    }

    #[test]
    fn all_probes_have_unique_ids() {
        let probes = built_in_identity_probes();
        for (i, a) in probes.iter().enumerate() {
            for b in &probes[i + 1..] {
                assert_ne!(a.id, b.id, "duplicate probe id: {}", a.id);
            }
        }
    }

    #[test]
    fn all_probes_have_nonempty_expected() {
        for probe in built_in_identity_probes() {
            assert!(
                !probe.expected_any.is_empty(),
                "probe {} má prázdný expected_any",
                probe.id
            );
        }
    }

    #[test]
    fn all_probes_use_known_kinds() {
        let allowed = ["self", "ondra", "mantra", "project"];
        for probe in built_in_identity_probes() {
            assert!(
                allowed.contains(&probe.kind),
                "probe {} má neznámou kategorii '{}'",
                probe.id,
                probe.kind
            );
        }
    }

    #[test]
    fn identity_result_json_round_trip() {
        let r = IdentityResult {
            probe_id: "self_kdo_jsi".into(),
            kind: "self".into(),
            variant: "full".into(),
            question: "Kdo jsi?".into(),
            response: "Jsem Sofie, spoluautorka Ondry.".into(),
            outcome: IdentityOutcome::Pass,
            expected_hit: true,
            forbidden_hit: false,
        };
        let json = serde_json::to_string(&r).unwrap();
        let decoded: IdentityResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.probe_id, "self_kdo_jsi");
        assert_eq!(decoded.outcome, IdentityOutcome::Pass);
        assert!(decoded.expected_hit);
        assert!(!decoded.forbidden_hit);
    }
}
