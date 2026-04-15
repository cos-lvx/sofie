//! Deterministický filler — neutrální EN věty pro dosažení cílové pozice.
//!
//! `FillerCorpus` drží fixní sadu vět nesouvisejících s žádným fact probes.
//! `plan(target_tokens)` vrátí sekvenci `(user, assistant_ack)` párů, které
//! po injektování do session posunou pozici alespoň o `target_tokens`.
//!
//! Velikost každé věty se aproximuje empiricky (4 znaky ≈ 1 BPE token pro EN).
//! Skutečná pozice se měří ze session po každém injekci — plán je jen odhad,
//! benchmark zaznamenává `actual_distance` do výsledků.

use serde::Serialize;

/// Korpus neutrálních EN vět — deterministický, cyklicky opakovatelný.
pub struct FillerCorpus {
    sentences: &'static [&'static str],
}

impl Default for FillerCorpus {
    fn default() -> Self {
        Self {
            sentences: DEFAULT_CORPUS,
        }
    }
}

impl FillerCorpus {
    pub fn new(sentences: &'static [&'static str]) -> Self {
        assert!(
            !sentences.is_empty(),
            "FillerCorpus requires at least one sentence"
        );
        Self { sentences }
    }

    /// Vytvoří plán injekcí — sekvenci párů `(user_filler, assistant_ack)`,
    /// které po naplnění posunou pozici alespoň o `target_tokens`.
    ///
    /// Odhad je založen na jednoduchém char/token poměru (4 znaky ≈ 1 token).
    /// Skutečná pozice se bude měřit ze session po injekci.
    pub fn plan(&self, target_tokens: usize) -> FillerPlan {
        // Pár user/ack + ChatML overhead ≈ 15 tokenů režie na turn
        const CHARS_PER_TOKEN: usize = 4;
        const CHATML_OVERHEAD_TOKENS: usize = 15;

        let mut turns: Vec<FillerTurn> = Vec::new();
        let mut estimated_tokens: usize = 0;
        let mut idx: usize = 0;

        while estimated_tokens < target_tokens {
            let sentence = self.sentences[idx % self.sentences.len()];
            let sentence_tokens = sentence.len() / CHARS_PER_TOKEN;
            let ack = "I see.";
            let ack_tokens = ack.len() / CHARS_PER_TOKEN;

            turns.push(FillerTurn {
                user: sentence,
                ack,
            });
            estimated_tokens += sentence_tokens + ack_tokens + CHATML_OVERHEAD_TOKENS;
            idx += 1;

            // Safety cap — nikdy víc než 256 turnů (pro 2048 token target dostatek)
            if turns.len() > 256 {
                break;
            }
        }

        FillerPlan {
            turns,
            estimated_tokens,
        }
    }
}

/// Plán injekcí — sekvence filler turnů s odhadem celkových tokenů.
#[derive(Debug, Clone, Serialize)]
pub struct FillerPlan {
    pub turns: Vec<FillerTurn>,
    pub estimated_tokens: usize,
}

/// Jedna filler injekce — user zpráva a assistant acknowledgment.
#[derive(Debug, Clone, Serialize)]
pub struct FillerTurn {
    pub user: &'static str,
    pub ack: &'static str,
}

/// Výchozí korpus — 6 neutrálních EN vět o prázdném domě, počasí, ulici.
/// Úmyslně zvoleno tak, aby nesdílelo proper nouns ani kontexty s built-in probes.
const DEFAULT_CORPUS: &[&str] = &[
    "The morning was quiet except for the distant hum of traffic beyond the empty square.",
    "A thin layer of frost had settled on the windowsills overnight, and the garden looked pale under the low sun.",
    "Somewhere down the street a church bell marked the hour, its sound muted by the fog.",
    "The house had been empty for years, yet the floorboards still creaked in familiar places whenever the wind pressed against the outer walls.",
    "Rain began to fall gently, first as mist and then as a slow steady drumming on the eaves.",
    "A single lamp burned in the hallway, throwing long shadows across the patterned tiles and up the staircase.",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_reaches_target() {
        let corpus = FillerCorpus::default();
        let plan = corpus.plan(200);
        assert!(plan.estimated_tokens >= 200);
        assert!(!plan.turns.is_empty());
    }

    #[test]
    fn plan_is_deterministic() {
        let corpus = FillerCorpus::default();
        let a = corpus.plan(500);
        let b = corpus.plan(500);
        assert_eq!(a.turns.len(), b.turns.len());
        assert_eq!(a.estimated_tokens, b.estimated_tokens);
        for (ta, tb) in a.turns.iter().zip(b.turns.iter()) {
            assert_eq!(ta.user, tb.user);
            assert_eq!(ta.ack, tb.ack);
        }
    }

    #[test]
    fn plan_cycles_through_corpus() {
        let corpus = FillerCorpus::default();
        // Target dost velký, aby se korpus opakoval alespoň jednou
        let plan = corpus.plan(2000);
        // První a (len+1)-ní turn by měly mít stejný user text
        let n = DEFAULT_CORPUS.len();
        if plan.turns.len() > n {
            assert_eq!(plan.turns[0].user, plan.turns[n].user);
        }
    }

    #[test]
    fn plan_zero_target_empty() {
        let corpus = FillerCorpus::default();
        let plan = corpus.plan(0);
        assert!(plan.turns.is_empty());
        assert_eq!(plan.estimated_tokens, 0);
    }

    #[test]
    fn corpus_has_six_sentences() {
        assert_eq!(DEFAULT_CORPUS.len(), 6);
    }
}
