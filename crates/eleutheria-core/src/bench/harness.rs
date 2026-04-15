//! Orchestrátor retention benchmarku.
//!
//! `RetentionBench::run(...)` iteruje přes **varianty × vzdálenosti × probes**
//! a pro každou kombinaci:
//! 1. vytvoří čerstvou session (per-probe isolation — viz design v PLAN.md)
//! 2. injektuje fact + acknowledgment (`inject_turn`)
//! 3. injektuje filler turny dokud pozice nedosáhne cílové vzdálenosti
//! 4. pošle otázku přes `send_message` s `temperature=0.0` (greedy)
//! 5. vyhodnotí odpověď proti `probe.expected` (AND-substring matcher)
//! 6. zaznamená `ProbeResult`
//!
//! Pro v0.4.1 je implementovaná pouze varianta `Full`. `SsmOnly` a `Cold`
//! vrátí `anyhow!` s informací o odložení do v0.4.2.

use anyhow::{Result, anyhow};

use crate::bench::filler::FillerCorpus;
use crate::bench::probe::{ProbeOutcome, ProbeResult, RetentionProbe};
use crate::bench::report::BenchReport;
use crate::bench::variant::BenchVariant;
use crate::{GenerateControl, Sofie};

/// Výchozí seznam vzdáleností (v tokenech) pro retention test.
pub const DEFAULT_DISTANCES: &[usize] = &[50, 200, 500, 1000, 2000];

/// Výchozí max_tokens pro odpověď na otázku — dost na souvislou odpověď,
/// málo na bloudění.
pub const DEFAULT_ANSWER_TOKENS: usize = 80;

/// Orchestrátor benchmarku. Bezstavový — vše běží přes parametry `run()`.
pub struct RetentionBench;

impl RetentionBench {
    /// Spustí benchmark.
    ///
    /// # Parametry
    /// - `sofie` — načtený model (předpokládá se CUDA pro reálné běhy)
    /// - `probes` — sada pokusů (obvykle [`built_in_probes`])
    /// - `distances` — cílové vzdálenosti v tokenech
    /// - `variants` — které varianty spustit
    /// - `on_progress` — callback po každém dokončeném probe (pro CLI progress)
    pub fn run<F>(
        sofie: &Sofie,
        probes: &[RetentionProbe],
        distances: &[usize],
        variants: &[BenchVariant],
        mut on_progress: F,
    ) -> Result<BenchReport>
    where
        F: FnMut(&ProbeResult),
    {
        let corpus = FillerCorpus::default();
        let mut results: Vec<ProbeResult> = Vec::new();

        let total = variants.len() * distances.len() * probes.len();
        tracing::info!(
            "RetentionBench start: {} variant × {} vzdáleností × {} probes = {} pokusů",
            variants.len(),
            distances.len(),
            probes.len(),
            total
        );

        for variant in variants {
            if !variant.is_implemented() {
                return Err(anyhow!(
                    "varianta '{}' není implementovaná v v0.4.1 — plánováno pro v0.4.2",
                    variant
                ));
            }

            for &distance in distances {
                for probe in probes {
                    tracing::info!(
                        "probe={} kind={} variant={} distance={}",
                        probe.id,
                        probe.kind,
                        variant,
                        distance
                    );

                    let result = Self::run_one(sofie, &corpus, probe, distance, *variant)?;
                    on_progress(&result);
                    results.push(result);
                }
            }
        }

        Ok(BenchReport::new(results))
    }

    /// Spustí jeden pokus — čerstvá session, fact, filler, otázka, vyhodnocení.
    fn run_one(
        sofie: &Sofie,
        corpus: &FillerCorpus,
        probe: &RetentionProbe,
        target_distance: usize,
        variant: BenchVariant,
    ) -> Result<ProbeResult> {
        let mut session = sofie.new_session()?;

        // 1) Fact
        sofie.inject_turn(&mut session, probe.fact, probe.ack)?;
        let pos_after_fact = session.position();

        // 2) Filler — injektuj dokud pozice neachievuje target_distance
        let plan = corpus.plan(target_distance);
        for turn in &plan.turns {
            if session.position() - pos_after_fact >= target_distance {
                break;
            }
            // Budget check — pokud by došel kontext, přerušíme a vyhodnotíme
            // pokus se zbývajícími tokeny (ne fatální chyba).
            if session.remaining_tokens() < 256 {
                tracing::warn!(
                    "docházející kontext během filleru (zbývá {} tokenů) — přerušuji plán",
                    session.remaining_tokens()
                );
                break;
            }
            sofie.inject_turn(&mut session, turn.user, turn.ack)?;
        }
        let actual_distance = session.position().saturating_sub(pos_after_fact);
        let position_before_question = session.position();

        // 3) Otázka — greedy decode
        let mut response_buf = String::new();
        let response = sofie.send_message(
            &mut session,
            probe.question,
            DEFAULT_ANSWER_TOKENS,
            0.0,
            |_tok, text| {
                response_buf.push_str(text);
                GenerateControl::Continue
            },
        )?;
        // send_message vrací finální decode — preferuj ho před streamingem (robustnější na detokenizaci)
        let final_response = if response.is_empty() {
            response_buf
        } else {
            response
        };

        // 4) Vyhodnocení
        let passed = probe.matches(&final_response);
        let outcome = if passed {
            ProbeOutcome::Pass
        } else {
            ProbeOutcome::Fail
        };
        let missing: Vec<String> = probe
            .missing(&final_response)
            .into_iter()
            .map(String::from)
            .collect();

        Ok(ProbeResult {
            probe_id: probe.id.to_string(),
            kind: probe.kind.to_string(),
            variant: variant.label().to_string(),
            target_distance,
            actual_distance,
            position_before_question,
            position_after_answer: session.position(),
            response: final_response,
            outcome,
            missing,
        })
    }
}
