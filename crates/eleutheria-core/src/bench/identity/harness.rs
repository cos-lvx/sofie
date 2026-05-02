//! Orchestrátor identity benchmarku.
//!
//! [`IdentityBench::run`] iteruje přes **varianty × probes** a pro každou
//! kombinaci:
//! 1. nastaví engine (clean / persona) podle varianty
//! 2. attach/detach Core Memory podle varianty
//! 3. vytvoří čerstvou session a pošle otázku přes `send_message`
//!    s `temperature=0.0` (greedy)
//! 4. vyhodnotí odpověď proti `probe.expected_any` (OR matcher)
//! 5. zaznamená [`IdentityResult`]
//!
//! Per-variant engine setup:
//! - **Cold**: `engine_clean` + `detach_core_memory()`
//! - **Core**: `engine_clean` + `attach_core_memory(core_memory.clone())`
//! - **Full**: `engine_persona` + `attach_core_memory(core_memory.clone())`

use anyhow::Result;

use crate::bench::identity::probe::{IdentityOutcome, IdentityProbe, IdentityResult};
use crate::bench::identity::report::IdentityReport;
use crate::bench::identity::variant::IdentityVariant;
use crate::{CoreMemoryArtifact, GenerateControl, Sofie};

/// Default max tokens pro identity generation — dost na souvislou odpověď,
/// málo na bloudění.
pub const DEFAULT_ANSWER_TOKENS: usize = 200;

/// Bezstavový orchestrátor — vše běží přes parametry `run()`.
pub struct IdentityBench;

impl IdentityBench {
    /// Spustí identity benchmark.
    ///
    /// # Parametry
    /// - `engine_clean` — Sofie loaded **bez** persona (`persona_path: None`).
    ///   Používá se pro varianty `Cold` a `Core`.
    /// - `engine_persona` — Sofie loaded **s** personou (default runtime).
    ///   Používá se pro variantu `Full`.
    /// - `core_memory` — autoritativní Core Memory artefakt; každá varianta,
    ///   která ho potřebuje, dostane vlastní `clone` (Tensor je Arc-counted).
    /// - `probes` — sada otázek (typicky [`super::probe::built_in_identity_probes`])
    /// - `variants` — které varianty spustit
    /// - `on_progress` — callback po každém probe (pro CLI progress)
    pub fn run<F>(
        engine_clean: &mut Sofie,
        engine_persona: &mut Sofie,
        core_memory: &CoreMemoryArtifact,
        probes: &[IdentityProbe],
        variants: &[IdentityVariant],
        mut on_progress: F,
    ) -> Result<IdentityReport>
    where
        F: FnMut(&IdentityResult),
    {
        let mut results: Vec<IdentityResult> = Vec::new();
        let total = variants.len() * probes.len();
        tracing::info!(
            "IdentityBench start: {} variant × {} probes = {} pokusů",
            variants.len(),
            probes.len(),
            total
        );

        for variant in variants {
            // Nastav engine + Core Memory podle varianty
            match variant {
                IdentityVariant::Cold => {
                    engine_clean.detach_core_memory();
                    Self::run_variant(
                        engine_clean,
                        *variant,
                        probes,
                        &mut on_progress,
                        &mut results,
                    )?;
                }
                IdentityVariant::Core => {
                    if !engine_clean.has_core_memory() {
                        engine_clean.attach_core_memory(core_memory.clone())?;
                    }
                    Self::run_variant(
                        engine_clean,
                        *variant,
                        probes,
                        &mut on_progress,
                        &mut results,
                    )?;
                }
                IdentityVariant::Full => {
                    if !engine_persona.has_core_memory() {
                        engine_persona.attach_core_memory(core_memory.clone())?;
                    }
                    Self::run_variant(
                        engine_persona,
                        *variant,
                        probes,
                        &mut on_progress,
                        &mut results,
                    )?;
                }
            }
        }

        Ok(IdentityReport::new(results))
    }

    /// Pomocná: pustí všechny probes pro jednu variant na daném engine
    /// (předpokládá, že engine je už nastavený — Core Memory attached/detached).
    fn run_variant<F>(
        sofie: &Sofie,
        variant: IdentityVariant,
        probes: &[IdentityProbe],
        on_progress: &mut F,
        results: &mut Vec<IdentityResult>,
    ) -> Result<()>
    where
        F: FnMut(&IdentityResult),
    {
        for probe in probes {
            tracing::info!("probe={} kind={} variant={}", probe.id, probe.kind, variant);
            let result = Self::run_one(sofie, probe, variant)?;
            on_progress(&result);
            results.push(result);
        }
        Ok(())
    }

    /// Jeden probe — čerstvá session, single-turn otázka, greedy decode.
    fn run_one(
        sofie: &Sofie,
        probe: &IdentityProbe,
        variant: IdentityVariant,
    ) -> Result<IdentityResult> {
        let mut session = sofie.new_session()?;

        let mut response_buf = String::new();
        let response = sofie.send_message(
            &mut session,
            probe.question,
            DEFAULT_ANSWER_TOKENS,
            0.0, // greedy
            |_tok, text| {
                response_buf.push_str(text);
                GenerateControl::Continue
            },
        )?;
        let final_response = if response.is_empty() {
            response_buf
        } else {
            response
        };

        let expected_hit = probe.matches_expected(&final_response);
        let forbidden_hit = probe.matches_forbidden(&final_response);
        let outcome = if expected_hit {
            IdentityOutcome::Pass
        } else {
            IdentityOutcome::Fail
        };

        Ok(IdentityResult {
            probe_id: probe.id.to_string(),
            kind: probe.kind.to_string(),
            variant: variant.label().to_string(),
            question: probe.question.to_string(),
            response: final_response,
            outcome,
            expected_hit,
            forbidden_hit,
        })
    }
}
