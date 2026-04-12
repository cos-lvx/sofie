/// Architecture Note:
/// Validuje finální prompt před inference:
///   - Celková délka vs token_budget (kontextové okno modelu)
///   - Poměr system/context/user obsahu
///   - Detekce potenciálních problémů (příliš dlouhý system, chybějící persona)
///
/// Target: v0.3.0+ (základní validace), v0.6.0 (model-assisted quality check)
///
/// Aktuálně: passthrough, loguje délku assembled_prompt pokud existuje.
use anyhow::Result;

use crate::prompt::pipeline::PromptStage;
use crate::prompt::types::PromptContext;

pub struct QualityGateStage;

impl PromptStage for QualityGateStage {
    fn name(&self) -> &str {
        "QualityGate"
    }

    fn process(&self, ctx: &mut PromptContext) -> Result<()> {
        if let Some(ref prompt) = ctx.assembled_prompt {
            tracing::debug!("QualityGate: assembled_prompt len={}", prompt.len());
        } else {
            tracing::debug!("QualityGate: no assembled_prompt yet");
        }
        Ok(())
    }
}
