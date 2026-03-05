/// Architecture Note:
/// Bude klasifikovat raw_input na InputIntent (Question/Instruction/Continuation/Freeform).
/// Bude detekovat issues (chybějící kontext, nejasná formulace).
/// Target: v0.4.0+ (potenciálně s pomocí modelu samotného pro self-reflection).
/// Aktuálně: passthrough, nastaví intent na Freeform.

use anyhow::Result;

use crate::prompt::pipeline::PromptStage;
use crate::prompt::types::{InputIntent, PromptContext};

pub struct InputClassifier;

impl PromptStage for InputClassifier {
    fn name(&self) -> &str {
        "InputClassifier"
    }

    fn process(&self, ctx: &mut PromptContext) -> Result<()> {
        ctx.intent = Some(InputIntent::Freeform);
        tracing::debug!("InputClassifier: passthrough, intent=Freeform");
        Ok(())
    }
}
