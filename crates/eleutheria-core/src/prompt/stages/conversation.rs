/// Architecture Note:
/// Bude injektovat conversation_history z krátkodobé paměti.
/// Strategie: sliding window nebo summarization (TBD v 0.4.0).
/// Aktuálně: passthrough (conversation_history zůstává prázdná).

use anyhow::Result;

use crate::prompt::pipeline::PromptStage;
use crate::prompt::types::PromptContext;

pub struct ConversationContextStage;

impl PromptStage for ConversationContextStage {
    fn name(&self) -> &str {
        "ConversationContext"
    }

    fn process(&self, ctx: &mut PromptContext) -> Result<()> {
        tracing::debug!(
            "ConversationContext: passthrough, history_len={}",
            ctx.conversation_history.len()
        );
        Ok(())
    }
}
