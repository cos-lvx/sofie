/// Architecture Note:
/// Bude dotahovat relevantní memory fragments z PostgreSQL + pgvector.
/// Sémantické vyhledávání podle raw_input, injekce do system promptu nebo
/// jako samostatné context bloky.
/// Target: v0.5.0 (Persistent Memory milestone).
/// Aktuálně: passthrough (memory_fragments zůstává prázdný).

use anyhow::Result;

use crate::prompt::pipeline::PromptStage;
use crate::prompt::types::PromptContext;

pub struct MemoryInjectionStage;

impl PromptStage for MemoryInjectionStage {
    fn name(&self) -> &str {
        "MemoryInjection"
    }

    fn process(&self, ctx: &mut PromptContext) -> Result<()> {
        tracing::debug!(
            "MemoryInjection: passthrough, fragments_len={}",
            ctx.memory_fragments.len()
        );
        Ok(())
    }
}
