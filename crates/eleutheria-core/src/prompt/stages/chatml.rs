use anyhow::Result;

use crate::prompt::pipeline::PromptStage;
use crate::prompt::types::{ChatRole, PromptContext};

pub struct ChatMLAssembly;

impl PromptStage for ChatMLAssembly {
    fn name(&self) -> &str {
        "ChatMLAssembly"
    }

    fn process(&self, ctx: &mut PromptContext) -> Result<()> {
        let mut prompt = String::new();

        // System prompt
        if let Some(ref system) = ctx.system_prompt {
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str(system);
            prompt.push_str("<|im_end|>\n");
        }

        // Conversation history
        for msg in &ctx.conversation_history {
            let role = match msg.role {
                ChatRole::System => "system",
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
            };
            prompt.push_str("<|im_start|>");
            prompt.push_str(role);
            prompt.push('\n');
            prompt.push_str(&msg.content);
            prompt.push_str("<|im_end|>\n");
        }

        // Current user input
        prompt.push_str("<|im_start|>user\n");
        prompt.push_str(&ctx.raw_input);
        prompt.push_str("<|im_end|>\n");

        // Assistant turn start
        prompt.push_str("<|im_start|>assistant\n");

        ctx.assembled_prompt = Some(prompt);
        Ok(())
    }
}
