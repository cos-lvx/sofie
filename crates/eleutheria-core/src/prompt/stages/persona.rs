use anyhow::Result;

use crate::prompt::pipeline::PromptStage;
use crate::prompt::types::PromptContext;

pub struct PersonaInjection;

impl PromptStage for PersonaInjection {
    fn name(&self) -> &str {
        "PersonaInjection"
    }

    fn process(&self, ctx: &mut PromptContext) -> Result<()> {
        let persona = match &ctx.persona {
            Some(p) => p,
            None => return Ok(()),
        };

        let mut text = format!("Jsi {}. {}\n", persona.name, persona.role);

        if !persona.instructions.is_empty() {
            text.push_str("\nInstrukce:\n");
            for instr in &persona.instructions {
                text.push_str("- ");
                text.push_str(instr);
                text.push('\n');
            }
        }

        if !persona.constraints.is_empty() {
            text.push_str("\nOmezení:\n");
            for c in &persona.constraints {
                text.push_str("- ");
                text.push_str(c);
                text.push('\n');
            }
        }

        text.push_str(&format!("\nStyl: {}", persona.voice));

        // Pokud system_prompt už existuje, prepend persona
        if let Some(ref existing) = ctx.system_prompt {
            text.push_str("\n---\n");
            text.push_str(existing);
        }

        ctx.system_prompt = Some(text);
        Ok(())
    }
}
