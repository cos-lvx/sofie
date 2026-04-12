use anyhow::Result;
use chrono::Datelike;

use crate::prompt::pipeline::PromptStage;
use crate::prompt::types::PromptContext;

pub struct TemplateExpansion;

fn czech_weekday(weekday: chrono::Weekday) -> &'static str {
    match weekday {
        chrono::Weekday::Mon => "pondělí",
        chrono::Weekday::Tue => "úterý",
        chrono::Weekday::Wed => "středa",
        chrono::Weekday::Thu => "čtvrtek",
        chrono::Weekday::Fri => "pátek",
        chrono::Weekday::Sat => "sobota",
        chrono::Weekday::Sun => "neděle",
    }
}

impl PromptStage for TemplateExpansion {
    fn name(&self) -> &str {
        "TemplateExpansion"
    }

    fn process(&self, ctx: &mut PromptContext) -> Result<()> {
        let system = match &ctx.system_prompt {
            Some(s) => s.clone(),
            None => return Ok(()),
        };

        // Built-in vars
        let now = chrono::Local::now();
        let mut vars = std::collections::HashMap::new();

        if let Some(ref persona) = ctx.persona {
            vars.insert("name".to_string(), persona.name.clone());
        }
        vars.insert("date".to_string(), now.format("%Y-%m-%d").to_string());
        vars.insert("time".to_string(), now.format("%H:%M").to_string());
        vars.insert(
            "weekday".to_string(),
            czech_weekday(now.weekday()).to_string(),
        );

        // Custom vars mají přednost
        for (k, v) in &ctx.template_vars {
            vars.insert(k.clone(), v.clone());
        }

        // Regex-free substituce: iteruj přes {{key}} páry
        let mut result = String::with_capacity(system.len());
        let mut rest = system.as_str();

        while let Some(start) = rest.find("{{") {
            result.push_str(&rest[..start]);
            let after_open = &rest[start + 2..];
            if let Some(end) = after_open.find("}}") {
                let key = &after_open[..end];
                if let Some(value) = vars.get(key) {
                    result.push_str(value);
                } else {
                    tracing::warn!("Template var '{{{{{}}}}}' not found", key);
                    result.push_str("{{");
                    result.push_str(key);
                    result.push_str("}}");
                }
                rest = &after_open[end + 2..];
            } else {
                // Neuzavřený {{ — zachovat as-is
                result.push_str("{{");
                rest = after_open;
            }
        }
        result.push_str(rest);

        ctx.system_prompt = Some(result);
        Ok(())
    }
}
