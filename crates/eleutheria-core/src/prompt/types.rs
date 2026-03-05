use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

/// Klasifikace záměru vstupu — pro budoucí InputClassifier stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputIntent {
    Question,
    Instruction,
    Continuation,
    Freeform,
}

/// Konfigurace persony — načítaná z TOML souboru.
#[derive(Debug, Clone, Deserialize)]
pub struct PersonaConfig {
    pub name: String,
    pub role: String,
    pub instructions: Vec<String>,
    pub constraints: Vec<String>,
    pub voice: String,
}

impl PersonaConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: PersonaConfig = toml::from_str(&content)?;
        Ok(config)
    }
}

/// Sdílený kontext procházející prompt pipeline.
pub struct PromptContext {
    // Vstup
    pub raw_input: String,
    pub intent: Option<InputIntent>,
    pub detected_issues: Vec<String>,

    // Persona & System
    pub persona: Option<PersonaConfig>,
    pub system_prompt: Option<String>,
    pub template_vars: HashMap<String, String>,

    // Konverzace & Paměť
    pub conversation_history: Vec<ChatMessage>,
    pub memory_fragments: Vec<String>,

    // Quality
    pub quality_score: Option<f32>,
    pub token_budget: Option<usize>,

    // Výstup
    pub assembled_prompt: Option<String>,
}

impl PromptContext {
    pub fn new(input: impl Into<String>) -> Self {
        Self {
            raw_input: input.into(),
            intent: None,
            detected_issues: Vec::new(),
            persona: None,
            system_prompt: None,
            template_vars: HashMap::new(),
            conversation_history: Vec::new(),
            memory_fragments: Vec::new(),
            quality_score: None,
            token_budget: None,
            assembled_prompt: None,
        }
    }
}
