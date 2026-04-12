use anyhow::Result;

use super::types::PromptContext;

pub trait PromptStage {
    fn name(&self) -> &str;
    fn process(&self, ctx: &mut PromptContext) -> Result<()>;
}

pub struct PromptPipeline {
    stages: Vec<Box<dyn PromptStage>>,
}

impl Default for PromptPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptPipeline {
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    pub fn add_stage(&mut self, stage: Box<dyn PromptStage>) {
        self.stages.push(stage);
    }

    pub fn run(&self, ctx: &mut PromptContext) -> Result<()> {
        for stage in &self.stages {
            tracing::debug!("Pipeline stage: {}", stage.name());
            stage.process(ctx)?;
        }
        Ok(())
    }
}
