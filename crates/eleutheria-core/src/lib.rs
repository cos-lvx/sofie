//! Eleutheria Core — Sofie's mind engine.
//! Custom Candle inference pro Falcon-H1-7B-Instruct.

pub mod falcon_h1;
pub mod prompt;

use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;
use tokenizers::Tokenizer;

pub use falcon_h1::checkpoint::{StateCheckpoint, StateFilter};
use falcon_h1::config::FalconH1Config;
use falcon_h1::model::FalconH1Model;
use falcon_h1::state::ModelState;

use prompt::pipeline::PromptPipeline;
use prompt::types::{PersonaConfig, PromptContext};

/// Řízení streaming generace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerateControl {
    Continue,
    Stop,
}

/// Sofie — lokální inference engine.
pub struct Sofie {
    model: FalconH1Model,
    tokenizer: Tokenizer,
    config: FalconH1Config,
    device: Device,
    dtype: DType,
    pipeline: PromptPipeline,
    persona: Option<PersonaConfig>,
}

impl Sofie {
    /// Načte Falcon-H1 z lokálního adresáře.
    /// model_dir: cesta k adresáři s config.json, tokenizer.json, *.safetensors
    /// persona_path: optional cesta k TOML persona souboru
    pub fn load(model_dir: &Path, use_cuda: bool, persona_path: Option<&Path>) -> Result<Self> {
        // 1. Device
        let device = if use_cuda {
            Device::new_cuda(0)?
        } else {
            Device::Cpu
        };
        tracing::info!("Device: {:?}", device);

        // 2. Config
        let config_path = model_dir.join("config.json");
        let config: FalconH1Config = serde_json::from_slice(&std::fs::read(&config_path)?)?;
        tracing::info!(
            "Config načten: {} layerů, vocab {}",
            config.num_hidden_layers,
            config.vocab_size
        );

        // 3. Tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("Tokenizer error: {}", e))?;
        tracing::info!("Tokenizer: {} tokenů", tokenizer.get_vocab_size(true));

        // 4. Váhy — najdi všechny safetensors shardy
        let mut shard_paths: Vec<std::path::PathBuf> = std::fs::read_dir(model_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        shard_paths.sort();
        tracing::info!("Načítám {} shardů", shard_paths.len());

        // GPU: BF16 (nativní formát vah, CUDA to podporuje)
        // CPU: F32 (candle CPU neumí BF16 matmul)
        let dtype = if use_cuda { DType::BF16 } else { DType::F32 };
        tracing::info!("DType: {:?}", dtype);

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&shard_paths, dtype, &device)? };

        // 5. Model
        let model = FalconH1Model::load(&config, vb, &device)?;
        tracing::info!("Model načten");

        // 6. Persona
        let persona = match persona_path {
            Some(path) => {
                let p = PersonaConfig::from_file(path)?;
                tracing::info!("Persona načtena: {}", p.name);
                Some(p)
            }
            None => None,
        };

        // 7. Pipeline
        use prompt::stages::chatml::ChatMLAssembly;
        use prompt::stages::classifier::InputClassifier;
        use prompt::stages::conversation::ConversationContextStage;
        use prompt::stages::memory::MemoryInjectionStage;
        use prompt::stages::persona::PersonaInjection;
        use prompt::stages::quality::QualityGateStage;
        use prompt::stages::template::TemplateExpansion;

        let mut pipeline = PromptPipeline::new();
        pipeline.add_stage(Box::new(InputClassifier));
        pipeline.add_stage(Box::new(PersonaInjection));
        pipeline.add_stage(Box::new(TemplateExpansion));
        pipeline.add_stage(Box::new(ConversationContextStage));
        pipeline.add_stage(Box::new(MemoryInjectionStage));
        pipeline.add_stage(Box::new(QualityGateStage));
        pipeline.add_stage(Box::new(ChatMLAssembly));
        tracing::info!("Pipeline: 7 stages");

        Ok(Self {
            model,
            tokenizer,
            config,
            device,
            dtype,
            pipeline,
            persona,
        })
    }
}

/// Výsledek generování — text + stav pro případný checkpoint.
pub struct GenerateResult {
    pub text: String,
    pub state: ModelState,
    pub position: usize,
}

impl Sofie {
    /// High-level chat API — projde prompt pipeline, pak generuje.
    pub fn chat_streaming(
        &self,
        user_message: &str,
        max_tokens: usize,
        temperature: f64,
        on_token: impl FnMut(u32, &str) -> GenerateControl,
    ) -> Result<GenerateResult> {
        // 1. PromptContext
        let mut ctx = PromptContext::new(user_message);
        ctx.persona = self.persona.clone();

        // 2. Pipeline
        self.pipeline.run(&mut ctx)?;

        // 3. Assembled prompt
        let prompt = ctx
            .assembled_prompt
            .ok_or_else(|| anyhow!("Pipeline nevyprodukovala assembled_prompt"))?;

        tracing::info!(
            "Assembled prompt ({} chars):\n{}",
            prompt.len(),
            &prompt[..prompt.len().min(200)]
        );

        // 4. Generuj přes low-level API
        self.generate_streaming(&prompt, max_tokens, temperature, None, on_token)
    }

    /// High-level chat bez streamingu.
    pub fn chat(
        &self,
        user_message: &str,
        max_tokens: usize,
        temperature: f64,
    ) -> Result<GenerateResult> {
        self.chat_streaming(user_message, max_tokens, temperature, |_, _| {
            GenerateControl::Continue
        })
    }

    /// Generuje text se streaming callbackem.
    /// `on_token` dostává (token_id, nový_text) a vrací GenerateControl.
    /// `initial_state` — volitelný pre-loaded state (z checkpointu).
    ///   Pokud None, vytvoří se čerstvý stav (nuly).
    ///   Tuple: (ModelState, pozice) — pozice = počet tokenů už zpracovaných ve stavu.
    pub fn generate_streaming(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        initial_state: Option<(ModelState, usize)>,
        mut on_token: impl FnMut(u32, &str) -> GenerateControl,
    ) -> Result<GenerateResult> {
        // 1. Tokenizace
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!("Tokenizer error: {}", e))?;
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        tracing::info!("Prompt: {} tokenů", prompt_ids.len());

        // 2. Stav modelu — buď z checkpointu, nebo čerstvý
        let (mut state, base_pos) = match initial_state {
            Some((s, p)) => {
                tracing::info!("Načten state z checkpointu (pozice {})", p);
                (s, p)
            }
            None => {
                let s = ModelState::new(&self.config, self.dtype, &self.device)?;
                (s, 0)
            }
        };

        // 3. Parallel prefill — celý prompt jedním průchodem
        let prompt_len = prompt_ids.len();
        tracing::info!(
            "Parallel prefill: {} tokenů (base_pos={})",
            prompt_len,
            base_pos
        );

        let prompt_tensor = Tensor::new(prompt_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let prefill_logits = self.model.forward(&prompt_tensor, base_pos, &mut state)?;

        let mut logits = prefill_logits.narrow(1, prompt_len - 1, 1)?;

        // 4. Generování se streamingem
        let mut generated: Vec<u32> = Vec::new();
        let mut emitted_len: usize = 0;
        let eos_primary = self.config.eos_token_id.unwrap_or(11);
        let eos_im_end: u32 = 228;
        tracing::info!(
            "EOS tokens: {} (primary), {} (im_end)",
            eos_primary,
            eos_im_end
        );

        for _ in 0..max_tokens {
            let logits_vec = logits.squeeze(0)?.squeeze(0)?;

            // Sampling v F32 — BF16 nepodporuje skalární aritmetiku v Candle
            let logits_f32 = logits_vec.to_dtype(DType::F32)?;

            let next_token = if temperature <= 0.0 {
                logits_f32.argmax(0)?.to_scalar::<u32>()?
            } else {
                let scaled = (logits_f32 / temperature)?;
                let probs = candle_nn::ops::softmax_last_dim(&scaled.unsqueeze(0)?)?.squeeze(0)?;
                sample_from_probs(&probs)?
            };

            if next_token == eos_primary || next_token == eos_im_end {
                tracing::info!(
                    "EOS po {} tokenech (token id: {})",
                    generated.len(),
                    next_token
                );
                break;
            }

            generated.push(next_token);

            // Diff-based dekódování — korektní pro BPE artefakty i multi-byte UTF-8
            let full_text = self
                .tokenizer
                .decode(&generated, true)
                .map_err(|e| anyhow!("Decode error: {}", e))?;
            let new_text = &full_text[emitted_len..];
            if !new_text.is_empty() {
                if on_token(next_token, new_text) == GenerateControl::Stop {
                    break;
                }
                emitted_len = full_text.len();
            }

            // Další krok
            let pos = base_pos + prompt_len + generated.len() - 1;
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            logits = self.model.forward(&input, pos, &mut state)?;
        }

        // 5. Finální dekódování celého výstupu
        let output = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow!("Decode error: {}", e))?;

        let final_pos = base_pos + prompt_len + generated.len();

        Ok(GenerateResult {
            text: output,
            state,
            position: final_pos,
        })
    }

    /// Vygeneruje text bez streamingu (wrapper přes generate_streaming).
    pub fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
    ) -> Result<GenerateResult> {
        self.generate_streaming(prompt, max_tokens, temperature, None, |_, _| {
            GenerateControl::Continue
        })
    }

    // -------------------------------------------------------------------
    // State checkpoint API
    // -------------------------------------------------------------------

    /// Uloží stav modelu na disk.
    pub fn save_state(
        &self,
        state: &ModelState,
        pos: usize,
        path: &Path,
        filter: StateFilter,
    ) -> Result<()> {
        let checkpoint = StateCheckpoint::from_model_state(state, pos, &self.config, filter)?;
        checkpoint.save(path)?;
        tracing::info!(
            "State uložen: {} (filtr: {})",
            path.display(),
            filter.label()
        );
        Ok(())
    }

    /// Načte stav modelu z disku a přesune na správný device/dtype.
    pub fn load_state(&self, path: &Path) -> Result<(ModelState, usize)> {
        let checkpoint = StateCheckpoint::load(path)?;
        checkpoint.validate_config(&self.config)?;
        let result = checkpoint.into_model_state(&self.config, &self.device, self.dtype)?;
        tracing::info!("State načten: {} (pozice: {})", path.display(), result.1);
        Ok(result)
    }

    /// Vrací konfiguraci modelu — pro inspekci a validaci.
    pub fn config(&self) -> &FalconH1Config {
        &self.config
    }
}

/// Vzorkování z pravděpodobnostní distribuce.
fn sample_from_probs(probs: &Tensor) -> Result<u32> {
    let probs_vec: Vec<f32> = probs.to_vec1()?;
    let mut rng_val: f64 = rand_simple();
    for (i, &p) in probs_vec.iter().enumerate() {
        rng_val -= p as f64;
        if rng_val <= 0.0 {
            return Ok(i as u32);
        }
    }
    // Fallback: poslední token
    Ok((probs_vec.len() - 1) as u32)
}

/// Jednoduchý pseudo-RNG (xorshift64). Pro PoC stačí.
fn rand_simple() -> f64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(299_792_458);
    let mut s = STATE.load(Ordering::Relaxed);
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    STATE.store(s, Ordering::Relaxed);
    (s as f64) / (u64::MAX as f64)
}
