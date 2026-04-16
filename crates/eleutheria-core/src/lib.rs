//! Eleutheria Core — Sofie's mind engine.
//! Custom Candle inference pro Falcon-H1-7B-Instruct.

pub mod bench;
pub mod falcon_h1;
pub mod prompt;
pub mod session;
pub mod training;

use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;
use tokenizers::Tokenizer;

pub use falcon_h1::checkpoint::{StateCheckpoint, StateFilter};
use falcon_h1::config::FalconH1Config;
use falcon_h1::model::FalconH1Model;
use falcon_h1::state::ModelState;
pub use session::SofieSession;

use prompt::pipeline::PromptPipeline;
use prompt::types::{PersonaConfig, PromptContext};

/// Řízení streaming generace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerateControl {
    Continue,
    Stop,
}

/// Výsledek generování — text + stav pro případný checkpoint.
pub struct GenerateResult {
    pub text: String,
    pub state: ModelState,
    pub position: usize,
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
    pub fn load(model_dir: &Path, use_cuda: bool, persona_path: Option<&Path>) -> Result<Self> {
        let device = if use_cuda {
            Device::new_cuda(0)?
        } else {
            Device::Cpu
        };
        tracing::info!("Device: {:?}", device);

        let config_path = model_dir.join("config.json");
        let config: FalconH1Config = serde_json::from_slice(&std::fs::read(&config_path)?)?;
        tracing::info!(
            "Config načten: {} layerů, vocab {}",
            config.num_hidden_layers,
            config.vocab_size
        );

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("Tokenizer error: {}", e))?;
        tracing::info!("Tokenizer: {} tokenů", tokenizer.get_vocab_size(true));

        let mut shard_paths: Vec<std::path::PathBuf> = std::fs::read_dir(model_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        shard_paths.sort();
        tracing::info!("Načítám {} shardů", shard_paths.len());

        let dtype = if use_cuda { DType::BF16 } else { DType::F32 };
        tracing::info!("DType: {:?}", dtype);

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&shard_paths, dtype, &device)? };

        let model = FalconH1Model::load(&config, vb, &device)?;
        tracing::info!("Model načten");

        let persona = match persona_path {
            Some(path) => {
                let p = PersonaConfig::from_file(path)?;
                tracing::info!("Persona načtena: {}", p.name);
                Some(p)
            }
            None => None,
        };

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

// ---------------------------------------------------------------------------
// Session API — multi-turn konverzace s inkrementálním prefillem
// ---------------------------------------------------------------------------

impl Sofie {
    /// Vytvoří novou session (prázdný stav).
    pub fn new_session(&self) -> Result<SofieSession> {
        let state = ModelState::new(&self.config, self.dtype, &self.device)?;
        Ok(SofieSession::new(state, &self.config))
    }

    /// Vytvoří session z checkpointu.
    pub fn resume_session(&self, path: &Path) -> Result<SofieSession> {
        let (state, pos) = self.load_state(path)?;
        Ok(SofieSession::from_checkpoint(state, pos, &self.config))
    }

    /// Injektuje řízený turn do session bez decoding — nakrmí model danými
    /// tokeny a posune stav, ale negeneruje vlastní odpověď. Slouží pro
    /// benchmarky a replay, kde chceme deterministické, reproducibilní
    /// chování (např. SSM state retention).
    ///
    /// Zachovává stejný invariant jako `send_message`: po dokončení je
    /// pozice session právě za posledním tokenem asistentovy (forced)
    /// odpovědi, uzavírací `<|im_end|>` ještě nebyl konzumován — příští
    /// delta ho připojí jako prefix.
    pub fn inject_turn(
        &self,
        session: &mut SofieSession,
        user_msg: &str,
        assistant_msg: &str,
    ) -> Result<()> {
        let (tokens, base_pos) = if !session.initialized {
            // Turn 1: plný pipeline, pak připoj forced assistant reply.
            let mut ctx = PromptContext::new(user_msg);
            ctx.persona = self.persona.clone();
            self.pipeline.run(&mut ctx)?;
            let base = ctx
                .assembled_prompt
                .ok_or_else(|| anyhow!("Pipeline nevyprodukovala assembled_prompt"))?;
            // `base` končí řetězcem "<|im_start|>assistant\n" — dopiš odpověď
            // bez trailing <|im_end|>, aby stav odpovídal invariantu send_message.
            let full = format!("{}{}", base, assistant_msg);

            tracing::info!(
                "inject_turn 1 — full pipeline + forced reply ({} chars)",
                full.len()
            );

            let encoding = self
                .tokenizer
                .encode(full.as_str(), true)
                .map_err(|e| anyhow!("Tokenizer error: {}", e))?;
            session.initialized = true;
            (encoding.get_ids().to_vec(), 0usize)
        } else {
            // Turn 2+: delta s uzavřením předchozího turnu + forced reply.
            let delta = format!(
                "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}",
                user_msg, assistant_msg
            );
            tracing::info!(
                "inject_turn {} — delta ({} chars)",
                session.turn_count() + 1,
                delta.len()
            );
            let encoding = self
                .tokenizer
                .encode(delta.as_str(), false)
                .map_err(|e| anyhow!("Tokenizer error: {}", e))?;
            (encoding.get_ids().to_vec(), session.position)
        };

        let prompt_len = tokens.len();
        let remaining = session.remaining_tokens();
        if remaining <= prompt_len {
            return Err(anyhow!(
                "kontext vyčerpán: zbývá {} tokenů, inject potřebuje {}",
                remaining,
                prompt_len
            ));
        }

        // Prefill — update state bez decoding.
        let prompt_tensor = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let _ = self
            .model
            .forward(&prompt_tensor, base_pos, &mut session.state)?;

        session.record_turn(user_msg, assistant_msg, base_pos + prompt_len);
        Ok(())
    }

    /// Pošle zprávu v rámci session — inkrementální prefill.
    ///
    /// Turn 1: plný pipeline (system + persona + user + assistant_start).
    /// Turn 2+: delta (im_end + user turn + assistant_start) — jen nové tokeny.
    pub fn send_message(
        &self,
        session: &mut SofieSession,
        message: &str,
        max_tokens: usize,
        temperature: f64,
        on_token: impl FnMut(u32, &str) -> GenerateControl,
    ) -> Result<String> {
        let (tokens, base_pos) = if !session.initialized {
            // Turn 1: plný pipeline
            let mut ctx = PromptContext::new(message);
            ctx.persona = self.persona.clone();
            self.pipeline.run(&mut ctx)?;
            let prompt = ctx
                .assembled_prompt
                .ok_or_else(|| anyhow!("Pipeline nevyprodukovala assembled_prompt"))?;

            tracing::info!(
                "Turn 1 — full pipeline ({} chars):\n{}",
                prompt.len(),
                &prompt[..prompt.len().min(200)]
            );

            let encoding = self
                .tokenizer
                .encode(prompt.as_str(), true)
                .map_err(|e| anyhow!("Tokenizer error: {}", e))?;
            session.initialized = true;
            (encoding.get_ids().to_vec(), 0usize)
        } else {
            // Turn 2+: delta — jen nové tokeny
            let delta = format!(
                "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                message
            );

            tracing::info!(
                "Turn {} — delta ({} chars)",
                session.turn_count() + 1,
                delta.len()
            );

            // false = bez BOS tokenu (jsme uprostřed sekvence)
            let encoding = self
                .tokenizer
                .encode(delta.as_str(), false)
                .map_err(|e| anyhow!("Tokenizer error: {}", e))?;
            (encoding.get_ids().to_vec(), session.position)
        };

        let prompt_len = tokens.len();
        tracing::info!("Prefill: {} tokenů (base_pos={})", prompt_len, base_pos);

        // Budget enforcement
        let remaining = session.remaining_tokens();
        if remaining <= prompt_len {
            return Err(anyhow!(
                "kontext vyčerpán: zbývá {} tokenů, potřebuji minimálně {} pro prefill",
                remaining,
                prompt_len
            ));
        }

        let budget_for_generation = remaining - prompt_len;
        let effective_max_tokens = if max_tokens > budget_for_generation {
            tracing::warn!(
                "max_tokens ({}) přesahuje zbývající kontext ({}), omezuji na {}",
                max_tokens,
                budget_for_generation,
                budget_for_generation
            );
            budget_for_generation
        } else {
            max_tokens
        };

        // Prefill
        let prompt_tensor = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let prefill_logits = self
            .model
            .forward(&prompt_tensor, base_pos, &mut session.state)?;
        let initial_logits = prefill_logits.narrow(1, prompt_len - 1, 1)?;

        // Generate
        let (response, generated_count) = self.generate_from_logits(
            initial_logits,
            &mut session.state,
            base_pos + prompt_len,
            effective_max_tokens,
            temperature,
            on_token,
        )?;

        // Aktualizuj session
        let new_position = base_pos + prompt_len + generated_count;
        session.record_turn(message, &response, new_position);

        // Varování při vysokém využití kontextu
        let usage = session.context_usage();
        if usage > 0.75 {
            tracing::warn!(
                "kontext {:.1}% využit ({} / {} tokenů)",
                usage * 100.0,
                session.position(),
                session.context_limit()
            );
        }

        Ok(response)
    }
}

// ---------------------------------------------------------------------------
// Privátní — generate loop (sdílený mezi send_message a generate_streaming)
// ---------------------------------------------------------------------------

impl Sofie {
    /// Generuj tokeny ze startovních logitů.
    /// Vrací (text, počet_generovaných_tokenů).
    fn generate_from_logits(
        &self,
        initial_logits: Tensor,
        state: &mut ModelState,
        start_pos: usize,
        max_tokens: usize,
        temperature: f64,
        mut on_token: impl FnMut(u32, &str) -> GenerateControl,
    ) -> Result<(String, usize)> {
        let mut logits = initial_logits;
        let mut generated: Vec<u32> = Vec::new();
        let mut emitted_len: usize = 0;
        let eos_primary = self.config.eos_token_id.unwrap_or(11);
        let eos_im_end: u32 = 228;

        for _ in 0..max_tokens {
            let logits_vec = logits.squeeze(0)?.squeeze(0)?;
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

            let full_text = self
                .tokenizer
                .decode(&generated, true)
                .map_err(|e| anyhow!("Decode error: {}", e))?;

            // Odolnost na BPE retokenizaci: nový `full_text` nemusí být
            // byte-prefix extension minulého (tokenizer může re-dekódovat
            // zpětně při novém tokenu), a `emitted_len` nemusí ležet na
            // UTF-8 char boundary — naivní `&full_text[emitted_len..]` pak
            // panikuje (BUG-009). Resync na nejbližší nižší char boundary.
            let safe_start =
                if emitted_len <= full_text.len() && full_text.is_char_boundary(emitted_len) {
                    emitted_len
                } else {
                    let mut s = emitted_len.min(full_text.len());
                    while s > 0 && !full_text.is_char_boundary(s) {
                        s -= 1;
                    }
                    s
                };
            let new_text = &full_text[safe_start..];
            if !new_text.is_empty() {
                if on_token(next_token, new_text) == GenerateControl::Stop {
                    break;
                }
                emitted_len = full_text.len();
            }

            let pos = start_pos + generated.len() - 1;
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            logits = self.model.forward(&input, pos, state)?;
        }

        let output = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow!("Decode error: {}", e))?;

        Ok((output, generated.len()))
    }
}

// ---------------------------------------------------------------------------
// Single-shot API (zpětná kompatibilita)
// ---------------------------------------------------------------------------

impl Sofie {
    /// High-level chat API — projde prompt pipeline, pak generuje.
    pub fn chat_streaming(
        &self,
        user_message: &str,
        max_tokens: usize,
        temperature: f64,
        on_token: impl FnMut(u32, &str) -> GenerateControl,
    ) -> Result<GenerateResult> {
        let mut ctx = PromptContext::new(user_message);
        ctx.persona = self.persona.clone();
        self.pipeline.run(&mut ctx)?;
        let prompt = ctx
            .assembled_prompt
            .ok_or_else(|| anyhow!("Pipeline nevyprodukovala assembled_prompt"))?;

        tracing::info!(
            "Assembled prompt ({} chars):\n{}",
            prompt.len(),
            &prompt[..prompt.len().min(200)]
        );

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

    /// Generuje text se streaming callbackem (single-shot).
    pub fn generate_streaming(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        initial_state: Option<(ModelState, usize)>,
        on_token: impl FnMut(u32, &str) -> GenerateControl,
    ) -> Result<GenerateResult> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!("Tokenizer error: {}", e))?;
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        tracing::info!("Prompt: {} tokenů", prompt_ids.len());

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

        let prompt_len = prompt_ids.len();
        tracing::info!(
            "Parallel prefill: {} tokenů (base_pos={})",
            prompt_len,
            base_pos
        );

        let prompt_tensor = Tensor::new(prompt_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let prefill_logits = self.model.forward(&prompt_tensor, base_pos, &mut state)?;
        let initial_logits = prefill_logits.narrow(1, prompt_len - 1, 1)?;

        let (output, generated_count) = self.generate_from_logits(
            initial_logits,
            &mut state,
            base_pos + prompt_len,
            max_tokens,
            temperature,
            on_token,
        )?;

        let final_pos = base_pos + prompt_len + generated_count;

        Ok(GenerateResult {
            text: output,
            state,
            position: final_pos,
        })
    }

    /// Vygeneruje text bez streamingu.
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

    /// Načte stav modelu z disku.
    pub fn load_state(&self, path: &Path) -> Result<(ModelState, usize)> {
        let checkpoint = StateCheckpoint::load(path)?;
        checkpoint.validate_config(&self.config)?;
        let result = checkpoint.into_model_state(&self.config, &self.device, self.dtype)?;
        tracing::info!("State načten: {} (pozice: {})", path.display(), result.1);
        Ok(result)
    }

    /// Vrací konfiguraci modelu.
    pub fn config(&self) -> &FalconH1Config {
        &self.config
    }

    // -------------------------------------------------------------------
    // Accessory pro training modul (pub(crate) by stačilo, ale training
    // je pub mod — veřejné jsou tedy kvůli internímu použití z training/smoke.rs).
    // -------------------------------------------------------------------

    /// Referencí na compute device (CPU/CUDA).
    pub fn device_ref(&self) -> &Device {
        &self.device
    }

    /// Runtime dtype (BF16 na CUDA, F32 na CPU).
    pub fn dtype_ref(&self) -> DType {
        self.dtype
    }

    /// Vytvoří čerstvý `ModelState` s nulovými komponenty — wrapper pro
    /// `ModelState::new`, aby training modul nemusel re-exportovat typ.
    pub fn new_model_state(&self) -> Result<ModelState> {
        Ok(ModelState::new(&self.config, self.dtype, &self.device)?)
    }

    /// Přímý forward pass přes Falcon-H1 model. Používá se v training modulu
    /// pro manuální kontrolu stavu (inject trainable Var před voláním).
    ///
    /// Vrací logits `[batch, seq_len, vocab_size]`.
    pub fn model_forward(
        &self,
        input_ids: &Tensor,
        base_pos: usize,
        state: &mut ModelState,
    ) -> Result<Tensor> {
        Ok(self.model.forward(input_ids, base_pos, state)?)
    }

    /// Vyfiltruje session na SSM-only stav — zachová SSM, zahodí KV cache
    /// a conv state. Resetuje pozici na 0 a označí session za neinicializovanou,
    /// takže příští `send_message` projde plnou pipeline jako turn 1.
    ///
    /// **Sémantika:** měří, kolik si Mamba-2 SSM state samostatně zachová
    /// informaci po N tokenech, když attention historie (KV cache) zmizí.
    /// Conv state je krátkodobé okno — pro retention testy přes stovky+
    /// tokenů irelevantní, proto zahozený. Nutné, protože RoPE indexy v KV
    /// musí být konzistentní — nelze nechat KV prázdnou s position > 0.
    ///
    /// **Použití:** retention benchmark varianta `SsmOnly` (v0.4.2+).
    pub fn filter_session_to_ssm_only(&self, session: &mut SofieSession) -> Result<()> {
        let checkpoint = StateCheckpoint::from_model_state(
            &session.state,
            session.position(),
            &self.config,
            StateFilter::ssm_only(),
        )?;
        let (new_state, _pos) =
            checkpoint.into_model_state(&self.config, &self.device, self.dtype)?;
        session.replace_state(new_state, 0, true);
        tracing::info!(
            "session vyfiltrována na ssm_only: KV cache + conv state vyhozeny, pozice resetována na 0"
        );
        Ok(())
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
