//! Eleutheria Core — Sofie's mind engine.
//! Custom Candle inference pro Falcon-H1-7B-Instruct.

pub mod falcon_h1;

use anyhow::{anyhow, Result};
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;
use std::path::Path;

use falcon_h1::config::FalconH1Config;
use falcon_h1::model::FalconH1Model;
use falcon_h1::state::ModelState;

/// Sofie — lokální inference engine.
pub struct Sofie {
    model: FalconH1Model,
    tokenizer: Tokenizer,
    config: FalconH1Config,
    device: Device,
}

impl Sofie {
    /// Načte Falcon-H1 z lokálního adresáře.
    /// model_dir: cesta k adresáři s config.json, tokenizer.json, *.safetensors
    pub fn load(model_dir: &Path, use_cuda: bool) -> Result<Self> {
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
        tracing::info!("Config načten: {} layerů, vocab {}", config.num_hidden_layers, config.vocab_size);

        // 3. Tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Tokenizer error: {}", e))?;
        tracing::info!("Tokenizer: {} tokenů", tokenizer.get_vocab_size(true));

        // 4. Váhy — najdi všechny safetensors shardy
        let mut shard_paths: Vec<std::path::PathBuf> = std::fs::read_dir(model_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
            .collect();
        shard_paths.sort();
        tracing::info!("Načítám {} shardů", shard_paths.len());

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&shard_paths, DType::F32, &device)?
        };

        // 5. Model
        let model = FalconH1Model::load(&config, vb, &device)?;
        tracing::info!("Model načten");

        Ok(Self { model, tokenizer, config, device })
    }
}

impl Sofie {
    /// Vygeneruje text na základě promptu.
    /// Vrací kompletní výstup (prompt + generovaný text).
    pub fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
    ) -> Result<String> {
        // 1. Tokenizace
        let encoding = self.tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!("Tokenizer error: {}", e))?;
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        tracing::info!("Prompt: {} tokenů", prompt_ids.len());

        // 2. Stav modelu — prázdný, 44 layerů
        let mut state = ModelState::new(&self.config, &self.device)?;

        // 3. Prefill — prompt token po tokenu
        //    (mixer.rs je rekurentní, zpracovává seq_len=1)
        let mut logits = None;
        for (i, &token_id) in prompt_ids.iter().enumerate() {
            let input = Tensor::new(&[token_id], &self.device)?.unsqueeze(0)?; // [1, 1]
            logits = Some(self.model.forward(&input, i, &mut state)?);
        }

        let mut logits = logits.ok_or_else(|| anyhow!("Prázdný prompt"))?;

        // 4. Generování — token po tokenu
        let mut generated: Vec<u32> = Vec::new();
        let eos_token = self.tokenizer.token_to_id("</s>")
            .or_else(|| self.tokenizer.token_to_id("<|endoftext|>"))
            .unwrap_or(u32::MAX);

        for _ in 0..max_tokens {
            // logits: [1, 1, vocab_size] → [vocab_size]
            let logits_vec = logits.squeeze(0)?.squeeze(0)?;

            // Sampling
            let next_token = if temperature <= 0.0 {
                // Greedy: argmax
                logits_vec.argmax(0)?.to_scalar::<u32>()?
            } else {
                // Temperature sampling
                let scaled = (logits_vec / temperature)?;
                let probs = candle_nn::ops::softmax_last_dim(&scaled.unsqueeze(0)?)?
                    .squeeze(0)?;
                sample_from_probs(&probs)?
            };

            if next_token == eos_token {
                tracing::info!("EOS po {} tokenech", generated.len());
                break;
            }

            generated.push(next_token);

            // Další krok
            let pos = prompt_ids.len() + generated.len() - 1;
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            logits = self.model.forward(&input, pos, &mut state)?;
        }

        // 5. Dekódování
        let output = self.tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow!("Decode error: {}", e))?;

        Ok(output)
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