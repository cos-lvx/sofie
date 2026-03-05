//! Eleutheria Core — Sofie's mind engine.
//! Custom Candle inference pro Falcon-H1-7B-Instruct.

pub mod falcon_h1;
pub mod prompt;

use anyhow::{anyhow, Result};
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;
use std::path::Path;

use falcon_h1::config::FalconH1Config;
use falcon_h1::model::FalconH1Model;
use falcon_h1::state::ModelState;

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

        // GPU: BF16 (nativní formát vah, CUDA to podporuje)
        // CPU: F32 (candle CPU neumí BF16 matmul)
        let dtype = if use_cuda { DType::BF16 } else { DType::F32 };
        tracing::info!("DType: {:?}", dtype);

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&shard_paths, dtype, &device)?
        };

        // 5. Model
        let model = FalconH1Model::load(&config, vb, &device)?;
        tracing::info!("Model načten");

        Ok(Self { model, tokenizer, config, device, dtype })
    }
}

impl Sofie {
    /// Generuje text se streaming callbackem.
    /// `on_token` dostává (token_id, nový_text) a vrací GenerateControl.
    /// Dekódování přes diff celého bufferu — korektní i pro multi-byte UTF-8 a BPE artefakty.
    pub fn generate_streaming(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        mut on_token: impl FnMut(u32, &str) -> GenerateControl,
    ) -> Result<String> {
        // 1. Tokenizace
        let encoding = self.tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!("Tokenizer error: {}", e))?;
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        tracing::info!("Prompt: {} tokenů", prompt_ids.len());

        // 2. Stav modelu
        let mut state = ModelState::new(&self.config, self.dtype, &self.device)?;

        // 3. Parallel prefill — celý prompt jedním průchodem
        let prompt_len = prompt_ids.len();
        tracing::info!("Parallel prefill: {} tokenů", prompt_len);

        let prompt_tensor = Tensor::new(prompt_ids.as_slice(), &self.device)?
            .unsqueeze(0)?;
        let prefill_logits = self.model.forward(&prompt_tensor, 0, &mut state)?;

        let mut logits = prefill_logits.narrow(1, prompt_len - 1, 1)?;

        // 4. Generování se streamingem
        let mut generated: Vec<u32> = Vec::new();
        let mut emitted_len: usize = 0;
        let eos_primary = self.config.eos_token_id.unwrap_or(11);
        let eos_im_end: u32 = 228;
        tracing::info!("EOS tokens: {} (primary), {} (im_end)", eos_primary, eos_im_end);

        for _ in 0..max_tokens {
            let logits_vec = logits.squeeze(0)?.squeeze(0)?;

            // Sampling v F32 — BF16 nepodporuje skalární aritmetiku v Candle
            let logits_f32 = logits_vec.to_dtype(DType::F32)?;

            let next_token = if temperature <= 0.0 {
                logits_f32.argmax(0)?.to_scalar::<u32>()?
            } else {
                let scaled = (logits_f32 / temperature)?;
                let probs = candle_nn::ops::softmax_last_dim(&scaled.unsqueeze(0)?)?
                    .squeeze(0)?;
                sample_from_probs(&probs)?
            };

            if next_token == eos_primary || next_token == eos_im_end {
                tracing::info!("EOS po {} tokenech (token id: {})", generated.len(), next_token);
                break;
            }

            generated.push(next_token);

            // Diff-based dekódování — korektní pro BPE artefakty i multi-byte UTF-8
            let full_text = self.tokenizer
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
            let pos = prompt_len + generated.len() - 1;
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            logits = self.model.forward(&input, pos, &mut state)?;
        }

        // 5. Finální dekódování celého výstupu
        let output = self.tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow!("Decode error: {}", e))?;

        Ok(output)
    }

    /// Vygeneruje text bez streamingu (wrapper přes generate_streaming).
    pub fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
    ) -> Result<String> {
        self.generate_streaming(prompt, max_tokens, temperature, |_, _| GenerateControl::Continue)
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