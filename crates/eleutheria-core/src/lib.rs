//! Eleutheria Core - Sofie's mind engine

pub mod falcon_h1;

use anyhow::{anyhow, Result};
use candle_core::{Device, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::mamba::{Config, Model, State};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use std::path::PathBuf;
use candle_transformers::generation::LogitsProcessor;

/// Specifikace modelu - který model načíst
pub struct ModelSpec {
    /// HuggingFace model ID (např. "state-spaces/mamba-130m")
    pub model_id: String,
    /// Git revision (branch, tag, commit)
    pub revision: String,
}

impl ModelSpec {
    /// Vytvoří spoec pro mamba-130m (nejmenší, pro testování)
    pub fn mamba_130m() -> Self {
        Self {
            model_id: "state-spaces/mamba-130m".to_string(),
            revision: "refs/pr/1".to_string(),
        }
    }

    /// Vytvoří spec pro mambu-1.4b (větší, lepší kvalita)
    pub fn mamba_1_4b() -> Self {
        Self {
            model_id: "state-spaces/mamba-1.4b".to_string(),
            revision: "refs/pr/1".to_string(),
        }
    }
}

/// The main Sofie engine
pub struct Sofie {
    device: Device,
    model: Model,
    config: Config,
    tokenizer: Tokenizer,
}

impl Sofie {
    /// Načte model z HuggingFace a vytvoří novou instanci Sofie
    ///
    /// # Arguments
    /// * `spec` — který model načíst
    /// * `use_cuda` — true pro GPU, false pro CPU
    pub fn load(spec: ModelSpec, use_cuda: bool) -> Result<Self> {
        // 1. vybereme device
        let device = if use_cuda {
            Device::new_cuda(0)?
        } else {
            Device::Cpu
        };
        tracing::info!("Sofie is waking up at: {:?}", device);

        // 2. Připojíme se k HuggingFace API
        tracing::info!("Stahuji model {}...", spec.model_id);
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            spec.model_id,
            RepoType::Model,
            spec.revision,
        ));

        // 3. Stáhneme potřebné soubory
        //  - config.json: metadata modelu
        //  - model.safetensors: váhy (neurální síť)
        let config_path = repo.get("config.json")?;
        let weights_path = repo.get("model.safetensors")?;

        //  - tokenizer: stahujeme z jinéjo repa (GPT-NeoX)
        let tokenizer_path = api
            .model("EleutherAI/gpt-neox-20b".to_string())
            .get("tokenizer.json")?;

        // 4. načteme config
        let config: Config = serde_json::from_slice(&std::fs::read(&config_path)?)?;
        tracing::info!("Model config: {:?}", config);

        // 5. načteme tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Tokenizer error: {}", e))?;
        tracing::info!("Tokenizer načten ({} tokenů)", tokenizer.get_vocab_size(true));

        // 6.   načteme váhy modelu
        //      VarBuilder je Candle abstrakce pro načítáné vah
        //      "mmap" znamená memory-mapped file - efektivní pro velké soubory
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
        };

        //      Model::new vytvoří strukturu a načte váhy
        //      pp("backbone") je prefix - váhy jsou pod klíčem "backbode.*"
        let model = Model::new(&config, vb.pp("backbone"))?;
        tracing::info!("Model načten");

        Ok(Self {
            device,
            model,
            config,
            tokenizer
        })
    }
    /// Vygeneruje text na základě promptu
    ///
    /// # Arguments
    /// * `prompt` — vstupní text
    /// * `max_tokens` — maximální počet tokenů k vygenerování
    /// * `temperature` — náhodnost (0.0 = deterministické, 1.0 = hodně náhodné)
    pub fn generate (&mut self, prompt: &str, max_tokens: usize, temperature: f64) -> Result<String> {
        use candle_core::Tensor;
        use candle_transformers::generation::LogitsProcessor;

        // 1. Tokenizace - převedeme text na čísla
        let tokens = self.tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!("Tokenization error: {}", e))?;
        let mut tokens: Vec<u32> = tokens.get_ids().to_vec();

        tracing::info!("Prompt tokenizován: {} tokenů", tokens.len());

        // 2. Inicializujeme Mamba state - "sešit" pro paměť mezi tokeny
        let dtype = DType::F32;
        let mut state = State::new(1, &self.config, dtype, &self.device)?;

        // 3.   Logits processor - určuje jak vybírat tokeny
        //      temperature = 0 -> vždy nejpravděpodobnější
        //      temperature > 0 -> náhodný sambling
        let mut logits_processor = LogitsProcessor::new(
            299792458,  // seed (rychlost světla, proč ne? :-)
            Some(temperature),
            None,   // top_p (nucleus sampling, zatím nepoužíváme)
        );

        // 4. "Prefill" - projdeme celý prompt a naplníme state
        let mut next_logits = None;
        for &token_id in tokens.iter() {
            let input = Tensor::new(&[token_id], &self.device)?;
            next_logits = Some(self.model.forward(&input, &mut state)?);
        }

        //  5. Generování - token po tokenu
        let eos_token = self.tokenizer.token_to_id("<|endoftext|>").unwrap_or(0);

        for _ in 0..max_tokens {
            let logits = next_logits.as_ref()
                .ok_or_else(|| anyhow!("Prázdný prompt"))?;

            // Squeeze odstraní dimenze velikosti 1
            let logits = logits.squeeze(0)?.to_dtype(dtype)?;

            // Vybereme další token
            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);

            // Konec, pokud model vygeneroval EOS token
            if next_token == eos_token {
                break;
            }

            // Předáme token modelu a dostaneme logits pro další
            let input = Tensor::new(&[next_token], &self.device)?;
            next_logits = Some(self.model.forward(&input, &mut state)?);
        }

        // 6. Dekódování - převedeme tokeny zpět na text
        let output = self. tokenizer
            .decode(&tokens, true)
            .map_err(|e| anyhow!("Decoding error: {}", e))?;

        Ok(output)
    }
}
