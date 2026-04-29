//! Save/Load trénovaného Core Memory artefaktu.
//!
//! `CoreMemoryArtifact` je serializační formát pro **trained initial SSM
//! states** — výstup `train_core_memory` po N optimizer stepech. Liší se
//! od `falcon_h1::checkpoint::StateCheckpoint` v sémantice:
//!
//! - `StateCheckpoint` (filter `core_memory`) ukládá runtime SSM **i**
//!   conv state v dtype modelu (BF16 na CUDA), s pozicí v sekvenci. Slouží
//!   pro session resume.
//! - `CoreMemoryArtifact` ukládá **pouze trénovaný initial SSM state** ve
//!   F32 (native dtype `Var`), bez conv state, bez pozice. Slouží jako
//!   "Sofie identity plugin" — co se Sofie naučila o sobě.
//!
//! Formát: safetensors s tensory `init_state.{i:02}` + `__metadata__`
//! hlavičkou (`kind=core_memory_trained`, eleutheria_version, training
//! statistiky).
//!
//! ## Životní cyklus
//!
//! ```text
//! Training:        save:                    Inference:
//!   CoreMemoryStack ──▶ from_stack ─▶ save ─▶ disk
//!                                              │
//!                                              ▼
//!                                       Sofie::attach_core_memory
//!                                              │
//!                                              ▼
//!                                       new_session() apply_to_state
//! ```
//!
//! Pro **resume tréninku** (v0.5.0-alpha.15+) `into_stack` zkonstruuje
//! čerstvý `CoreMemoryStack` s `Var`-y inicializovanými z uložených
//! tensorů — gradient může pokračovat ze stejného bodu.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Result, Tensor, Var};
use safetensors::tensor as st;

use crate::falcon_h1::config::FalconH1Config;
use crate::falcon_h1::state::ModelState;
use crate::training::core_memory::{CoreMemory, CoreMemoryStack};

/// Identifikuje formát souboru — odlišení od `StateCheckpoint`.
const ARTIFACT_KIND: &str = "core_memory_trained";
/// Verze formátu pro budoucí migrace.
const FORMAT_VERSION: u32 = 1;
/// Prefix tensor names per layer.
const TENSOR_PREFIX: &str = "init_state";

// ---------------------------------------------------------------------------
// CoreMemoryMeta
// ---------------------------------------------------------------------------

/// Metadata trénovaného Core Memory artefaktu.
///
/// Drží jak strukturní informace pro validaci kompatibility (rozměry SSM
/// stavu, počet vrstev), tak telemetrii tréninku (kolik kroků, jaký best
/// loss) — ta poslední je informativní, neslouží k validaci.
#[derive(Debug, Clone)]
pub struct CoreMemoryMeta {
    pub eleutheria_version: String,
    pub format_version: u32,
    pub kind: String,
    pub num_layers: usize,
    pub n_heads: usize,
    pub headdim: usize,
    pub d_state: usize,
    pub dtype: String,
    pub timestamp: String,
    /// Telemetrie z `TrainingResult` — informativní.
    pub training_steps: Option<usize>,
    pub best_loss: Option<f64>,
    pub final_loss: Option<f64>,
    /// Volitelná lidská poznámka (např. "law_pack 8 epoch, 1.5B").
    pub notes: Option<String>,
}

impl CoreMemoryMeta {
    fn new(
        config: &FalconH1Config,
        training_steps: Option<usize>,
        best_loss: Option<f64>,
        final_loss: Option<f64>,
        notes: Option<String>,
    ) -> Self {
        Self {
            eleutheria_version: env!("CARGO_PKG_VERSION").to_string(),
            format_version: FORMAT_VERSION,
            kind: ARTIFACT_KIND.into(),
            num_layers: config.num_hidden_layers,
            n_heads: config.mamba_n_heads,
            headdim: config.mamba_d_head,
            d_state: config.mamba_d_state,
            dtype: format!("{:?}", DType::F32),
            timestamp: chrono::Utc::now().to_rfc3339(),
            training_steps,
            best_loss,
            final_loss,
            notes,
        }
    }

    /// Serializace do safetensors metadata mapy (vše String).
    fn to_metadata_map(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("eleutheria_version".into(), self.eleutheria_version.clone());
        m.insert("format_version".into(), self.format_version.to_string());
        m.insert("kind".into(), self.kind.clone());
        m.insert("num_layers".into(), self.num_layers.to_string());
        m.insert("n_heads".into(), self.n_heads.to_string());
        m.insert("headdim".into(), self.headdim.to_string());
        m.insert("d_state".into(), self.d_state.to_string());
        m.insert("dtype".into(), self.dtype.clone());
        m.insert("timestamp".into(), self.timestamp.clone());
        if let Some(s) = self.training_steps {
            m.insert("training_steps".into(), s.to_string());
        }
        if let Some(l) = self.best_loss {
            m.insert("best_loss".into(), format!("{:.6}", l));
        }
        if let Some(l) = self.final_loss {
            m.insert("final_loss".into(), format!("{:.6}", l));
        }
        if let Some(n) = &self.notes {
            m.insert("notes".into(), n.clone());
        }
        m
    }

    /// Deserializace z safetensors metadata mapy.
    fn from_metadata_map(m: &HashMap<String, String>) -> Result<Self> {
        let get = |key: &str| -> Result<&str> {
            m.get(key).map(|s| s.as_str()).ok_or_else(|| {
                candle_core::Error::Msg(format!("core memory metadata chybí klíč: {key}"))
            })
        };
        let parse_usize = |key: &str| -> Result<usize> {
            get(key)?
                .parse::<usize>()
                .map_err(|e| candle_core::Error::Msg(format!("{key}: {e}")))
        };

        let kind = get("kind")?.to_string();
        if kind != ARTIFACT_KIND {
            return Err(candle_core::Error::Msg(format!(
                "očekávám kind={ARTIFACT_KIND}, soubor má kind={kind}",
            )));
        }

        Ok(Self {
            eleutheria_version: get("eleutheria_version")?.to_string(),
            format_version: get("format_version")?
                .parse::<u32>()
                .map_err(|e| candle_core::Error::Msg(format!("format_version: {e}")))?,
            kind,
            num_layers: parse_usize("num_layers")?,
            n_heads: parse_usize("n_heads")?,
            headdim: parse_usize("headdim")?,
            d_state: parse_usize("d_state")?,
            dtype: get("dtype")?.to_string(),
            timestamp: get("timestamp")?.to_string(),
            training_steps: m.get("training_steps").and_then(|s| s.parse().ok()),
            best_loss: m.get("best_loss").and_then(|s| s.parse().ok()),
            final_loss: m.get("final_loss").and_then(|s| s.parse().ok()),
            notes: m.get("notes").cloned(),
        })
    }
}

impl std::fmt::Display for CoreMemoryMeta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Eleutheria Core Memory (trained)")?;
        writeln!(f, "  verze:           {}", self.eleutheria_version)?;
        writeln!(f, "  formát:          v{}", self.format_version)?;
        writeln!(f, "  timestamp:       {}", self.timestamp)?;
        writeln!(f, "  layerů:          {}", self.num_layers)?;
        writeln!(
            f,
            "  rozměry:         n_heads={}, headdim={}, d_state={}",
            self.n_heads, self.headdim, self.d_state
        )?;
        writeln!(f, "  dtype:           {}", self.dtype)?;
        if let Some(s) = self.training_steps {
            writeln!(f, "  training steps:  {s}")?;
        }
        if let Some(l) = self.best_loss {
            writeln!(f, "  best loss:       {l:.4}")?;
        }
        if let Some(l) = self.final_loss {
            writeln!(f, "  final loss:      {l:.4}")?;
        }
        if let Some(n) = &self.notes {
            writeln!(f, "  notes:           {n}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CoreMemoryArtifact
// ---------------------------------------------------------------------------

/// Trained Core Memory na disku — wrapper s metadaty + per-layer F32 tensory.
///
/// Konvence: tensory jsou na CPU ve F32 (matchuje native dtype `Var`).
/// Při aplikaci na live `ModelState` se konvertují na runtime dtype/device.
#[derive(Debug)]
pub struct CoreMemoryArtifact {
    meta: CoreMemoryMeta,
    /// Per-layer init SSM state, indexované podle `layer_idx`. Vždy F32, CPU.
    layers: Vec<Tensor>,
}

impl CoreMemoryArtifact {
    /// Build artefakt z trainable stacku — kopíruje Vars na CPU jako F32.
    ///
    /// `training_steps`/`best_loss`/`final_loss` jsou volitelná telemetrie
    /// (pro lidskou orientaci v `inspect`), neovlivňují validaci.
    pub fn from_stack(
        stack: &CoreMemoryStack,
        config: &FalconH1Config,
        training_steps: Option<usize>,
        best_loss: Option<f64>,
        final_loss: Option<f64>,
        notes: Option<String>,
    ) -> Result<Self> {
        if stack.num_layers() != config.num_hidden_layers {
            return Err(candle_core::Error::Msg(format!(
                "CoreMemoryStack má {} vrstev, config má {} — musí souhlasit",
                stack.num_layers(),
                config.num_hidden_layers
            )));
        }

        let mut layers: Vec<Tensor> = Vec::with_capacity(stack.num_layers());
        for core in &stack.layers {
            let tensor = core
                .init_state
                .as_tensor()
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?;
            layers.push(tensor);
        }

        let meta = CoreMemoryMeta::new(config, training_steps, best_loss, final_loss, notes);
        Ok(Self { meta, layers })
    }

    /// Uloží artefakt na disk jako safetensors s metadaty.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let metadata = self.meta.to_metadata_map();
        let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(self.layers.len());
        for (i, t) in self.layers.iter().enumerate() {
            tensors.insert(format!("{TENSOR_PREFIX}.{i:02}"), t.clone());
        }
        st::serialize_to_file(&tensors, Some(metadata), path.as_ref())
            .map_err(|e| candle_core::Error::Msg(format!("safetensors save: {e}")))?;
        Ok(())
    }

    /// Načte artefakt z disku. Tensory zůstávají na CPU ve F32.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        use candle_core::safetensors::Load;

        let data = std::fs::read(path.as_ref())
            .map_err(|e| candle_core::Error::Msg(format!("čtení souboru: {e}")))?;

        // Metadata
        let (_, raw_meta) = st::SafeTensors::read_metadata(&data)
            .map_err(|e| candle_core::Error::Msg(format!("safetensors metadata: {e}")))?;
        let meta_map = raw_meta.metadata().as_ref().ok_or_else(|| {
            candle_core::Error::Msg("core memory artefakt nemá __metadata__ hlavičku".into())
        })?;
        let meta = CoreMemoryMeta::from_metadata_map(meta_map)?;

        // Tensory — řadíme podle layer indexu z názvu, abychom byli odolní
        // proti neuspořádanému HashMap iteration.
        let st = st::SafeTensors::deserialize(&data)
            .map_err(|e| candle_core::Error::Msg(format!("safetensors deserialize: {e}")))?;

        let mut layers: Vec<Option<Tensor>> = (0..meta.num_layers).map(|_| None).collect();
        for (name, view) in st.tensors().into_iter() {
            let Some(rest) = name.strip_prefix(&format!("{TENSOR_PREFIX}.")) else {
                continue;
            };
            let idx: usize = rest.parse().map_err(|e| {
                candle_core::Error::Msg(format!("neplatný layer index v '{name}': {e}"))
            })?;
            if idx >= meta.num_layers {
                return Err(candle_core::Error::Msg(format!(
                    "layer index {idx} mimo num_layers {}",
                    meta.num_layers
                )));
            }
            let tensor = view.load(&Device::Cpu)?.to_dtype(DType::F32)?;
            layers[idx] = Some(tensor);
        }

        let layers: Vec<Tensor> = layers
            .into_iter()
            .enumerate()
            .map(|(i, slot)| {
                slot.ok_or_else(|| {
                    candle_core::Error::Msg(format!("v artefaktu chybí tensor pro layer {i}"))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { meta, layers })
    }

    /// Pouze metadata — pro rychlou inspekci bez načítání tensorů.
    pub fn inspect<P: AsRef<Path>>(path: P) -> Result<CoreMemoryMeta> {
        let data = std::fs::read(path.as_ref())
            .map_err(|e| candle_core::Error::Msg(format!("čtení souboru: {e}")))?;
        let (_, raw_meta) = st::SafeTensors::read_metadata(&data)
            .map_err(|e| candle_core::Error::Msg(format!("safetensors metadata: {e}")))?;
        let meta_map = raw_meta.metadata().as_ref().ok_or_else(|| {
            candle_core::Error::Msg("core memory artefakt nemá __metadata__ hlavičku".into())
        })?;
        CoreMemoryMeta::from_metadata_map(meta_map)
    }

    /// Validuje kompatibilitu artefaktu s daným modelem.
    pub fn validate_config(&self, config: &FalconH1Config) -> Result<()> {
        let m = &self.meta;
        let mut errors = Vec::new();
        if m.num_layers != config.num_hidden_layers {
            errors.push(format!(
                "num_layers: artefakt={}, config={}",
                m.num_layers, config.num_hidden_layers
            ));
        }
        if m.n_heads != config.mamba_n_heads {
            errors.push(format!(
                "n_heads: artefakt={}, config={}",
                m.n_heads, config.mamba_n_heads
            ));
        }
        if m.headdim != config.mamba_d_head {
            errors.push(format!(
                "headdim: artefakt={}, config={}",
                m.headdim, config.mamba_d_head
            ));
        }
        if m.d_state != config.mamba_d_state {
            errors.push(format!(
                "d_state: artefakt={}, config={}",
                m.d_state, config.mamba_d_state
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(candle_core::Error::Msg(format!(
                "core memory artefakt nekompatibilní s konfigurací:\n  {}",
                errors.join("\n  ")
            )))
        }
    }

    /// Aplikuje per-layer init_state na živý `ModelState`. Tensory se
    /// konvertují na runtime dtype + device. Conv state a KV cache
    /// se nedotýkají (artefakt je nenese).
    pub fn apply_to_state(
        &self,
        state: &mut ModelState,
        device: &Device,
        dtype: DType,
    ) -> Result<()> {
        if self.layers.len() != state.layers.len() {
            return Err(candle_core::Error::Msg(format!(
                "artefakt má {} vrstev, ModelState má {} — nekompatibilní",
                self.layers.len(),
                state.layers.len()
            )));
        }
        for (i, layer_state) in state.layers.iter_mut().enumerate() {
            let trained = self.layers[i].to_dtype(dtype)?.to_device(device)?;
            layer_state.ssm_state = trained;
        }
        Ok(())
    }

    /// Hot-load do trainable `CoreMemoryStack` — umožňuje resume tréninku
    /// (v0.5.0-alpha.15+). Vytváří čerstvé `Var` instances ze saved tensorů.
    pub fn into_stack(self, config: &FalconH1Config, device: &Device) -> Result<CoreMemoryStack> {
        self.validate_config(config)?;
        let layers: Vec<CoreMemory> = self
            .layers
            .into_iter()
            .enumerate()
            .map(|(i, t)| {
                let init_state = Var::from_tensor(&t.to_device(device)?)?;
                Ok(CoreMemory {
                    init_state,
                    layer_idx: i,
                    n_heads: config.mamba_n_heads,
                    headdim: config.mamba_d_head,
                    d_state: config.mamba_d_state,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(CoreMemoryStack { layers })
    }

    /// Reference na metadata (např. pro logging po `attach_core_memory`).
    pub fn meta(&self) -> &CoreMemoryMeta {
        &self.meta
    }

    /// Počet vrstev (pro diagnostiku).
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ---------------------------------------------------------------------------
// Testy
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_config() -> FalconH1Config {
        FalconH1Config {
            vocab_size: 100,
            hidden_size: 32,
            num_hidden_layers: 4,
            intermediate_size: 64,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 16,
            mamba_d_state: 8,
            mamba_n_heads: 2,
            mamba_d_head: 16,
            mamba_d_ssm: 32,
            mamba_d_conv: 4,
            mamba_expand: 2,
            mamba_n_groups: 1,
            mamba_chunk_size: 256,
            mamba_conv_bias: true,
            mamba_proj_bias: false,
            mamba_norm_before_gate: false,
            mamba_rms_norm: true,
            mamba_use_mlp: true,
            embedding_multiplier: 5.66,
            lm_head_multiplier: 0.0195,
            ssm_in_multiplier: 0.4167,
            ssm_out_multiplier: 0.1179,
            ssm_multipliers: vec![0.2946],
            attention_in_multiplier: 1.0,
            attention_out_multiplier: 0.1042,
            key_multiplier: 1.0,
            mlp_multipliers: vec![0.2946],
            rms_norm_eps: 1e-5,
            eos_token_id: Some(11),
            rope_theta: 1e11,
            tie_word_embeddings: false,
            max_position_embeddings: 1000,
        }
    }

    fn unique_path(name: &str) -> std::path::PathBuf {
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("eleutheria_test_{name}_{pid}_{nanos}.safetensors"))
    }

    #[test]
    fn round_trip_preserves_per_layer_tensors() -> Result<()> {
        let config = dummy_config();
        let stack = CoreMemoryStack::randn_small(&config, &Device::Cpu)?;

        // Originální tensory pro porovnání po round-trip.
        let originals: Vec<Tensor> = stack
            .layers
            .iter()
            .map(|c| c.init_state.as_tensor().copy())
            .collect::<Result<Vec<_>>>()?;

        let artifact = CoreMemoryArtifact::from_stack(
            &stack,
            &config,
            Some(123),
            Some(2.5),
            Some(2.7),
            Some("test".into()),
        )?;
        let path = unique_path("round_trip");
        artifact.save(&path)?;

        let loaded = CoreMemoryArtifact::load(&path)?;
        assert_eq!(loaded.meta.num_layers, config.num_hidden_layers);
        assert_eq!(loaded.meta.training_steps, Some(123));
        assert_eq!(loaded.meta.best_loss, Some(2.5));
        assert_eq!(loaded.meta.kind, ARTIFACT_KIND);
        assert_eq!(loaded.layers.len(), config.num_hidden_layers);

        for (i, (orig, restored)) in originals.iter().zip(loaded.layers.iter()).enumerate() {
            assert_eq!(orig.dims(), restored.dims(), "layer {i} shape mismatch");
            let diff: f32 = (orig - restored)?.abs()?.sum_all()?.to_scalar()?;
            assert!(diff < 1e-6, "layer {i} tensor differs (sum |Δ| = {diff})");
        }

        std::fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn apply_to_state_replaces_ssm_states_only() -> Result<()> {
        let config = dummy_config();
        let stack = CoreMemoryStack::randn_small(&config, &Device::Cpu)?;
        let artifact = CoreMemoryArtifact::from_stack(&stack, &config, None, None, None, None)?;

        let mut state = ModelState::new(&config, DType::F32, &Device::Cpu)?;
        // Pre-condition: SSM states mají být nulové.
        for layer in &state.layers {
            let s: f32 = layer.ssm_state.abs()?.sum_all()?.to_scalar()?;
            assert_eq!(s, 0.0);
        }
        artifact.apply_to_state(&mut state, &Device::Cpu, DType::F32)?;

        // Post-condition: SSM states odpovídají artefaktu.
        for (i, layer) in state.layers.iter().enumerate() {
            let diff: f32 = (&layer.ssm_state - &artifact.layers[i])?
                .abs()?
                .sum_all()?
                .to_scalar()?;
            assert!(diff < 1e-6, "layer {i} state ≠ artefakt");
        }

        // Conv state musí zůstat nedotčený (nulový).
        for layer in &state.layers {
            let s: f32 = layer.conv_state.abs()?.sum_all()?.to_scalar()?;
            assert_eq!(s, 0.0, "conv state se neměl měnit");
        }

        Ok(())
    }

    #[test]
    fn inspect_returns_metadata_only() -> Result<()> {
        let config = dummy_config();
        let stack = CoreMemoryStack::randn_small(&config, &Device::Cpu)?;
        let artifact = CoreMemoryArtifact::from_stack(
            &stack,
            &config,
            Some(42),
            Some(1.0),
            None,
            Some("ok".into()),
        )?;
        let path = unique_path("inspect");
        artifact.save(&path)?;

        let meta = CoreMemoryArtifact::inspect(&path)?;
        assert_eq!(meta.kind, ARTIFACT_KIND);
        assert_eq!(meta.num_layers, config.num_hidden_layers);
        assert_eq!(meta.n_heads, config.mamba_n_heads);
        assert_eq!(meta.training_steps, Some(42));
        assert_eq!(meta.best_loss, Some(1.0));
        assert_eq!(meta.notes.as_deref(), Some("ok"));

        std::fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn incompatible_config_rejected() -> Result<()> {
        let config = dummy_config();
        let stack = CoreMemoryStack::randn_small(&config, &Device::Cpu)?;
        let artifact = CoreMemoryArtifact::from_stack(&stack, &config, None, None, None, None)?;

        let mut bad = dummy_config();
        bad.mamba_d_state = 16; // místo 8
        let result = artifact.validate_config(&bad);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("d_state"), "chyba má zmínit d_state: {msg}");

        Ok(())
    }

    #[test]
    fn into_stack_preserves_tensors_and_yields_trainable_vars() -> Result<()> {
        let config = dummy_config();
        let stack = CoreMemoryStack::randn_small(&config, &Device::Cpu)?;
        let originals: Vec<Tensor> = stack
            .layers
            .iter()
            .map(|c| c.init_state.as_tensor().copy())
            .collect::<Result<Vec<_>>>()?;

        let artifact = CoreMemoryArtifact::from_stack(&stack, &config, None, None, None, None)?;
        let restored = artifact.into_stack(&config, &Device::Cpu)?;
        assert_eq!(restored.num_layers(), config.num_hidden_layers);

        for (i, (orig, layer)) in originals.iter().zip(restored.layers.iter()).enumerate() {
            let diff: f32 = (orig - layer.init_state.as_tensor())?
                .abs()?
                .sum_all()?
                .to_scalar()?;
            assert!(diff < 1e-6, "layer {i} mismatch po into_stack");
            assert_eq!(layer.layer_idx, i);
        }

        // Vars musí být trainable — non-empty Vec přes vars_owned.
        let vars = restored.vars_owned();
        assert_eq!(vars.len(), config.num_hidden_layers);
        Ok(())
    }

    #[test]
    fn load_rejects_wrong_kind() -> Result<()> {
        // Vytvoř safetensors s "kind" = něco jiného → load musí spadnout.
        let path = unique_path("wrong_kind");
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "init_state.00".into(),
            Tensor::zeros((2, 16, 8), DType::F32, &Device::Cpu)?,
        );
        let mut meta = HashMap::new();
        meta.insert("eleutheria_version".into(), "test".into());
        meta.insert("format_version".into(), "1".into());
        meta.insert("kind".into(), "session".into()); // špatný kind
        meta.insert("num_layers".into(), "1".into());
        meta.insert("n_heads".into(), "2".into());
        meta.insert("headdim".into(), "16".into());
        meta.insert("d_state".into(), "8".into());
        meta.insert("dtype".into(), "F32".into());
        meta.insert("timestamp".into(), "now".into());
        st::serialize_to_file(&tensors, Some(meta), &path)
            .map_err(|e| candle_core::Error::Msg(format!("save: {e}")))?;

        let result = CoreMemoryArtifact::load(&path);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("kind="), "chyba má zmínit kind: {msg}");

        std::fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn metadata_display_renders_telemetry() {
        let meta = CoreMemoryMeta {
            eleutheria_version: "0.5.0-alpha.14".into(),
            format_version: 1,
            kind: ARTIFACT_KIND.into(),
            num_layers: 24,
            n_heads: 24,
            headdim: 64,
            d_state: 256,
            dtype: "F32".into(),
            timestamp: "2026-04-29T10:00:00+00:00".into(),
            training_steps: Some(500),
            best_loss: Some(2.31),
            final_loss: Some(2.45),
            notes: Some("law_pack v1".into()),
        };
        let s = format!("{meta}");
        assert!(s.contains("training steps:  500"));
        assert!(s.contains("best loss:       2.3100"));
        assert!(s.contains("law_pack v1"));
    }
}
