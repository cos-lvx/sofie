//! Serializace a deserializace stavů Falcon-H1.
//!
//! `StateCheckpoint` je přenosový formát mezi živým `ModelState` (GPU, runtime)
//! a diskem (safetensors). Podporuje selektivní save/load přes `StateFilter`:
//! - `full()` — kompletní session resume (SSM + conv + KV cache)
//! - `core_memory()` — SSM + conv bez KV (pro state tuning, Core Memory)
//! - `ssm_only()` — pouze SSM state (experimentální state injection)
//!
//! Formát: safetensors s `__metadata__` hlavičkou obsahující konfiguraci modelu,
//! pozici v sekvenci, filtr a timestamp.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use safetensors::tensor as st;

use super::config::FalconH1Config;
use super::state::ModelState;

/// Candle Result alias — celý modul pracuje s candle_core::Error.
type Result<T> = candle_core::Result<T>;

// ---------------------------------------------------------------------------
// StateFilter
// ---------------------------------------------------------------------------

/// Filtr pro selektivní save/load komponent stavu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateFilter {
    pub ssm_state: bool,
    pub conv_state: bool,
    pub kv_cache: bool,
}

impl StateFilter {
    /// Vše — kompletní session resume.
    pub fn full() -> Self {
        Self {
            ssm_state: true,
            conv_state: true,
            kv_cache: true,
        }
    }

    /// Jádro paměti — SSM + conv, bez KV cache.
    /// Pro state tuning (v0.5.0) a Core Memory.
    pub fn core_memory() -> Self {
        Self {
            ssm_state: true,
            conv_state: true,
            kv_cache: false,
        }
    }

    /// Pouze SSM — experimentální state injection.
    pub fn ssm_only() -> Self {
        Self {
            ssm_state: true,
            conv_state: false,
            kv_cache: false,
        }
    }

    /// Popis filtru pro metadata.
    pub fn label(&self) -> &'static str {
        match (self.ssm_state, self.conv_state, self.kv_cache) {
            (true, true, true) => "full",
            (true, true, false) => "core_memory",
            (true, false, false) => "ssm_only",
            _ => "custom",
        }
    }

    /// Parsuj label zpět na filtr.
    fn from_label(label: &str) -> Self {
        match label {
            "full" => Self::full(),
            "core_memory" => Self::core_memory(),
            "ssm_only" => Self::ssm_only(),
            _ => Self::full(), // fallback — detekujeme přítomnost tensorů při loadu
        }
    }
}

// ---------------------------------------------------------------------------
// CheckpointMeta
// ---------------------------------------------------------------------------

/// Metadata checkpointu — vše potřebné pro validaci a restore.
#[derive(Debug, Clone)]
pub struct CheckpointMeta {
    pub eleutheria_version: String,
    pub format_version: u32,
    pub num_layers: usize,
    pub position: usize,
    pub dtype: String,
    pub filter: StateFilter,
    pub timestamp: String,
    // Rozměry modelu pro validaci kompatibility
    pub d_state: usize,
    pub d_conv: usize,
    pub n_heads: usize,
    pub headdim: usize,
    pub n_kv_heads: usize,
    pub kv_head_dim: usize,
    pub kv_seq_len: Option<usize>,
}

impl CheckpointMeta {
    /// Vytvoří metadata z aktuálního stavu a konfigurace.
    fn from_state(
        config: &FalconH1Config,
        pos: usize,
        dtype: DType,
        filter: StateFilter,
        kv_seq_len: Option<usize>,
    ) -> Self {
        Self {
            eleutheria_version: env!("CARGO_PKG_VERSION").to_string(),
            format_version: 1,
            num_layers: config.num_hidden_layers,
            position: pos,
            dtype: format!("{:?}", dtype),
            filter,
            timestamp: chrono::Utc::now().to_rfc3339(),
            d_state: config.mamba_d_state,
            d_conv: config.mamba_d_conv,
            n_heads: config.mamba_n_heads,
            headdim: config.mamba_d_head,
            n_kv_heads: config.num_key_value_heads,
            kv_head_dim: config.head_dim,
            kv_seq_len,
        }
    }

    /// Serializuj do safetensors metadata mapy (vše jako String).
    fn to_metadata_map(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("eleutheria_version".into(), self.eleutheria_version.clone());
        m.insert("format_version".into(), self.format_version.to_string());
        m.insert("num_layers".into(), self.num_layers.to_string());
        m.insert("position".into(), self.position.to_string());
        m.insert("dtype".into(), self.dtype.clone());
        m.insert("filter".into(), self.filter.label().into());
        m.insert("timestamp".into(), self.timestamp.clone());
        m.insert("d_state".into(), self.d_state.to_string());
        m.insert("d_conv".into(), self.d_conv.to_string());
        m.insert("n_heads".into(), self.n_heads.to_string());
        m.insert("headdim".into(), self.headdim.to_string());
        m.insert("n_kv_heads".into(), self.n_kv_heads.to_string());
        m.insert("kv_head_dim".into(), self.kv_head_dim.to_string());
        m.insert("has_ssm_state".into(), self.filter.ssm_state.to_string());
        m.insert("has_conv_state".into(), self.filter.conv_state.to_string());
        m.insert("has_kv_cache".into(), self.filter.kv_cache.to_string());
        if let Some(seq_len) = self.kv_seq_len {
            m.insert("kv_seq_len".into(), seq_len.to_string());
        }
        m
    }

    /// Parsuj z safetensors metadata mapy.
    fn from_metadata_map(m: &HashMap<String, String>) -> Result<Self> {
        let get = |key: &str| -> Result<&str> {
            m.get(key).map(|s| s.as_str()).ok_or_else(|| {
                candle_core::Error::Msg(format!("checkpoint metadata chybí klíč: {key}"))
            })
        };

        let parse_usize = |key: &str| -> Result<usize> {
            get(key)?
                .parse::<usize>()
                .map_err(|e| candle_core::Error::Msg(format!("{key}: {e}")))
        };

        Ok(Self {
            eleutheria_version: get("eleutheria_version")?.to_string(),
            format_version: get("format_version")?
                .parse::<u32>()
                .map_err(|e| candle_core::Error::Msg(format!("format_version: {e}")))?,
            num_layers: parse_usize("num_layers")?,
            position: parse_usize("position")?,
            dtype: get("dtype")?.to_string(),
            filter: StateFilter::from_label(get("filter")?),
            timestamp: get("timestamp")?.to_string(),
            d_state: parse_usize("d_state")?,
            d_conv: parse_usize("d_conv")?,
            n_heads: parse_usize("n_heads")?,
            headdim: parse_usize("headdim")?,
            n_kv_heads: parse_usize("n_kv_heads")?,
            kv_head_dim: parse_usize("kv_head_dim")?,
            kv_seq_len: m.get("kv_seq_len").and_then(|s| s.parse::<usize>().ok()),
        })
    }
}

impl std::fmt::Display for CheckpointMeta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Eleutheria State Checkpoint")?;
        writeln!(f, "  verze:      {}", self.eleutheria_version)?;
        writeln!(f, "  formát:     v{}", self.format_version)?;
        writeln!(f, "  timestamp:  {}", self.timestamp)?;
        writeln!(f, "  filtr:      {}", self.filter.label())?;
        writeln!(f, "  pozice:     {} tokenů", self.position)?;
        writeln!(f, "  dtype:      {}", self.dtype)?;
        writeln!(f, "  layerů:     {}", self.num_layers)?;
        writeln!(
            f,
            "  rozměry:    d_state={}, n_heads={}, headdim={}",
            self.d_state, self.n_heads, self.headdim
        )?;
        writeln!(
            f,
            "  attention:  n_kv_heads={}, kv_head_dim={}",
            self.n_kv_heads, self.kv_head_dim
        )?;
        if let Some(seq_len) = self.kv_seq_len {
            writeln!(f, "  KV cache:   {} tokenů", seq_len)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// StateCheckpoint
// ---------------------------------------------------------------------------

/// Přenosový formát pro stav modelu.
/// Drží CPU tensory + metadata, připravený k save/load.
pub struct StateCheckpoint {
    pub meta: CheckpointMeta,
    tensors: HashMap<String, Tensor>,
}

impl StateCheckpoint {
    /// Vytvoří checkpoint z živého ModelState.
    /// Tensory se kopírují na CPU.
    pub fn from_model_state(
        state: &ModelState,
        pos: usize,
        config: &FalconH1Config,
        filter: StateFilter,
    ) -> Result<Self> {
        let mut tensors = HashMap::new();
        let mut kv_seq_len: Option<usize> = None;

        for (i, layer) in state.layers.iter().enumerate() {
            let prefix = format!("layer.{i:02}");

            if filter.ssm_state {
                tensors.insert(
                    format!("{prefix}.ssm_state"),
                    layer.ssm_state.to_device(&Device::Cpu)?,
                );
            }

            if filter.conv_state {
                tensors.insert(
                    format!("{prefix}.conv_state"),
                    layer.conv_state.to_device(&Device::Cpu)?,
                );
            }

            if filter.kv_cache {
                let k = layer.k_cache.to_device(&Device::Cpu)?;
                // KV cache seq_len — bereme z prvního layeru, předpokládáme konzistenci
                if kv_seq_len.is_none() {
                    kv_seq_len = Some(k.dim(2)?);
                }
                tensors.insert(format!("{prefix}.k_cache"), k);
                tensors.insert(
                    format!("{prefix}.v_cache"),
                    layer.v_cache.to_device(&Device::Cpu)?,
                );
            }
        }

        let meta = CheckpointMeta::from_state(
            config,
            pos,
            state.layers[0].ssm_state.dtype(),
            filter,
            kv_seq_len,
        );

        Ok(Self { meta, tensors })
    }

    /// Uloží checkpoint na disk jako safetensors s metadaty.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let metadata = self.meta.to_metadata_map();
        st::serialize_to_file(&self.tensors, Some(metadata), path.as_ref())
            .map_err(|e| candle_core::Error::Msg(format!("safetensors save: {e}")))?;
        Ok(())
    }

    /// Načte checkpoint z disku. Tensory zůstávají na CPU.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        use candle_core::safetensors::Load;

        let data = std::fs::read(path.as_ref())
            .map_err(|e| candle_core::Error::Msg(format!("čtení souboru: {e}")))?;

        // Metadata
        let (_, raw_meta) = st::SafeTensors::read_metadata(&data)
            .map_err(|e| candle_core::Error::Msg(format!("safetensors metadata: {e}")))?;
        let meta_map = raw_meta.metadata().as_ref().ok_or_else(|| {
            candle_core::Error::Msg("checkpoint nemá __metadata__ hlavičku".into())
        })?;
        let meta = CheckpointMeta::from_metadata_map(meta_map)?;

        // Tensory
        let st = st::SafeTensors::deserialize(&data)
            .map_err(|e| candle_core::Error::Msg(format!("safetensors deserialize: {e}")))?;
        let tensors: HashMap<String, Tensor> = st
            .tensors()
            .into_iter()
            .map(|(name, view)| {
                let tensor = view.load(&Device::Cpu)?;
                Ok((name, tensor))
            })
            .collect::<Result<_>>()?;

        Ok(Self { meta, tensors })
    }

    /// Metadata bez načítání tensorů — pro rychlou inspekci.
    pub fn inspect<P: AsRef<Path>>(path: P) -> Result<CheckpointMeta> {
        let data = std::fs::read(path.as_ref())
            .map_err(|e| candle_core::Error::Msg(format!("čtení souboru: {e}")))?;
        let (_, raw_meta) = st::SafeTensors::read_metadata(&data)
            .map_err(|e| candle_core::Error::Msg(format!("safetensors metadata: {e}")))?;
        let meta_map = raw_meta.metadata().as_ref().ok_or_else(|| {
            candle_core::Error::Msg("checkpoint nemá __metadata__ hlavičku".into())
        })?;
        CheckpointMeta::from_metadata_map(meta_map)
    }

    /// Aplikuje checkpoint na existující ModelState.
    /// Přepíše pouze komponenty přítomné v checkpointu.
    /// Vrací pozici ze checkpointu.
    pub fn apply_to_model_state(
        &self,
        state: &mut ModelState,
        device: &Device,
        dtype: DType,
    ) -> Result<usize> {
        self.validate_layer_count(state.layers.len())?;

        for (i, layer) in state.layers.iter_mut().enumerate() {
            let prefix = format!("layer.{i:02}");

            if let Some(t) = self.tensors.get(&format!("{prefix}.ssm_state")) {
                layer.ssm_state = t.to_dtype(dtype)?.to_device(device)?;
            }

            if let Some(t) = self.tensors.get(&format!("{prefix}.conv_state")) {
                layer.conv_state = t.to_dtype(dtype)?.to_device(device)?;
            }

            if let Some(t) = self.tensors.get(&format!("{prefix}.k_cache")) {
                layer.k_cache = t.to_dtype(dtype)?.to_device(device)?;
            }

            if let Some(t) = self.tensors.get(&format!("{prefix}.v_cache")) {
                layer.v_cache = t.to_dtype(dtype)?.to_device(device)?;
            }
        }

        Ok(self.meta.position)
    }

    /// Vytvoří čerstvý ModelState z checkpointu.
    /// Komponenty chybějící v checkpointu se inicializují na nuly.
    pub fn into_model_state(
        &self,
        config: &FalconH1Config,
        device: &Device,
        dtype: DType,
    ) -> Result<(ModelState, usize)> {
        let mut state = ModelState::new(config, dtype, device)?;
        let pos = self.apply_to_model_state(&mut state, device, dtype)?;
        Ok((state, pos))
    }

    /// Validuje kompatibilitu checkpointu s danou konfigurací.
    pub fn validate_config(&self, config: &FalconH1Config) -> Result<()> {
        let m = &self.meta;
        let mut errors = Vec::new();

        if m.num_layers != config.num_hidden_layers {
            errors.push(format!(
                "num_layers: checkpoint={}, config={}",
                m.num_layers, config.num_hidden_layers
            ));
        }
        if m.d_state != config.mamba_d_state {
            errors.push(format!(
                "d_state: checkpoint={}, config={}",
                m.d_state, config.mamba_d_state
            ));
        }
        if m.n_heads != config.mamba_n_heads {
            errors.push(format!(
                "n_heads: checkpoint={}, config={}",
                m.n_heads, config.mamba_n_heads
            ));
        }
        if m.headdim != config.mamba_d_head {
            errors.push(format!(
                "headdim: checkpoint={}, config={}",
                m.headdim, config.mamba_d_head
            ));
        }
        if m.n_kv_heads != config.num_key_value_heads {
            errors.push(format!(
                "n_kv_heads: checkpoint={}, config={}",
                m.n_kv_heads, config.num_key_value_heads
            ));
        }
        if m.kv_head_dim != config.head_dim {
            errors.push(format!(
                "kv_head_dim: checkpoint={}, config={}",
                m.kv_head_dim, config.head_dim
            ));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(candle_core::Error::Msg(format!(
                "checkpoint nekompatibilní s konfigurací:\n  {}",
                errors.join("\n  ")
            )))
        }
    }

    /// Interní validace počtu layerů.
    fn validate_layer_count(&self, expected: usize) -> Result<()> {
        if self.meta.num_layers != expected {
            return Err(candle_core::Error::Msg(format!(
                "checkpoint má {} layerů, model očekává {}",
                self.meta.num_layers, expected
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Testy
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Pomocná konfigurace pro testy — malý "model" s 2 layery.
    fn test_config() -> FalconH1Config {
        FalconH1Config {
            vocab_size: 100,
            hidden_size: 32,
            num_hidden_layers: 2,
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
            max_position_embeddings: 131072,
        }
    }

    /// Vytvoří ModelState s nenulovými hodnotami pro testování round-trip.
    fn test_state_with_data(config: &FalconH1Config) -> Result<ModelState> {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let mut state = ModelState::new(config, dtype, &device)?;

        for (i, layer) in state.layers.iter_mut().enumerate() {
            let val = (i + 1) as f32;
            // SSM state: naplň konstantní hodnotou
            layer.ssm_state =
                Tensor::full(val, layer.ssm_state.shape(), &device)?.to_dtype(dtype)?;
            // Conv state: jinou hodnotou
            layer.conv_state =
                Tensor::full(val * 10.0, layer.conv_state.shape(), &device)?.to_dtype(dtype)?;
            // KV cache: simuluj 3 tokeny
            let kv_shape = (1, config.num_key_value_heads, 3, config.head_dim);
            layer.k_cache = Tensor::full(val * 100.0, kv_shape, &device)?.to_dtype(dtype)?;
            layer.v_cache = Tensor::full(val * 200.0, kv_shape, &device)?.to_dtype(dtype)?;
        }

        Ok(state)
    }

    /// Porovnej dva tensory element-wise.
    fn tensors_equal(a: &Tensor, b: &Tensor) -> Result<bool> {
        if a.shape() != b.shape() {
            return Ok(false);
        }
        let diff = (a - b)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        Ok(diff == 0.0)
    }

    #[test]
    fn test_round_trip_full() -> Result<()> {
        let config = test_config();
        let state = test_state_with_data(&config)?;
        let pos = 42;

        // Save
        let checkpoint =
            StateCheckpoint::from_model_state(&state, pos, &config, StateFilter::full())?;
        let tmp = std::env::temp_dir().join("eleutheria_test_full.safetensors");
        checkpoint.save(&tmp)?;

        // Load
        let loaded = StateCheckpoint::load(&tmp)?;
        let (restored, restored_pos) =
            loaded.into_model_state(&config, &Device::Cpu, DType::F32)?;

        // Ověř
        assert_eq!(restored_pos, pos);
        assert_eq!(restored.layers.len(), state.layers.len());
        for (i, (orig, rest)) in state.layers.iter().zip(restored.layers.iter()).enumerate() {
            assert!(
                tensors_equal(&orig.ssm_state, &rest.ssm_state)?,
                "layer {i} ssm_state se liší"
            );
            assert!(
                tensors_equal(&orig.conv_state, &rest.conv_state)?,
                "layer {i} conv_state se liší"
            );
            assert!(
                tensors_equal(&orig.k_cache, &rest.k_cache)?,
                "layer {i} k_cache se liší"
            );
            assert!(
                tensors_equal(&orig.v_cache, &rest.v_cache)?,
                "layer {i} v_cache se liší"
            );
        }

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }

    #[test]
    fn test_round_trip_core_memory() -> Result<()> {
        let config = test_config();
        let state = test_state_with_data(&config)?;

        // Save jen SSM + conv
        let checkpoint =
            StateCheckpoint::from_model_state(&state, 100, &config, StateFilter::core_memory())?;
        let tmp = std::env::temp_dir().join("eleutheria_test_core.safetensors");
        checkpoint.save(&tmp)?;

        // Load do čerstvého stavu
        let loaded = StateCheckpoint::load(&tmp)?;
        assert_eq!(loaded.meta.filter, StateFilter::core_memory());

        let (restored, _) = loaded.into_model_state(&config, &Device::Cpu, DType::F32)?;

        // SSM a conv musí sedět
        for (i, (orig, rest)) in state.layers.iter().zip(restored.layers.iter()).enumerate() {
            assert!(
                tensors_equal(&orig.ssm_state, &rest.ssm_state)?,
                "layer {i} ssm_state se liší"
            );
            assert!(
                tensors_equal(&orig.conv_state, &rest.conv_state)?,
                "layer {i} conv_state se liší"
            );
        }

        // KV cache musí být prázdná (dim 2 == 0)
        for (i, layer) in restored.layers.iter().enumerate() {
            assert_eq!(
                layer.k_cache.dim(2)?,
                0,
                "layer {i} k_cache měla být prázdná"
            );
            assert_eq!(
                layer.v_cache.dim(2)?,
                0,
                "layer {i} v_cache měla být prázdná"
            );
        }

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }

    #[test]
    fn test_metadata_round_trip() -> Result<()> {
        let config = test_config();
        let state = test_state_with_data(&config)?;

        let checkpoint =
            StateCheckpoint::from_model_state(&state, 1523, &config, StateFilter::full())?;
        let tmp = std::env::temp_dir().join("eleutheria_test_meta.safetensors");
        checkpoint.save(&tmp)?;

        // Inspect (bez načítání tensorů)
        let meta = StateCheckpoint::inspect(&tmp)?;
        assert_eq!(meta.position, 1523);
        assert_eq!(meta.num_layers, 2);
        assert_eq!(meta.d_state, 8);
        assert_eq!(meta.n_heads, 2);
        assert_eq!(meta.headdim, 16);
        assert_eq!(meta.n_kv_heads, 1);
        assert_eq!(meta.kv_head_dim, 16);
        assert_eq!(meta.filter, StateFilter::full());
        assert_eq!(meta.kv_seq_len, Some(3));
        assert_eq!(meta.format_version, 1);

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }

    #[test]
    fn test_config_validation_incompatible() -> Result<()> {
        let config = test_config();
        let state = test_state_with_data(&config)?;

        let checkpoint =
            StateCheckpoint::from_model_state(&state, 0, &config, StateFilter::full())?;

        // Nekompatibilní konfigurace — jiný d_state
        let mut bad_config = test_config();
        bad_config.mamba_d_state = 999;

        let result = checkpoint.validate_config(&bad_config);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("d_state"),
            "chyba by měla zmínit d_state: {err_msg}"
        );

        Ok(())
    }

    #[test]
    fn test_apply_selective() -> Result<()> {
        let config = test_config();
        let state = test_state_with_data(&config)?;

        // Ulož jen SSM
        let checkpoint =
            StateCheckpoint::from_model_state(&state, 50, &config, StateFilter::ssm_only())?;
        let tmp = std::env::temp_dir().join("eleutheria_test_ssm.safetensors");
        checkpoint.save(&tmp)?;

        // Načti a aplikuj na čerstvý stav
        let loaded = StateCheckpoint::load(&tmp)?;
        let mut fresh = ModelState::new(&config, DType::F32, &Device::Cpu)?;

        // Conv a KV by měly zůstat nulové
        let pos = loaded.apply_to_model_state(&mut fresh, &Device::Cpu, DType::F32)?;
        assert_eq!(pos, 50);

        // SSM state musí sedět
        for (i, (orig, rest)) in state.layers.iter().zip(fresh.layers.iter()).enumerate() {
            assert!(
                tensors_equal(&orig.ssm_state, &rest.ssm_state)?,
                "layer {i} ssm_state se liší"
            );
        }

        // Conv state by měl být nulový (nebyl v checkpointu)
        for (i, layer) in fresh.layers.iter().enumerate() {
            let sum = layer.conv_state.abs()?.sum_all()?.to_scalar::<f32>()?;
            assert_eq!(sum, 0.0, "layer {i} conv_state měl být nulový");
        }

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }

    #[test]
    fn test_state_filter_labels() {
        assert_eq!(StateFilter::full().label(), "full");
        assert_eq!(StateFilter::core_memory().label(), "core_memory");
        assert_eq!(StateFilter::ssm_only().label(), "ssm_only");

        let custom = StateFilter {
            ssm_state: false,
            conv_state: true,
            kv_cache: false,
        };
        assert_eq!(custom.label(), "custom");
    }
}
