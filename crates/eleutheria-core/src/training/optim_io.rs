//! Save/Load AdamW optimizer state — sourozenec `CoreMemoryArtifact`.
//!
//! `OptimizerArtifact` ukládá per-Var `first_moment` (m) + `second_moment` (v)
//! plus `step_t` counter, aby `train-core-memory --resume-from` mohl
//! pokračovat se **stejným efektivním LR a velocity bufferem**, místo aby
//! Adam musel znova projít warmup fází (RN-006).
//!
//! ## Sourozenecká konvence
//!
//! Vedle `core_memory.safetensors` ukládáme `core_memory.optim.safetensors`
//! (přípona `.optim.safetensors`). Auto-discovery: pokud existuje při
//! `--resume-from <core>`, načte se i optimizer state. Pokud chybí,
//! `--resume-from` proběhne s prázdným Adamem (alpha.15 chování — backwards
//! compatible). Žádný explicit flag pro toto se nezavádí — buď sourozenec je,
//! nebo není.
//!
//! ## Formát
//!
//! Safetensors s tensory:
//! - `m.{i:02}` per layer (first_moment)
//! - `v.{i:02}` per layer (second_moment)
//!
//! Plus `__metadata__` hlavička s `kind=core_memory_optim`, `step_t`,
//! AdamW parametry (lr, beta1, beta2, eps, weight_decay) — pro budoucí
//! audit (jaké HP byly při tréninku, který tato moments produkoval).
//!
//! ## Validace
//!
//! Při load se ověří `kind`, počet vrstev, shape per Var. Nesouhlasící
//! konfigurace je `Err`. `step_t` musí být ≥ 1 (po prvním stepu); 0
//! by znamenalo prázdný state — to je pak chyba `restore_moments`
//! (uživatel by měl prostě nepoužívat sourozence).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::ParamsAdamW;
use safetensors::tensor as st;

use crate::falcon_h1::config::FalconH1Config;
use crate::training::adamw_state::EleutheriaAdamW;

/// Identifikuje formát souboru — odlišení od `StateCheckpoint` a `CoreMemoryArtifact`.
const ARTIFACT_KIND: &str = "core_memory_optim";
/// Verze formátu.
const FORMAT_VERSION: u32 = 1;
/// Prefix tensorů pro first_moment a second_moment.
const M_PREFIX: &str = "m";
const V_PREFIX: &str = "v";

/// Vrátí očekávanou cestu sourozeneckého `.optim.safetensors` pro daný
/// `CoreMemoryArtifact` path. Konvence: `<core_memory>.optim.safetensors`.
///
/// Příklady:
/// - `core_memory.safetensors` → `core_memory.optim.safetensors`
/// - `~/.eleutheria/cm.safetensors` → `~/.eleutheria/cm.optim.safetensors`
/// - bez extenze: `cm` → `cm.optim.safetensors`
pub fn sibling_path<P: AsRef<Path>>(core_memory_path: P) -> PathBuf {
    let p = core_memory_path.as_ref();
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("artifact");
    let optim_filename = format!("{stem}.optim.safetensors");
    match p.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.join(optim_filename),
        _ => PathBuf::from(optim_filename),
    }
}

// ---------------------------------------------------------------------------
// OptimizerMeta
// ---------------------------------------------------------------------------

/// Metadata AdamW state artefaktu. Drží `step_t` a Adam HP, plus strukturní
/// info pro validaci kompatibility (počet vrstev, shape per Var).
#[derive(Debug, Clone)]
pub struct OptimizerMeta {
    pub eleutheria_version: String,
    pub format_version: u32,
    pub kind: String,
    pub num_layers: usize,
    pub n_heads: usize,
    pub headdim: usize,
    pub d_state: usize,
    pub dtype: String,
    pub timestamp: String,
    /// Kolik `step()` volání optimizer stihl před uložením. Po restore
    /// se nastaví `EleutheriaAdamW::step_t = step_t`.
    pub step_t: usize,
    /// Adam parametry — informativní (validace kontroluje jen shape).
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl OptimizerMeta {
    fn new(config: &FalconH1Config, step_t: usize, params: &ParamsAdamW) -> Self {
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
            step_t,
            lr: params.lr,
            beta1: params.beta1,
            beta2: params.beta2,
            eps: params.eps,
            weight_decay: params.weight_decay,
        }
    }

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
        m.insert("step_t".into(), self.step_t.to_string());
        m.insert("lr".into(), format!("{:.10e}", self.lr));
        m.insert("beta1".into(), format!("{:.10}", self.beta1));
        m.insert("beta2".into(), format!("{:.10}", self.beta2));
        m.insert("eps".into(), format!("{:.10e}", self.eps));
        m.insert("weight_decay".into(), format!("{:.10}", self.weight_decay));
        m
    }

    fn from_metadata_map(m: &HashMap<String, String>) -> Result<Self> {
        let get = |key: &str| -> Result<&str> {
            m.get(key).map(|s| s.as_str()).ok_or_else(|| {
                candle_core::Error::Msg(format!("optimizer metadata chybí klíč: {key}"))
            })
        };
        let parse_usize = |key: &str| -> Result<usize> {
            get(key)?
                .parse::<usize>()
                .map_err(|e| candle_core::Error::Msg(format!("{key}: {e}")))
        };
        let parse_f64 = |key: &str| -> Result<f64> {
            get(key)?
                .parse::<f64>()
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
            step_t: parse_usize("step_t")?,
            lr: parse_f64("lr")?,
            beta1: parse_f64("beta1")?,
            beta2: parse_f64("beta2")?,
            eps: parse_f64("eps")?,
            weight_decay: parse_f64("weight_decay")?,
        })
    }
}

impl std::fmt::Display for OptimizerMeta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Eleutheria AdamW optimizer state")?;
        writeln!(f, "  verze:           {}", self.eleutheria_version)?;
        writeln!(f, "  formát:          v{}", self.format_version)?;
        writeln!(f, "  timestamp:       {}", self.timestamp)?;
        writeln!(f, "  step_t:          {}", self.step_t)?;
        writeln!(f, "  layerů:          {}", self.num_layers)?;
        writeln!(
            f,
            "  rozměry:         n_heads={}, headdim={}, d_state={}",
            self.n_heads, self.headdim, self.d_state
        )?;
        writeln!(f, "  dtype:           {}", self.dtype)?;
        writeln!(
            f,
            "  AdamW HP:        lr={:.4e}, β1={:.3}, β2={:.4}, eps={:.1e}, wd={:.3}",
            self.lr, self.beta1, self.beta2, self.eps, self.weight_decay
        )?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// OptimizerArtifact
// ---------------------------------------------------------------------------

/// AdamW state na disku — wrapper s metadaty + per-Var (m, v) F32 tensory.
#[derive(Debug)]
pub struct OptimizerArtifact {
    meta: OptimizerMeta,
    /// Per-Var `(first_moment, second_moment)`, vždy F32 na CPU.
    moments: Vec<(Tensor, Tensor)>,
}

impl OptimizerArtifact {
    /// Build artefakt z optimizéru — kopíruje m, v na CPU jako F32.
    pub fn from_optimizer(opt: &EleutheriaAdamW, config: &FalconH1Config) -> Result<Self> {
        if opt.state().len() != config.num_hidden_layers {
            return Err(candle_core::Error::Msg(format!(
                "EleutheriaAdamW má {} Var-ů, config má {} layerů — musí souhlasit",
                opt.state().len(),
                config.num_hidden_layers
            )));
        }
        let moments = opt.snapshot_moments()?;
        let meta = OptimizerMeta::new(config, opt.step_t(), opt.params());
        Ok(Self { meta, moments })
    }

    /// Uloží artefakt na disk jako safetensors s metadaty.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let metadata = self.meta.to_metadata_map();
        let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(self.moments.len() * 2);
        for (i, (m, v)) in self.moments.iter().enumerate() {
            tensors.insert(format!("{M_PREFIX}.{i:02}"), m.clone());
            tensors.insert(format!("{V_PREFIX}.{i:02}"), v.clone());
        }
        st::serialize_to_file(&tensors, Some(metadata), path.as_ref())
            .map_err(|e| candle_core::Error::Msg(format!("safetensors save: {e}")))?;
        Ok(())
    }

    /// Načte artefakt z disku.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        use candle_core::safetensors::Load;

        let data = std::fs::read(path.as_ref())
            .map_err(|e| candle_core::Error::Msg(format!("čtení souboru: {e}")))?;

        let (_, raw_meta) = st::SafeTensors::read_metadata(&data)
            .map_err(|e| candle_core::Error::Msg(format!("safetensors metadata: {e}")))?;
        let meta_map = raw_meta.metadata().as_ref().ok_or_else(|| {
            candle_core::Error::Msg("optimizer artefakt nemá __metadata__ hlavičku".into())
        })?;
        let meta = OptimizerMeta::from_metadata_map(meta_map)?;

        let safetensors = st::SafeTensors::deserialize(&data)
            .map_err(|e| candle_core::Error::Msg(format!("safetensors deserialize: {e}")))?;

        let mut m_slots: Vec<Option<Tensor>> = (0..meta.num_layers).map(|_| None).collect();
        let mut v_slots: Vec<Option<Tensor>> = (0..meta.num_layers).map(|_| None).collect();

        for (name, view) in safetensors.tensors().into_iter() {
            let (slots, rest) = if let Some(rest) = name.strip_prefix(&format!("{M_PREFIX}.")) {
                (&mut m_slots, rest)
            } else if let Some(rest) = name.strip_prefix(&format!("{V_PREFIX}.")) {
                (&mut v_slots, rest)
            } else {
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
            slots[idx] = Some(tensor);
        }

        let moments: Vec<(Tensor, Tensor)> = m_slots
            .into_iter()
            .zip(v_slots)
            .enumerate()
            .map(|(i, (m, v))| {
                let m = m.ok_or_else(|| {
                    candle_core::Error::Msg(format!("v artefaktu chybí {M_PREFIX}.{i:02}"))
                })?;
                let v = v.ok_or_else(|| {
                    candle_core::Error::Msg(format!("v artefaktu chybí {V_PREFIX}.{i:02}"))
                })?;
                Ok((m, v))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { meta, moments })
    }

    /// Pouze metadata — pro rychlou inspekci.
    pub fn inspect<P: AsRef<Path>>(path: P) -> Result<OptimizerMeta> {
        let data = std::fs::read(path.as_ref())
            .map_err(|e| candle_core::Error::Msg(format!("čtení souboru: {e}")))?;
        let (_, raw_meta) = st::SafeTensors::read_metadata(&data)
            .map_err(|e| candle_core::Error::Msg(format!("safetensors metadata: {e}")))?;
        let meta_map = raw_meta.metadata().as_ref().ok_or_else(|| {
            candle_core::Error::Msg("optimizer artefakt nemá __metadata__ hlavičku".into())
        })?;
        OptimizerMeta::from_metadata_map(meta_map)
    }

    /// Validuje kompatibilitu artefaktu s daným modelem (rozměry SSM stavu).
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
                "optimizer artefakt nekompatibilní s konfigurací:\n  {}",
                errors.join("\n  ")
            )))
        }
    }

    /// Aplikuje uložený m, v + step_t na čerstvě vytvořený optimizer.
    /// Vrací `Err` pokud rozměry nesouhlasí.
    pub fn apply_to_optimizer(&self, opt: &mut EleutheriaAdamW) -> Result<()> {
        opt.restore_moments(&self.moments, self.meta.step_t)
    }

    /// Reference na metadata.
    pub fn meta(&self) -> &OptimizerMeta {
        &self.meta
    }

    /// Počet vrstev v artefaktu.
    pub fn num_layers(&self) -> usize {
        self.moments.len()
    }
}

// ---------------------------------------------------------------------------
// Testy
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::core_memory::CoreMemoryStack;
    use candle_core::Var;
    use candle_nn::optim::Optimizer;

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

    fn unique_path(name: &str) -> PathBuf {
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("eleutheria_test_{name}_{pid}_{nanos}.safetensors"))
    }

    /// Bootstrap GradStore (private constructor v Candle) přes triviální
    /// backward na dummy Var, pak override grad pro každý real var.
    fn fake_grads_for(vars: &[Var], grad_value: f32) -> Result<candle_core::backprop::GradStore> {
        let device = vars
            .first()
            .map(|v| v.device().clone())
            .unwrap_or(Device::Cpu);
        let dummy = Var::zeros((1,), DType::F32, &device)?;
        let loss = dummy.as_tensor().sum_all()?;
        let mut grads = loss.backward()?;
        for var in vars {
            let g = Tensor::ones(var.shape(), var.dtype(), var.device())?;
            let g = (g * grad_value as f64)?;
            grads.insert(var.as_tensor(), g);
        }
        Ok(grads)
    }

    /// Sibling path konvence pro typické vstupy.
    #[test]
    fn sibling_path_appends_optim_extension() {
        let p = sibling_path("/foo/core_memory.safetensors");
        assert_eq!(
            p.to_string_lossy(),
            "/foo/core_memory.optim.safetensors",
            "získané: {}",
            p.display()
        );

        let p = sibling_path("artifact.safetensors");
        assert_eq!(p.to_string_lossy(), "artifact.optim.safetensors");

        let p = sibling_path("/tmp/cm");
        assert_eq!(p.to_string_lossy(), "/tmp/cm.optim.safetensors");
    }

    /// Round-trip přes safetensors zachovává m, v i step_t.
    #[test]
    fn round_trip_preserves_moments_and_step_t() -> Result<()> {
        let config = dummy_config();
        let stack = CoreMemoryStack::randn_small(&config, &Device::Cpu)?;
        let vars = stack.vars_owned();
        let mut opt = EleutheriaAdamW::new(vars.clone(), ParamsAdamW::default())?;

        // Provedeme 3 stepy s různými gradienty, ať m, v ani step_t nejsou nulové.
        for s in 0..3 {
            let grads = fake_grads_for(&vars, 0.1 + s as f32 * 0.05)?;
            opt.step(&grads)?;
        }
        assert_eq!(opt.step_t(), 3);

        // Reference snapshot pro porovnání po round-trip.
        let snapshot_before = opt.snapshot_moments()?;

        let artifact = OptimizerArtifact::from_optimizer(&opt, &config)?;
        let path = unique_path("optim_round_trip");
        artifact.save(&path)?;

        let loaded = OptimizerArtifact::load(&path)?;
        assert_eq!(loaded.meta().step_t, 3);
        assert_eq!(loaded.meta().num_layers, config.num_hidden_layers);
        assert_eq!(loaded.meta().kind, ARTIFACT_KIND);
        assert_eq!(loaded.num_layers(), config.num_hidden_layers);

        for (i, ((m_a, v_a), (m_b, v_b))) in snapshot_before
            .iter()
            .zip(loaded.moments.iter())
            .enumerate()
        {
            let dm: f32 = (m_a - m_b)?.abs()?.sum_all()?.to_scalar()?;
            let dv: f32 = (v_a - v_b)?.abs()?.sum_all()?.to_scalar()?;
            assert!(dm < 1e-6, "layer {i} m mismatch po round-trip ({dm})");
            assert!(dv < 1e-6, "layer {i} v mismatch po round-trip ({dv})");
        }

        std::fs::remove_file(&path).ok();
        Ok(())
    }

    /// Apply na čerstvý optimizer obnoví m, v, step_t přesně.
    #[test]
    fn apply_to_optimizer_restores_state() -> Result<()> {
        let config = dummy_config();
        let stack = CoreMemoryStack::randn_small(&config, &Device::Cpu)?;
        let vars = stack.vars_owned();
        let mut opt_a = EleutheriaAdamW::new(vars.clone(), ParamsAdamW::default())?;
        for s in 0..3 {
            let grads = fake_grads_for(&vars, 0.1 + s as f32 * 0.05)?;
            opt_a.step(&grads)?;
        }
        let snap_a = opt_a.snapshot_moments()?;

        let artifact = OptimizerArtifact::from_optimizer(&opt_a, &config)?;

        // Nový optimizer: state je nulový + step_t=0
        let mut opt_b = EleutheriaAdamW::new(vars.clone(), ParamsAdamW::default())?;
        assert_eq!(opt_b.step_t(), 0);

        artifact.apply_to_optimizer(&mut opt_b)?;
        assert_eq!(opt_b.step_t(), 3);

        let snap_b = opt_b.snapshot_moments()?;
        for (i, ((m_a, v_a), (m_b, v_b))) in snap_a.iter().zip(snap_b.iter()).enumerate() {
            let dm: f32 = (m_a - m_b)?.abs()?.sum_all()?.to_scalar()?;
            let dv: f32 = (v_a - v_b)?.abs()?.sum_all()?.to_scalar()?;
            assert!(dm < 1e-6, "layer {i} m po apply ({dm})");
            assert!(dv < 1e-6, "layer {i} v po apply ({dv})");
        }
        Ok(())
    }

    /// Inspect vrátí jen metadata bez načítání tensorů.
    #[test]
    fn inspect_returns_metadata_only() -> Result<()> {
        let config = dummy_config();
        let stack = CoreMemoryStack::randn_small(&config, &Device::Cpu)?;
        let vars = stack.vars_owned();
        let mut opt = EleutheriaAdamW::new(vars.clone(), ParamsAdamW::default())?;
        let grads = fake_grads_for(&vars, 0.1)?;
        opt.step(&grads)?;

        let artifact = OptimizerArtifact::from_optimizer(&opt, &config)?;
        let path = unique_path("optim_inspect");
        artifact.save(&path)?;

        let meta = OptimizerArtifact::inspect(&path)?;
        assert_eq!(meta.kind, ARTIFACT_KIND);
        assert_eq!(meta.step_t, 1);
        assert_eq!(meta.num_layers, config.num_hidden_layers);

        std::fs::remove_file(&path).ok();
        Ok(())
    }

    /// Validate odhalí nekompatibilní rozměry.
    #[test]
    fn validate_config_rejects_shape_mismatch() -> Result<()> {
        let config = dummy_config();
        let stack = CoreMemoryStack::randn_small(&config, &Device::Cpu)?;
        let vars = stack.vars_owned();
        let mut opt = EleutheriaAdamW::new(vars.clone(), ParamsAdamW::default())?;
        let grads = fake_grads_for(&vars, 0.1)?;
        opt.step(&grads)?;

        let artifact = OptimizerArtifact::from_optimizer(&opt, &config)?;

        let mut bad = dummy_config();
        bad.mamba_d_state = 16;
        let result = artifact.validate_config(&bad);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("d_state"));

        Ok(())
    }

    /// Load odmítne soubor s jiným `kind`.
    #[test]
    fn load_rejects_wrong_kind() -> Result<()> {
        let path = unique_path("optim_wrong_kind");
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "m.00".into(),
            Tensor::zeros((2, 16, 8), DType::F32, &Device::Cpu)?,
        );
        tensors.insert(
            "v.00".into(),
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
        meta.insert("step_t".into(), "1".into());
        meta.insert("lr".into(), "0.001".into());
        meta.insert("beta1".into(), "0.9".into());
        meta.insert("beta2".into(), "0.999".into());
        meta.insert("eps".into(), "1e-8".into());
        meta.insert("weight_decay".into(), "0.01".into());
        st::serialize_to_file(&tensors, Some(meta), &path)
            .map_err(|e| candle_core::Error::Msg(format!("save: {e}")))?;

        let result = OptimizerArtifact::load(&path);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("kind="));

        std::fs::remove_file(&path).ok();
        Ok(())
    }

    /// Display formátuje step_t a HP.
    #[test]
    fn metadata_display_renders_step_and_hp() {
        let meta = OptimizerMeta {
            eleutheria_version: "0.5.0-alpha.16".into(),
            format_version: 1,
            kind: ARTIFACT_KIND.into(),
            num_layers: 24,
            n_heads: 24,
            headdim: 64,
            d_state: 256,
            dtype: "F32".into(),
            timestamp: "2026-04-29T10:00:00+00:00".into(),
            step_t: 500,
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        };
        let s = format!("{meta}");
        assert!(s.contains("step_t:          500"));
        assert!(s.contains("lr=1.0000e-3"));
        assert!(s.contains("β1=0.900"));
    }
}
