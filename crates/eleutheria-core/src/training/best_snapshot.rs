//! `BestSnapshotTracker` — zachycuje stav s nejnižší loss během tréninku.
//!
//! Po RN-008 (KI-007 refutovaná) a RN-009 (KI-008 hypotéza refutovaná) víme,
//! že Phase 2 overshoot v Mamba-2 + SSM tréninku je hluboce strukturní —
//! ani Adam state ani LR scheduling ho neeliminují. **Důsledek:** noisy
//! trajektorie je realita, ne bug. `from_stack` ukládá `Var` hodnoty
//! v okamžiku save volání (= final state), což pro noisy trajektorii
//! často znamená state s loss daleko od best.
//!
//! Tento modul řeší KI-009 deterministicky: shadow CPU buffer per Var,
//! aktualizuje se **pouze** při skutečném zlepšení loss. Po skončení
//! tréninku má `from_snapshot` přímou cestu k best stavu.
//!
//! ## Design
//!
//! - **F32 CPU storage** — matchuje native dtype `Var` v `CoreMemoryStack`
//!   a CPU layout `CoreMemoryArtifact`, takže round-trip je bez konverze.
//! - **Lazy update** — copy GPU→CPU (PCIe transfer ~30 ms na 1.5B/24
//!   vrstev) probíhá jen když `loss < best`. Pro typickou noisy trajektorii
//!   s 5-10 best update events za 156 stepů to je ~150-300 ms total
//!   overhead, neblokující.
//! - **Opt-in** — `TrainingConfig.track_best: bool`. Default `false`
//!   zachovává alpha.16/17 chování (save final). `--save-best` CLI flag
//!   to zapne.
//! - **Periodic flush na disk (alpha.20, KI-012)** — `flush_to_disk`
//!   serializuje aktuální shadow buffer jako `CoreMemoryArtifact`
//!   atomicky (write na `.<name>.tmp`, rename). Pojistka proti pádu /
//!   preempci cloud GPU instance — bez ní se best v RAM ztratí
//!   s celým procesem (~3-5 hodin compute zahozeno).

use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};

use crate::falcon_h1::config::FalconH1Config;
use crate::training::core_memory::CoreMemoryStack;
use crate::training::core_memory_io::CoreMemoryArtifact;

/// Shadow CPU buffer s nejlepším stavem trajektorie.
///
/// Drží per-Var F32 tensory na CPU. Aktualizuje se přes `update_if_better`
/// — pokud aktuální loss zlepšuje historický best, kopíruje aktuální Var
/// hodnoty na CPU. Po tréninku `into_snapshot()` vrátí buffery (nebo
/// `None`, pokud nikdy nebyly zachyceny — degenerate run).
#[derive(Debug)]
pub struct BestSnapshotTracker {
    /// Per-Var F32 CPU tensor s best stavem. `None` = ještě nikdy
    /// nezachyceno (před prvním successful update).
    layers: Option<Vec<Tensor>>,
    /// Nejlepší loss zaznamenaná dosud.
    best_loss: f64,
    /// Step counter (0-indexovaný), kdy byl best zachycen. -1 = nikdy.
    best_step: i64,
    /// Kolik update_if_better volání (sanity counter).
    total_updates: usize,
    /// Kolik snapshotů skutečně uloženo (= kolikrát loss zlepšila).
    successful_updates: usize,
}

impl BestSnapshotTracker {
    /// Nový tracker s prázdným bufferem.
    pub fn new() -> Self {
        Self {
            layers: None,
            best_loss: f64::INFINITY,
            best_step: -1,
            total_updates: 0,
            successful_updates: 0,
        }
    }

    /// Pokud `loss < self.best_loss`, zachyť aktuální Var hodnoty z `stack`
    /// jako F32 CPU tensory. Jinak skip (no GPU→CPU transfer).
    ///
    /// `step` je per-run counter (0-indexovaný), drží se v meta pro audit
    /// (best_step()).
    pub fn update_if_better(
        &mut self,
        loss: f64,
        step: usize,
        stack: &CoreMemoryStack,
    ) -> Result<()> {
        self.total_updates += 1;
        if !loss.is_finite() {
            return Ok(()); // NaN nikdy nezlepšuje best
        }
        if loss < self.best_loss {
            let mut copies = Vec::with_capacity(stack.num_layers());
            for core in &stack.layers {
                let cpu_f32 = core
                    .init_state
                    .as_tensor()
                    .to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?;
                copies.push(cpu_f32);
            }
            self.layers = Some(copies);
            self.best_loss = loss;
            self.best_step = step as i64;
            self.successful_updates += 1;
        }
        Ok(())
    }

    /// `true` pokud aspoň jednou byl zachycen snapshot (= aspoň jeden
    /// micro-batch měl finite loss).
    pub fn has_snapshot(&self) -> bool {
        self.layers.is_some()
    }

    /// Nejlepší loss (`f64::INFINITY` pokud žádný snapshot).
    pub fn best_loss(&self) -> f64 {
        self.best_loss
    }

    /// Step, ve kterém byl best zachycen (0-indexed). `None` = nikdy.
    pub fn best_step(&self) -> Option<usize> {
        if self.best_step < 0 {
            None
        } else {
            Some(self.best_step as usize)
        }
    }

    /// Počet `update_if_better` volání celkem.
    pub fn total_updates(&self) -> usize {
        self.total_updates
    }

    /// Počet úspěšných snapshot zachycení (= jak často loss klesla pod best).
    pub fn successful_updates(&self) -> usize {
        self.successful_updates
    }

    /// Konzumuj tracker a vrať per-Var F32 CPU tensory pro
    /// `CoreMemoryArtifact::from_snapshot`. `None` = nikdy nezachyceno.
    pub fn into_snapshot(self) -> Option<Vec<Tensor>> {
        self.layers
    }

    /// Atomicky uloží aktuální best snapshot na disk jako
    /// `CoreMemoryArtifact` (alpha.20, KI-012). Pokud tracker dosud
    /// nemá zachycený snapshot (`has_snapshot() == false`), vrátí
    /// `Ok(false)` bez I/O. Při úspěchu vrací `Ok(true)`.
    ///
    /// **Atomic write:** zapíše do sourozeneckého `.<name>.tmp`,
    /// pak `rename(tmp, path)`. Rename na stejném filesystému je
    /// atomic na POSIX — pád mezi save a rename nepřepíše prior verzi.
    /// Tedy z pohledu čtenáře cílový soubor buď drží předchozí verzi,
    /// nebo nově zapsanou — nikdy half-written.
    ///
    /// **Cena:** clone tensorů (Arc-based, levné) + safetensors
    /// serialize (~75 MB pro 1.5B Falcon-H1, ~200-500 ms na lokální
    /// disk, na cloud SSD podobné). Pro periodic flush každých 5-10
    /// stepů na production setupu (44 s/step) je overhead < 1 %.
    ///
    /// Pokud potřebuješ flush jako součást training loopu, který
    /// nesmí selhat při disk chybě, obal v `.ok()` na callsite —
    /// chyba se loguje, training pokračuje.
    pub fn flush_to_disk(
        &self,
        path: &Path,
        config: &FalconH1Config,
        cumulative_steps: Option<usize>,
        best_loss: Option<f64>,
        final_loss: Option<f64>,
        notes: Option<String>,
    ) -> Result<bool> {
        let Some(layers) = &self.layers else {
            return Ok(false);
        };
        // Clone tensorů — `from_snapshot` konzumuje `Vec<Tensor>`. Tensors
        // jsou Arc-based, clone neduplikuje storage.
        let snapshot: Vec<Tensor> = layers.clone();
        let artifact = CoreMemoryArtifact::from_snapshot(
            snapshot,
            config,
            cumulative_steps,
            best_loss,
            final_loss,
            notes,
        )?;
        atomic_save_artifact(&artifact, path)?;
        Ok(true)
    }
}

/// Atomic write `CoreMemoryArtifact` na `path` přes sourozenecké
/// `.<filename>.tmp` + `rename`. Rename na stejném filesystému je
/// atomic na POSIX (Linux). Při crash mezi save a rename zůstane
/// na cílové cestě předchozí verze (pokud existovala) — žádný
/// half-written soubor.
fn atomic_save_artifact(artifact: &CoreMemoryArtifact, path: &Path) -> Result<()> {
    let file_name = path.file_name().and_then(|n| n.to_str()).ok_or_else(|| {
        candle_core::Error::Msg(format!("flush path bez file_name: {}", path.display()))
    })?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let tmp_path = parent.join(format!(".{file_name}.tmp"));
    artifact.save(&tmp_path)?;
    std::fs::rename(&tmp_path, path).map_err(|e| {
        // Smaž tmp pokud rename selhal — neostaneme s orphan soubory.
        let _ = std::fs::remove_file(&tmp_path);
        candle_core::Error::Msg(format!(
            "atomic rename {} → {}: {e}",
            tmp_path.display(),
            path.display()
        ))
    })?;
    Ok(())
}

impl Default for BestSnapshotTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::falcon_h1::config::FalconH1Config;

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

    #[test]
    fn fresh_tracker_has_no_snapshot() {
        let t = BestSnapshotTracker::new();
        assert!(!t.has_snapshot());
        assert_eq!(t.best_loss(), f64::INFINITY);
        assert_eq!(t.best_step(), None);
        assert_eq!(t.total_updates(), 0);
        assert_eq!(t.successful_updates(), 0);
    }

    #[test]
    fn first_finite_loss_creates_snapshot() -> Result<()> {
        let config = dummy_config();
        let stack = CoreMemoryStack::randn_small(&config, &Device::Cpu)?;
        let mut t = BestSnapshotTracker::new();
        t.update_if_better(5.0, 0, &stack)?;
        assert!(t.has_snapshot());
        assert_eq!(t.best_loss(), 5.0);
        assert_eq!(t.best_step(), Some(0));
        assert_eq!(t.successful_updates(), 1);
        Ok(())
    }

    #[test]
    fn worse_loss_does_not_overwrite() -> Result<()> {
        let config = dummy_config();
        let stack = CoreMemoryStack::randn_small(&config, &Device::Cpu)?;
        let mut t = BestSnapshotTracker::new();
        t.update_if_better(5.0, 0, &stack)?;
        t.update_if_better(7.0, 1, &stack)?;
        assert_eq!(t.best_loss(), 5.0);
        assert_eq!(t.best_step(), Some(0));
        assert_eq!(t.total_updates(), 2);
        assert_eq!(t.successful_updates(), 1);
        Ok(())
    }

    #[test]
    fn nan_loss_skips_update() -> Result<()> {
        let config = dummy_config();
        let stack = CoreMemoryStack::randn_small(&config, &Device::Cpu)?;
        let mut t = BestSnapshotTracker::new();
        t.update_if_better(f64::NAN, 0, &stack)?;
        assert!(!t.has_snapshot());
        assert_eq!(t.successful_updates(), 0);
        // total_updates se ale inkrementuje (sanity counter)
        assert_eq!(t.total_updates(), 1);
        Ok(())
    }

    /// Klíčový test: simulujeme noisy trajektorii s best v mid-runu.
    /// Tracker musí zachytit snapshot ze step 2 (loss=2.5), ne ze step 4
    /// (loss=8.0).
    #[test]
    fn captures_state_at_best_step_not_final() -> Result<()> {
        let config = dummy_config();
        let device = Device::Cpu;

        // Vytvoř 5 různých stacků, každý s rozeznatelným init_state.
        // Stack i má tensory naplněné konstantou (i+1).
        let stacks: Vec<CoreMemoryStack> = (0..5)
            .map(|i| {
                let mut s = CoreMemoryStack::zeros(&config, &device).unwrap();
                for layer in &mut s.layers {
                    let shape = layer.init_state.as_tensor().shape().clone();
                    let value = (i + 1) as f64;
                    let new_tensor = Tensor::ones(&shape, DType::F32, &device).unwrap();
                    let new_tensor = (new_tensor * value).unwrap();
                    layer.init_state.set(&new_tensor).unwrap();
                }
                s
            })
            .collect();

        // Trajektorie: [10, 5, 2.5, 8, 6.5]. Best je step 2 (loss=2.5).
        let losses = [10.0, 5.0, 2.5, 8.0, 6.5];
        let mut t = BestSnapshotTracker::new();
        for (step, &loss) in losses.iter().enumerate() {
            t.update_if_better(loss, step, &stacks[step])?;
        }

        assert_eq!(t.best_loss(), 2.5);
        assert_eq!(t.best_step(), Some(2));
        // 3 successful updates: step 0 (init), step 1 (5 < 10), step 2 (2.5 < 5)
        assert_eq!(t.successful_updates(), 3);
        assert_eq!(t.total_updates(), 5);

        // Verify snapshot drží tensory ze step 2 (value=3.0), ne ze step 4 (value=5.0).
        let snapshot = t.into_snapshot().expect("snapshot must exist");
        assert_eq!(snapshot.len(), config.num_hidden_layers);
        for (i, tensor) in snapshot.iter().enumerate() {
            let mean: f32 = tensor.mean_all()?.to_scalar()?;
            assert!(
                (mean - 3.0).abs() < 1e-5,
                "layer {i} should have value=3.0 (step 2), got {mean}"
            );
        }
        Ok(())
    }

    /// Tracker přežije celý run bez best (všechny loss = NaN). into_snapshot
    /// vrátí None — caller pak fallbackuje na from_stack.
    #[test]
    fn no_finite_loss_yields_none_snapshot() -> Result<()> {
        let config = dummy_config();
        let stack = CoreMemoryStack::randn_small(&config, &Device::Cpu)?;
        let mut t = BestSnapshotTracker::new();
        t.update_if_better(f64::NAN, 0, &stack)?;
        t.update_if_better(f64::INFINITY, 1, &stack)?;
        // INFINITY není < INFINITY, takže neukládá. Ale je finite? Ne — INFINITY není finite.
        assert!(!t.has_snapshot());
        let snap = t.into_snapshot();
        assert!(snap.is_none());
        Ok(())
    }

    fn unique_path(name: &str) -> std::path::PathBuf {
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("eleutheria_flush_{name}_{pid}_{nanos}.safetensors"))
    }

    /// flush_to_disk uloží snapshot atomicky a load znovu načte stejné
    /// tensory + metadata.
    #[test]
    fn flush_to_disk_round_trip() -> Result<()> {
        let config = dummy_config();
        let device = Device::Cpu;

        // Vyrob stack s tenzory naplněnými konstantou 7.0, simuluj 1 best update.
        let mut stack = CoreMemoryStack::zeros(&config, &device)?;
        for layer in &mut stack.layers {
            let shape = layer.init_state.as_tensor().shape().clone();
            let new_tensor = (Tensor::ones(&shape, DType::F32, &device)? * 7.0)?;
            layer.init_state.set(&new_tensor)?;
        }
        let mut t = BestSnapshotTracker::new();
        t.update_if_better(0.42, 113, &stack)?;
        assert!(t.has_snapshot());

        let path = unique_path("round_trip");
        let flushed = t.flush_to_disk(
            &path,
            &config,
            Some(150),
            Some(0.42),
            Some(3.69),
            Some("flush test".into()),
        )?;
        assert!(
            flushed,
            "flush_to_disk vrátilo true při existujícím snapshotu"
        );
        assert!(path.exists(), "flush vytvořil cílový soubor");

        // .tmp soubor nesmí zůstat
        let parent = path.parent().unwrap();
        let file_name = path.file_name().unwrap().to_str().unwrap();
        let tmp = parent.join(format!(".{file_name}.tmp"));
        assert!(
            !tmp.exists(),
            "tmp soubor zůstal na disku: {}",
            tmp.display()
        );

        // Load + verify
        let loaded = CoreMemoryArtifact::load(&path)?;
        assert_eq!(loaded.meta().training_steps, Some(150));
        assert_eq!(loaded.meta().best_loss, Some(0.42));
        assert_eq!(loaded.meta().final_loss, Some(3.69));
        assert_eq!(loaded.meta().notes.as_deref(), Some("flush test"));
        assert_eq!(loaded.num_layers(), config.num_hidden_layers);

        // Tracker zůstává validní po flush — můžeme znovu update + flush
        // (insurance smysl: opakované flushe každých N stepů).
        for layer in &mut stack.layers {
            let shape = layer.init_state.as_tensor().shape().clone();
            let new_tensor = (Tensor::ones(&shape, DType::F32, &device)? * 9.0)?;
            layer.init_state.set(&new_tensor)?;
        }
        t.update_if_better(0.30, 200, &stack)?;
        assert_eq!(t.best_loss(), 0.30);
        // Druhý flush přepíše předchozí soubor (atomic rename = overwrite).
        let flushed_again = t.flush_to_disk(&path, &config, Some(200), Some(0.30), None, None)?;
        assert!(flushed_again);
        let loaded2 = CoreMemoryArtifact::load(&path)?;
        assert_eq!(loaded2.meta().best_loss, Some(0.30));
        assert_eq!(loaded2.meta().training_steps, Some(200));

        std::fs::remove_file(&path).ok();
        Ok(())
    }

    /// flush_to_disk vrací false (no-op) pokud tracker nemá snapshot.
    #[test]
    fn flush_to_disk_noop_when_no_snapshot() -> Result<()> {
        let config = dummy_config();
        let t = BestSnapshotTracker::new();
        let path = unique_path("noop");
        let flushed = t.flush_to_disk(&path, &config, None, None, None, None)?;
        assert!(!flushed, "flush bez snapshotu musí vrátit false");
        assert!(!path.exists(), "flush bez snapshotu nesmí vytvořit soubor");
        Ok(())
    }

    /// flush nepřepíše prior soubor částečně — atomic rename garantuje,
    /// že před save tmp existuje na sourozenecké cestě, a path obsahuje
    /// buď prior verzi, nebo novou (nikdy half-written).
    #[test]
    fn flush_uses_dotted_tmp_sibling() -> Result<()> {
        let config = dummy_config();
        let device = Device::Cpu;
        let stack = CoreMemoryStack::randn_small(&config, &device)?;
        let mut t = BestSnapshotTracker::new();
        t.update_if_better(1.0, 0, &stack)?;

        let path = unique_path("dotted");
        t.flush_to_disk(&path, &config, None, None, None, None)?;

        // Po flushi musí existovat path, ne tmp.
        assert!(path.exists());
        let parent = path.parent().unwrap();
        let file_name = path.file_name().unwrap().to_str().unwrap();
        let tmp = parent.join(format!(".{file_name}.tmp"));
        assert!(!tmp.exists());

        std::fs::remove_file(&path).ok();
        Ok(())
    }
}
