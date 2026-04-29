//! Eleutheria CLI — Sofie's local mind.
use anyhow::{Result, anyhow};
use clap::{Args as ClapArgs, Parser, Subcommand};
use eleutheria_core::bench::{
    BenchVariant, RetentionBench, built_in_probes, harness::DEFAULT_DISTANCES,
};
use eleutheria_core::falcon_h1::layer::LayerStop;
use eleutheria_core::training::trace;
use eleutheria_core::{
    CoreMemoryArtifact, GenerateControl, Sofie, SofieSession, StateCheckpoint, StateFilter,
};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::str::FromStr;

/// Eleutheria — lokální inference engine pro Falcon-H1.
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Model: "1.5b", "7b", nebo přímá cesta k adresáři
    #[arg(short, long, default_value = "1.5b")]
    model: String,

    /// Prompt pro single-shot generování (bez --prompt = REPL mód)
    #[arg(short, long)]
    prompt: Option<String>,

    /// Maximální počet tokenů k vygenerování
    #[arg(short = 'n', long, default_value_t = 100)]
    max_tokens: usize,

    /// Temperature (0.0 = deterministické, 1.0 = náhodné)
    #[arg(short, long, default_value_t = 0.8)]
    temperature: f64,

    /// Použít CUDA (GPU)
    #[arg(long)]
    cuda: bool,

    /// Cesta k persona TOML souboru
    #[arg(long, default_value = "persona/sofie.toml")]
    persona: String,

    /// Načíst state z checkpointu
    #[arg(long)]
    load_state: Option<PathBuf>,

    /// Uložit state po generování (single-shot mód)
    #[arg(long)]
    save_state: Option<PathBuf>,

    /// Filtr pro uložení stavu: full, core, ssm
    #[arg(long, default_value = "full")]
    state_filter: String,

    /// Inspekce checkpointu — vypiš metadata a skonči
    #[arg(long)]
    inspect_state: Option<PathBuf>,

    /// Inspekce trénovaného Core Memory artefaktu — vypiš metadata a skonči
    #[arg(long)]
    inspect_core_memory: Option<PathBuf>,

    /// Pokračuj v poslední session (~/.eleutheria/last_session.safetensors)
    #[arg(long)]
    resume: bool,

    /// Cesta k trénovanému Core Memory artefaktu. Pokud neuvedeno,
    /// auto-discovery z `~/.eleutheria/core_memory.safetensors`. Použij
    /// `--no-core-memory` pro vypnutí.
    #[arg(long)]
    core_memory: Option<PathBuf>,

    /// Zakaž auto-load Core Memory (i defaultní path se přeskočí).
    #[arg(long)]
    no_core_memory: bool,

    /// Specializované podkomandy (benchmarky, nástroje)
    #[command(subcommand)]
    command: Option<Command>,
}

/// Podkomandy Eleutherie — jednotlivé specializované módy.
#[derive(Subcommand, Debug)]
enum Command {
    /// Retention benchmark — měří SSM state retenci na různých vzdálenostech.
    BenchRetention(BenchRetentionArgs),
    /// Core Memory autograd smoke test (Fáze 5 alpha) — ověří, že gradient
    /// teče od loss zpět k trainable initial SSM state jedné vrstvy.
    TrainCoreMemorySmoke(TrainCoreMemorySmokeArgs),
    /// Multi-layer Core Memory smoke (Fáze 5 alpha.10+) — trénuje init_state
    /// všech vrstev najednou s cross-entropy loss na next-token. Ověří,
    /// že autograd teče ke všem trainable Vars a gradient signál je
    /// rozumně distribuovaný napříč vrstvami.
    TrainCoreMemoryMulti(TrainCoreMemoryMultiArgs),
    /// Core Memory training loop (Fáze 5 alpha.11+) — produkční varianta
    /// multi-layer smoke. Načte textový dataset, tokenizuje, chunkuje,
    /// trénuje přes epochs × batches s gradient accumulation a AdamW.
    /// Bez save/load (alpha.12) a bez distilátu (alpha.13) — minimum
    /// funkční loop pro ověření, že loss klesá.
    TrainCoreMemory(TrainCoreMemoryArgs),
}

/// Argumenty pro `bench-retention` subkomand.
#[derive(ClapArgs, Debug)]
struct BenchRetentionArgs {
    /// Varianta benchmarku: full | ssm_only | cold | all
    #[arg(long, default_value = "full")]
    variant: String,

    /// Vzdálenosti v tokenech, čárkou oddělené (default: 50,200,500,1000,2000)
    #[arg(long)]
    distances: Option<String>,

    /// Base path pro výstup (bez přípony; doplní se `.json` a `.md`)
    #[arg(long, default_value = "/tmp/eleutheria_retention_bench")]
    output: String,

    /// Volitelná poznámka k runu — zapíše se do `meta.notes`
    #[arg(long)]
    notes: Option<String>,

    /// Zapnout Sofie personu i pro bench (default: vypnuto).
    ///
    /// Bez persony běží bench s čistým ChatML `<|im_start|>user\n…` bez
    /// system promptu — měří **model-level SSM retenci**, ne Sofie-specific
    /// behavior. Důvody pro default-off:
    /// - persona je česky, probes jsou EN → jazyková inkonzistence v SSM kontextu
    /// - persona instruuje "mysli v krocích" → delší odpovědi, klíčová slova
    ///   často mimo 80-token budget
    /// - model může odpovědět česky navzdory "odpovídej v jazyce, ve kterém
    ///   ti bylo napsáno" → false negatives v matcheru (hledá EN substrings)
    /// - ~180 tokenů persony posouvá absolute position v SSM, zkresluje
    ///   měření krátkých vzdáleností
    #[arg(long)]
    with_persona: bool,
}

/// Argumenty pro `train-core-memory-smoke` subkomand.
#[derive(ClapArgs, Debug)]
struct TrainCoreMemorySmokeArgs {
    /// Počet tokenů ve vstupu (default 10 — minimum pro nízký peak VRAM).
    #[arg(long, default_value_t = 10)]
    seq_len: usize,

    /// Index Mamba-2 vrstvy, jejíž initial state trénujeme.
    #[arg(long, default_value_t = 0)]
    layer_idx: usize,

    /// AdamW learning rate. Default 1e-3 — bezpečná hodnota pro smoke test
    /// (RWKV doporučuje 1.0 pro State Tuning, ale až po warmup s funkčním
    /// forward pass; pro alpha bring-up chceme dosáhnout step bez exploze).
    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f64,

    /// Zastav forward po vrstvě `cut_at_layer` (včetně) a loss počítej
    /// z hidden stream místo z logits. Izoluje backward na úsek
    /// `[layer_idx ..= cut_at_layer]` — diagnostika pro hledání NaN op.
    #[arg(long)]
    cut_at_layer: Option<usize>,

    /// Diagnostický sweep — spustí smoke pro každý cut od `layer_idx` až
    /// po poslední vrstvu + plný forward, vypíše tabulku výsledků.
    /// Navíc měří forward hidden norms po každé vrstvě.
    #[arg(long)]
    sweep: bool,

    /// Gradient clipping (max L2 norm). Standardní Mamba-2 recept: 1.0.
    /// Pokud je Peri-LN massive activations root cause NaN gradientu,
    /// clipping ho odblokuje (research verdict z v0.5.0-alpha.5).
    #[arg(long)]
    grad_clip: Option<f64>,

    /// Sub-layer cut bod na **poslední** trénované vrstvě (vrstva
    /// `cut_at_layer`). Varianty: `pre-norm`, `ssm`, `attn`, `residual1`,
    /// `post-norm`, `mlp-gate`, `mlp-silu-mul`, `mlp-down`, `full` (default).
    /// Umožňuje binary search lokalizaci op s NaN backward (BUG-010 alpha.8).
    #[arg(long)]
    cut_at_component: Option<String>,

    /// Zapne forward tensor stats sink — po forward pass vytiskne tabulku
    /// abs_max/abs_min_nonzero/mean/l2/NaN/Inf pro každý probe bod v layer,
    /// mixer, attention. Diagnostika pro BUG-010 alpha.8. Nezasahuje backward.
    #[arg(long)]
    trace: bool,
}

/// Argumenty pro `train-core-memory` — produkční training loop.
#[derive(ClapArgs, Debug)]
struct TrainCoreMemoryArgs {
    /// Cesta k textovému souboru (UTF-8) s trénovacím korpusem.
    #[arg(long)]
    dataset: PathBuf,

    /// Počet epoch.
    #[arg(long, default_value_t = 1)]
    epochs: usize,

    /// Délka trénovací sekvence v tokenech.
    #[arg(long, default_value_t = 16)]
    seq_len: usize,

    /// Micro-batch size (počet sekvencí v jednom forward passu).
    /// Pro RTX 4050 6 GB s seq_len 16 použij 1; větší VRAM umožní 2–4.
    #[arg(long, default_value_t = 1)]
    batch_size: usize,

    /// Gradient accumulation steps — kolik micro-batches se akumuluje
    /// před jedním optimizer.step(). Efektivní batch = batch_size × grad_accum.
    #[arg(long, default_value_t = 4)]
    grad_accum: usize,

    /// AdamW learning rate.
    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f64,

    /// Max L2 norm pro gradient clipping (0 = vypnuto).
    #[arg(long, default_value_t = 1.0)]
    grad_clip: f64,

    /// Jak často logovat running loss (po N optimizer steps).
    #[arg(long, default_value_t = 10)]
    log_every: usize,

    /// Seed pro shuffle dataset pořadí (epoch_idx se přičte).
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Přidat BOS token na začátek korpusu (default true pro sjednocený
    /// start jako v inference; false pro kontinuální korpus bez BOS mid-stream).
    #[arg(long, default_value_t = true)]
    add_bos: bool,

    /// Použít gradient checkpointing (per-layer chunked backward).
    /// Sníží peak memory cca 10–20× za cenu ~2× delšího každého kroku.
    /// Cílové použití: CUDA s ≤ 12 GB VRAM kde plný backward graf nesedí
    /// (KI-005). Pro malý seq_len na CPU obvykle není potřeba.
    #[arg(long, default_value_t = false)]
    checkpoint: bool,

    /// Cesta pro uložení trénovaného Core Memory artefaktu (alpha.14).
    /// Pokud neuvedeno, training proběhne bez persistence — použij pro
    /// hyperparameter sweep, kde výsledek nepotřebuješ uchovat.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Volitelná textová poznámka k uloženému artefaktu (např.
    /// "law_pack 8 epoch, 1.5B"). Zapíše se do metadata.
    #[arg(long)]
    notes: Option<String>,

    /// Cesta k existujícímu `CoreMemoryArtifact` pro **resume tréninku**
    /// (alpha.15). Místo `randn_small` startu se Core Memory načte z disku
    /// a `into_stack` zkonstruuje čerstvé `Var`-y inicializované z saved
    /// init_states. Counter `training_steps` v output metadatech je
    /// kumulativní (saved + tento run).
    ///
    /// **AdamW state (alpha.16+):** Pokud vedle artefaktu existuje
    /// sourozenec `<resume-from>.optim.safetensors`, načte se i AdamW
    /// state (m, v moments + step_t) přes `OptimizerArtifact`. RN-008
    /// ukázalo, že to **nezabraňuje** Phase 2 overshoot v cross-domain
    /// resume — overshoot je dataset-driven. Pokud sourozenec chybí,
    /// proběhne soft resume.
    #[arg(long)]
    resume_from: Option<PathBuf>,

    /// **LR warmup (alpha.17, KI-008).** Lineární ramp `0 → learning_rate`
    /// přes prvních N optimizer stepů. Eliminuje Phase 2 overshoot
    /// (RN-002). Default 0 = žádný warmup (alpha.16 chování).
    /// Doporučeno: 50 pro tiny smoke runs, 5 % `total_steps` pro
    /// produkční tréninky. Per-run counter — resume run prochází
    /// warmupem znovu (záměrně).
    #[arg(long, default_value_t = 0)]
    warmup_steps: usize,

    /// **Cosine decay floor (alpha.17, KI-008).** Pokud > 0, LR po
    /// warmupu cosine-decayuje z `learning_rate` na `lr_min` přes
    /// zbývající steps. Default 0 = žádné decay (po warmupu konstantní LR).
    /// Doporučeno: 1e-5 pro produkční tréninky pro jemnější závěr.
    #[arg(long, default_value_t = 0.0)]
    lr_min: f64,
}

/// Argumenty pro `train-core-memory-multi` subkomand — multi-layer smoke
/// test s cross-entropy loss (Fáze 5 alpha.10+).
#[derive(ClapArgs, Debug)]
struct TrainCoreMemoryMultiArgs {
    /// Počet tokenů ve vstupu. Min 2 (cross-entropy potřebuje next-token
    /// target). Default 4 — konzervativní pro 6 GB VRAM na RTX 4050.
    /// Pro větší seq_len použij gradient accumulation (alpha.11+).
    #[arg(long, default_value_t = 4)]
    seq_len: usize,

    /// AdamW learning rate. Default 1e-3 — bezpečný pro smoke bring-up.
    /// Produkční training (alpha.11+) bude ladit přes LR sweep; RWKV
    /// doporučuje 1.0 pro State Tuning, ale až po warmup.
    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f64,

    /// Gradient clipping (max L2 norm, globálně napříč všemi vrstvami).
    /// Default 1.0 — standardní Mamba-2 recept, chrání před případnou
    /// gradient explozí pro edge-case vstupy.
    #[arg(long, default_value_t = 1.0)]
    grad_clip: f64,
}

fn parse_state_filter(s: &str) -> StateFilter {
    match s {
        "full" => StateFilter::full(),
        "core" | "core_memory" => StateFilter::core_memory(),
        "ssm" | "ssm_only" => StateFilter::ssm_only(),
        other => {
            eprintln!("Neznámý filtr '{}', používám 'full'", other);
            StateFilter::full()
        }
    }
}

/// Cesta k automaticky ukládané session.
fn default_session_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home)
        .join(".eleutheria")
        .join("last_session.safetensors")
}

/// Default cesta k trénované Core Memory pro auto-discovery.
fn default_core_memory_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home)
        .join(".eleutheria")
        .join("core_memory.safetensors")
}

/// Zajisti existenci adresáře pro session.
fn ensure_session_dir() -> Result<PathBuf> {
    let path = default_session_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(path)
}

/// Zajisti existenci adresáře pro daný path (vytvoří parent dirs).
fn ensure_parent_dir(path: &std::path::Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow!("nelze vytvořit {}: {}", parent.display(), e))?;
    }
    Ok(())
}

/// Při resume tréninku skládá notes z prior metadat + nové CLI poznámky.
/// `None + None = None`, jinak pomlčkou oddělené.
fn compose_notes(
    prior: Option<&eleutheria_core::CoreMemoryMeta>,
    new_notes: Option<&str>,
) -> Option<String> {
    let prior_notes = prior
        .and_then(|m| m.notes.as_deref())
        .filter(|s| !s.is_empty());
    let new_notes = new_notes.filter(|s| !s.is_empty());
    match (prior_notes, new_notes) {
        (None, None) => None,
        (Some(a), None) => Some(a.to_string()),
        (None, Some(b)) => Some(b.to_string()),
        (Some(a), Some(b)) => Some(format!("{a} | {b}")),
    }
}

/// Vyřeš zdroj Core Memory podle CLI argumentů. Vrací `None` pokud
/// uživatel auto-load explicitně vypnul nebo není kde brát.
fn resolve_core_memory_path(args: &Args) -> Option<PathBuf> {
    if args.no_core_memory {
        return None;
    }
    if let Some(p) = &args.core_memory {
        return Some(p.clone());
    }
    let default_p = default_core_memory_path();
    if default_p.exists() {
        Some(default_p)
    } else {
        None
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    // Režim inspekce session checkpointu
    if let Some(path) = &args.inspect_state {
        let meta = StateCheckpoint::inspect(path)?;
        print!("{meta}");
        return Ok(());
    }

    // Režim inspekce trénovaného Core Memory artefaktu
    if let Some(path) = &args.inspect_core_memory {
        let meta = CoreMemoryArtifact::inspect(path)?;
        print!("{meta}");
        return Ok(());
    }

    // Načti model
    println!("Eleutheria se probouzí...");
    let model_dir = match args.model.as_str() {
        "1.5b" => PathBuf::from("/home/lvx/Models/falcon-h1-1.5b-instruct"),
        "7b" => PathBuf::from("/home/lvx/Models/falcon-h1-7b-instruct"),
        other => PathBuf::from(other),
    };

    println!("Model: {} ({})", args.model, model_dir.display());
    println!("Device: {}\n", if args.cuda { "CUDA" } else { "CPU" });

    // Bench defaultně běží bez persony (čistý SSM signál). Opt-in přes
    // --with-persona. Ostatní módy (REPL, single-shot) načtou personu jako dřív.
    let bench_suppresses_persona = matches!(
        &args.command,
        Some(Command::BenchRetention(ba)) if !ba.with_persona
    );

    let persona_path = PathBuf::from(&args.persona);
    let persona_opt = if bench_suppresses_persona {
        tracing::info!(
            "bench mód: persona vypnuta pro čistý signál (použij --with-persona pro opt-in)"
        );
        None
    } else if persona_path.exists() {
        Some(persona_path.as_path())
    } else {
        tracing::warn!("Persona soubor nenalezen: {}", args.persona);
        None
    };

    let mut sofie = Sofie::load(&model_dir, args.cuda, persona_opt)?;

    // Auto-attach Core Memory pokud existuje a není opt-out. Training
    // subkomandy (smoke/multi/train) Core Memory nepotřebují — trénují
    // od nuly nebo z vlastního stacku — ale neškodí, když je připojená
    // (training stack injektuje Var-y nezávisle na Sofie::core_memory).
    if let Some(cm_path) = resolve_core_memory_path(&args) {
        match CoreMemoryArtifact::load(&cm_path) {
            Ok(art) => {
                println!("Core Memory: {}", cm_path.display());
                if let Some(meta) = Some(art.meta()).filter(|_| true) {
                    println!(
                        "  {} vrstev, training_steps={:?}, best_loss={:?}\n",
                        meta.num_layers, meta.training_steps, meta.best_loss
                    );
                }
                sofie.attach_core_memory(art)?;
            }
            Err(e) => {
                tracing::warn!(
                    "Core Memory na {} nelze načíst: {} — pokračuji bez Core Memory",
                    cm_path.display(),
                    e
                );
            }
        }
    } else if args.no_core_memory {
        tracing::info!("Core Memory auto-load vypnut (--no-core-memory)");
    }

    if let Some(cmd) = &args.command {
        match cmd {
            Command::BenchRetention(ba) => run_bench_retention(&sofie, &args, ba),
            Command::TrainCoreMemorySmoke(ta) => run_train_smoke(&sofie, ta),
            Command::TrainCoreMemoryMulti(ta) => run_train_multi(&sofie, ta),
            Command::TrainCoreMemory(ta) => run_train(&sofie, ta),
        }
    } else {
        match &args.prompt {
            Some(prompt) => run_single_shot(&sofie, &args, prompt),
            None => run_repl(&sofie, &args),
        }
    }
}

/// Core Memory autograd smoke test — ověří, že gradient teče od loss zpět
/// k trainable initial SSM state.
fn run_train_smoke(sofie: &Sofie, ta: &TrainCoreMemorySmokeArgs) -> Result<()> {
    println!(
        "\nCore Memory smoke test — seq_len={}, layer_idx={}, lr={}\n",
        ta.seq_len, ta.layer_idx, ta.learning_rate
    );

    if ta.sweep {
        return run_train_smoke_sweep(sofie, ta);
    }

    // Parse sub-layer cut bod (--cut-at-component). Default: Full (žádný sub-cut).
    let stop = match &ta.cut_at_component {
        Some(s) => LayerStop::from_str(s).map_err(|e| anyhow!(e))?,
        None => LayerStop::Full,
    };

    // Pokud je trace nebo cut-at-component aktivní, jdi přes unified `_component` API.
    // Jinak zachovej staré API (pro backward compatibility a lehčí overhead).
    let needs_component_path = ta.trace
        || ta.cut_at_component.is_some()
        || ta.grad_clip.is_some()
        || ta.cut_at_layer.is_some();

    let (result, trace_entries) = if needs_component_path {
        sofie.smoke_train_core_memory_component(
            ta.seq_len,
            ta.layer_idx,
            ta.learning_rate,
            ta.cut_at_layer,
            ta.grad_clip,
            stop,
            ta.trace,
        )?
    } else {
        (
            sofie.smoke_train_core_memory(ta.seq_len, ta.layer_idx, ta.learning_rate)?,
            None,
        )
    };

    println!("Výsledky:");
    println!("  layer_idx:              {}", result.layer_idx);
    println!("  seq_len:                {}", result.seq_len);
    println!("  loss:                   {:.6}", result.loss_value);
    println!(
        "  gradient L2 (pre-clip): {:.6e}",
        result.pre_clip_gradient_norm
    );
    println!("  gradient L2 norm:       {:.6e}", result.gradient_norm);
    println!(
        "  init_state |before|:    {:.6e}",
        result.init_state_norm_before
    );
    println!(
        "  init_state |after|:     {:.6e}",
        result.init_state_norm_after
    );
    println!(
        "  init_state delta:       {:.6e}",
        result.init_state_delta_norm
    );
    println!("  wall time:              {} ms", result.wall_time_ms);
    if ta.cut_at_component.is_some() {
        println!("  stop component:         {}", stop.label());
    }
    println!();

    // Forward tensor stats — pokud uživatel zapnul --trace.
    if let Some(entries) = trace_entries {
        println!("Forward trace ({} probes):", entries.len());
        print!("{}", trace::render_table(&entries));
        println!();
    }

    if result.passed() {
        println!("✓ PASS — autograd teče, gradient je non-zero, init_state se pohnul.");
        println!("  Fáze 5 state tuning workflow je feasibilní v Candle.");
        Ok(())
    } else {
        eprintln!("✗ FAIL — gradient je zanedbatelný nebo init_state se nehnul.");
        eprintln!(
            "  gradient_norm={:.3e}, delta_norm={:.3e} (prah 1e-8)",
            result.gradient_norm, result.init_state_delta_norm
        );
        Err(anyhow!("smoke test selhal — nutná investigace"))
    }
}

/// Diagnostický sweep — forward hidden norms + smoke přes cut range.
fn run_train_smoke_sweep(sofie: &Sofie, ta: &TrainCoreMemorySmokeArgs) -> Result<()> {
    println!(
        "\nCore Memory diagnostic sweep — seq_len={}, layer_idx={}, lr={}\n",
        ta.seq_len, ta.layer_idx, ta.learning_rate
    );

    // 1) Forward hidden norms — L2 norma aktivace po každé vrstvě
    println!("Forward hidden norms (fresh state, žádný trained Var):");
    let norms = sofie.measure_forward_hidden_norms(ta.seq_len)?;
    println!("  layer    |hidden|_2       finite?");
    println!("  -----    -------------    -------");
    for (i, norm) in norms.iter().enumerate() {
        let marker = if norm.is_finite() { "✓" } else { "✗" };
        println!("  {:>5}    {:>13.6e}    {}", i, norm, marker);
    }
    println!();

    // 2) Smoke sweep — pro fixní layer_idx přes všechny cut hodnoty
    println!("Smoke sweep (trainable Var, loss na aktivaci):");
    let results = sofie.smoke_sweep(ta.seq_len, ta.layer_idx, ta.learning_rate)?;
    println!(
        "  {:<20}  {:>13}  {:>13}  {:>13}  status",
        "variant", "loss", "grad L2", "delta"
    );
    println!(
        "  {:<20}  {:>13}  {:>13}  {:>13}  ------",
        "-------", "----", "-------", "-----"
    );
    for (desc, r) in &results {
        let status = if r.passed() {
            "PASS"
        } else if !r.gradient_norm.is_finite() {
            "NaN/Inf"
        } else {
            "underflow"
        };
        println!(
            "  {:<20}  {:>13.6e}  {:>13.6e}  {:>13.6e}  {}",
            desc, r.loss_value, r.gradient_norm, r.init_state_delta_norm, status
        );
    }
    println!();

    Ok(())
}

/// Core Memory training loop — produkční varianta.
/// Načte textový korpus, tokenizuje, chunkuje, trénuje přes epochs.
fn run_train(sofie: &Sofie, ta: &TrainCoreMemoryArgs) -> Result<()> {
    use eleutheria_core::CoreMemoryArtifact;
    use eleutheria_core::training::optim_io::{OptimizerArtifact, sibling_path};
    use eleutheria_core::training::{CoreMemoryStack, LrSchedule, TokenDataset, TrainingConfig};

    println!("\nCore Memory training — dataset: {}", ta.dataset.display());
    println!(
        "  epochs={}, batch_size={}, seq_len={}, grad_accum={}, lr={}, clip={}",
        ta.epochs, ta.batch_size, ta.seq_len, ta.grad_accum, ta.learning_rate, ta.grad_clip
    );

    // 1) Načti text
    let text = std::fs::read_to_string(&ta.dataset)
        .map_err(|e| anyhow!("chyba čtení {}: {}", ta.dataset.display(), e))?;
    println!("  korpus: {} bytes", text.len());

    // 2) Tokenize + dataset
    let dataset = TokenDataset::from_text(&text, sofie.tokenizer_ref(), ta.seq_len, ta.add_bos)
        .map_err(|e| anyhow!("dataset build: {e}"))?;
    println!(
        "  tokenů: {}, chunků: {} (seq_len {})",
        dataset.total_tokens(),
        dataset.num_chunks(),
        dataset.seq_len()
    );

    // 3) Vytvoř trainable Core Memory stack — buď randn_small nebo resume.
    //    Při resume akumulujeme původní training_steps a notes do meta
    //    output artefaktu, ať historie tréninku nezmizí. Pokud existuje
    //    sourozenec `<core>.optim.safetensors`, načteme i AdamW state
    //    (alpha.16, KI-007) — eliminuje overshoot fázi po resume (RN-006).
    let (stack, prior_meta, resume_optim) = if let Some(resume_path) = &ta.resume_from {
        println!("  resume z: {}", resume_path.display());
        let art = CoreMemoryArtifact::load(resume_path)
            .map_err(|e| anyhow!("CoreMemoryArtifact::load: {e}"))?;
        art.validate_config(sofie.config())
            .map_err(|e| anyhow!("artefakt nekompatibilní s modelem: {e}"))?;
        let prior = art.meta().clone();
        let stack = art
            .into_stack(sofie.config(), sofie.device_ref())
            .map_err(|e| anyhow!("into_stack: {e}"))?;
        println!(
            "  předchozí: {} steps, best_loss={:?}, timestamp={}",
            prior.training_steps.unwrap_or(0),
            prior.best_loss,
            prior.timestamp
        );

        // Auto-discovery sourozeneckého .optim.safetensors (alpha.16).
        let optim_path = sibling_path(resume_path);
        let resume_optim = if optim_path.exists() {
            match OptimizerArtifact::load(&optim_path) {
                Ok(art) => match art.validate_config(sofie.config()) {
                    Ok(()) => {
                        println!(
                            "  AdamW state: {} (step_t={})",
                            optim_path.display(),
                            art.meta().step_t,
                        );
                        Some(art)
                    }
                    Err(e) => {
                        eprintln!(
                            "  AdamW sourozenec na {} nekompatibilní s modelem: {} — pokračuji s prázdným Adamem",
                            optim_path.display(),
                            e
                        );
                        None
                    }
                },
                Err(e) => {
                    eprintln!(
                        "  AdamW sourozenec na {} nelze načíst: {} — pokračuji s prázdným Adamem",
                        optim_path.display(),
                        e
                    );
                    None
                }
            }
        } else {
            println!(
                "  AdamW state: prázdný (sourozenec {} neexistuje, soft resume)",
                optim_path.display()
            );
            None
        };
        println!();
        (stack, Some(prior), resume_optim)
    } else {
        let stack = CoreMemoryStack::randn_small(sofie.config(), sofie.device_ref())
            .map_err(|e| anyhow!("CoreMemoryStack: {e}"))?;
        println!("  CoreMemory: {} vrstev, randn init\n", stack.num_layers());
        (stack, None, None)
    };

    // 4) Trénuj
    let grad_clip = if ta.grad_clip > 0.0 {
        Some(ta.grad_clip)
    } else {
        None
    };
    // Pre-compute total optimizer steps pro LR schedule. Per-epoch:
    // micro-batches = ceil(num_chunks / batch_size), optimizer steps =
    // ceil(micro-batches / grad_accum). Při tail micro-batchu (méně než
    // grad_accum, viz train.rs:249) se přidá +1 step na konci epoch,
    // pokud něco zbývá. Konzervativně počítáme bez tailu — total_steps
    // je horní odhad pro warmup prvků; cosine decay si saharuje
    // clampem v lr_at_step.
    let micro_batches_per_epoch = dataset.num_chunks().div_ceil(ta.batch_size);
    let optimizer_steps_per_epoch = micro_batches_per_epoch.div_ceil(ta.grad_accum);
    let total_steps = ta.epochs * optimizer_steps_per_epoch;

    // LR schedule (alpha.17). None → konstantní LR (alpha.16 chování).
    let lr_schedule = if ta.warmup_steps > 0 || ta.lr_min > 0.0 {
        let s = if ta.lr_min > 0.0 {
            LrSchedule::warmup_cosine(ta.learning_rate, ta.warmup_steps, total_steps, ta.lr_min)
        } else {
            LrSchedule::warmup(ta.learning_rate, ta.warmup_steps)
        };
        println!(
            "  LR schedule: warmup={} stepů, total={} stepů, target={:.4e}, min={:.4e}, kind={:?}",
            ta.warmup_steps, total_steps, ta.learning_rate, ta.lr_min, s.kind,
        );
        Some(s)
    } else {
        None
    };

    let config = TrainingConfig {
        epochs: ta.epochs,
        batch_size: ta.batch_size,
        grad_accum_steps: ta.grad_accum,
        learning_rate: ta.learning_rate,
        grad_clip,
        shuffle_seed: ta.seed,
        log_every_n_steps: ta.log_every,
        checkpoint: ta.checkpoint,
        lr_schedule,
    };
    if ta.checkpoint {
        println!("  gradient checkpointing: ON (per-layer chunked backward)");
    }

    let (result, optimizer) =
        sofie.train_core_memory(&stack, &dataset, &config, resume_optim.as_ref())?;

    println!("\nVýsledky tréninku:");
    println!("  total steps:           {}", result.total_steps);
    println!("  total micro-batches:   {}", result.total_micro_batches);
    println!("  initial loss:          {:.4}", result.initial_loss);
    println!("  final loss:            {:.4}", result.final_loss);
    println!("  best loss:             {:.4}", result.best_loss);
    println!("  loss per epoch:");
    for (i, l) in result.loss_per_epoch.iter().enumerate() {
        println!("    epoch {}: {:.4}", i, l);
    }
    println!("  wall time:             {} ms", result.wall_time_ms);
    println!(
        "  expected random baseline: ln(vocab)≈{:.4}",
        (sofie.config().vocab_size as f64).ln()
    );
    println!();

    // Persistence — trained Core Memory na disk (alpha.14).
    // Při resume akumulujeme prior_meta.training_steps + this run; best_loss
    // bereme min z obou, final_loss z tohoto runu (nejaktuálnější stav).
    if let Some(out_path) = &ta.output {
        ensure_parent_dir(out_path)?;
        let final_loss = if result.final_loss.is_finite() {
            Some(result.final_loss)
        } else {
            None
        };
        let best_loss_run = if result.best_loss.is_finite() {
            Some(result.best_loss)
        } else {
            None
        };

        let prior_steps = prior_meta
            .as_ref()
            .and_then(|m| m.training_steps)
            .unwrap_or(0);
        let cumulative_steps = prior_steps + result.total_steps;
        let best_loss_combined =
            match (prior_meta.as_ref().and_then(|m| m.best_loss), best_loss_run) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
        let combined_notes = compose_notes(prior_meta.as_ref(), ta.notes.as_deref());

        let artifact = CoreMemoryArtifact::from_stack(
            &stack,
            sofie.config(),
            Some(cumulative_steps),
            best_loss_combined,
            final_loss,
            combined_notes,
        )
        .map_err(|e| anyhow!("CoreMemoryArtifact build: {e}"))?;
        artifact
            .save(out_path)
            .map_err(|e| anyhow!("CoreMemoryArtifact save: {e}"))?;
        println!(
            "Core Memory uložena: {} ({} vrstev, {} steps total, +{} this run, best_loss={:.4})",
            out_path.display(),
            stack.num_layers(),
            cumulative_steps,
            result.total_steps,
            best_loss_combined.unwrap_or(f64::NAN)
        );

        // AdamW state vedle Core Memory (alpha.16, KI-007). Sourozenec
        // <out_path>.optim.safetensors umožní příští `--resume-from` pokračovat
        // bez warmup overshoot fáze.
        let optim_path = sibling_path(out_path);
        let optim_art = OptimizerArtifact::from_optimizer(&optimizer, sofie.config())
            .map_err(|e| anyhow!("OptimizerArtifact build: {e}"))?;
        optim_art
            .save(&optim_path)
            .map_err(|e| anyhow!("OptimizerArtifact save: {e}"))?;
        println!(
            "AdamW state uložen: {} (step_t={})",
            optim_path.display(),
            optimizer.step_t()
        );
    } else {
        println!("(Core Memory neuložena — bez --output flagu zůstává pouze v paměti.)");
    }
    println!();

    if result.loss_decreased {
        println!("✓ Loss klesl — Core Memory se učí.");
        if result.final_loss < (sofie.config().vocab_size as f64).ln() {
            println!("  Navíc pod random baseline — signifikantní signál.");
        }
        Ok(())
    } else {
        eprintln!("✗ Loss neklesl — trénink se zasekl nebo je konfigurace špatná.");
        eprintln!(
            "  initial={:.4}, final={:.4}",
            result.initial_loss, result.final_loss
        );
        Err(anyhow!("training loss nedecreased"))
    }
}

/// Multi-layer Core Memory smoke test — trénuje init_state všech vrstev
/// najednou s cross-entropy loss na next-token prediction.
fn run_train_multi(sofie: &Sofie, ta: &TrainCoreMemoryMultiArgs) -> Result<()> {
    println!(
        "\nCore Memory multi-layer smoke — seq_len={}, lr={}, grad_clip={}\n",
        ta.seq_len, ta.learning_rate, ta.grad_clip
    );

    let grad_clip = if ta.grad_clip > 0.0 {
        Some(ta.grad_clip)
    } else {
        None
    };

    let result =
        sofie.smoke_train_core_memory_multilayer(ta.seq_len, ta.learning_rate, grad_clip)?;

    println!("Výsledky:");
    println!("  num_layers:              {}", result.num_layers);
    println!("  seq_len:                 {}", result.seq_len);
    println!("  loss (cross-entropy):    {:.6}", result.loss_value);
    println!(
        "  expected if random:      ln(vocab)≈{:.4}",
        (sofie.config().vocab_size as f64).ln()
    );
    println!(
        "  total gradient L2 (pre-clip): {:.6e}",
        result.pre_clip_total_gradient_norm
    );
    println!(
        "  total gradient L2 norm:       {:.6e}",
        result.total_gradient_norm
    );
    println!("  wall time:              {} ms", result.wall_time_ms);
    println!();

    // Per-layer tabulka
    println!("Per-layer gradient + init norms:");
    println!(
        "  {:>5}  {:>13}  {:>13}  {:>13}  {:>10}",
        "layer", "grad L2", "init |before|", "init |after|", "Δ init"
    );
    println!(
        "  {:>5}  {:>13}  {:>13}  {:>13}  {:>10}",
        "-----", "-------", "-------------", "------------", "------"
    );
    for i in 0..result.num_layers {
        let g = result.per_layer_gradient_norms[i];
        let b = result.per_layer_init_norms_before[i];
        let a = result.per_layer_init_norms_after[i];
        let delta = (a - b).abs();
        println!(
            "  {:>5}  {:>13.4e}  {:>13.4e}  {:>13.4e}  {:>10.4e}",
            i, g, b, a, delta
        );
    }
    println!();

    if result.passed() {
        println!("✓ PASS — autograd teče ke všem vrstvám, gradient distribuovaný.");
        println!("  Multi-layer state tuning funkční — alpha.10 milestone splněn.");
        Ok(())
    } else {
        eprintln!("✗ FAIL — gradient neprotekl ke dostatečnému počtu vrstev.");
        eprintln!(
            "  total_grad={:.3e}, non-trivial layers={}/{}",
            result.total_gradient_norm,
            result
                .per_layer_gradient_norms
                .iter()
                .filter(|&&n| n.is_finite() && n > 1e-10)
                .count(),
            result.num_layers
        );
        Err(anyhow!("multi-layer smoke test selhal"))
    }
}

/// Retention benchmark — harness pro měření SSM state retence.
fn run_bench_retention(sofie: &Sofie, args: &Args, ba: &BenchRetentionArgs) -> Result<()> {
    let variants: Vec<BenchVariant> = if ba.variant == "all" {
        BenchVariant::all().to_vec()
    } else {
        vec![BenchVariant::from_str(&ba.variant).map_err(|e| anyhow!(e))?]
    };

    let distances: Vec<usize> = match &ba.distances {
        Some(s) => s
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse::<usize>()
                    .map_err(|e| anyhow!("bad distance '{}': {}", s, e))
            })
            .collect::<Result<Vec<_>>>()?,
        None => DEFAULT_DISTANCES.to_vec(),
    };

    let variant_labels: Vec<&str> = variants.iter().map(|v| v.label()).collect();
    println!(
        "\nRetention benchmark — variants={:?}, distances={:?}, probes={}\n",
        variant_labels,
        distances,
        built_in_probes().len()
    );

    let report = RetentionBench::run(sofie, built_in_probes(), &distances, &variants, |r| {
        let mark = match r.outcome {
            eleutheria_core::bench::ProbeOutcome::Pass => "PASS",
            eleutheria_core::bench::ProbeOutcome::Fail => "FAIL",
        };
        println!(
            "  [{}] {} {} @ {} (actual {}): {}",
            mark,
            r.variant,
            r.probe_id,
            r.target_distance,
            r.actual_distance,
            truncate(&r.response, 100)
        );
    })?;

    let device_label = if args.cuda { "CUDA" } else { "CPU" };
    let report = report
        .with_model_name(&args.model)
        .with_device(device_label);
    let report = if let Some(n) = &ba.notes {
        report.with_notes(n.clone())
    } else {
        report
    };

    let base = PathBuf::from(&ba.output);
    let (json_path, md_path) = report.write_to(&base)?;
    println!("\nReport zapsán:");
    println!("  JSON:     {}", json_path.display());
    println!("  Markdown: {}", md_path.display());

    Ok(())
}

/// Single-shot mód — jeden prompt, jedna odpověď.
fn run_single_shot(sofie: &Sofie, args: &Args, prompt: &str) -> Result<()> {
    let initial_state = match &args.load_state {
        Some(path) => {
            println!("Načítám state z: {}", path.display());
            let (state, pos) = sofie.load_state(path)?;
            println!("  pozice: {} tokenů\n", pos);
            Some((state, pos))
        }
        None => None,
    };

    println!("Prompt: {}\n", prompt);
    println!("Sofie říká:");
    io::stdout().flush()?;

    let result = sofie.generate_streaming(
        prompt,
        args.max_tokens,
        args.temperature,
        initial_state,
        |_, text| {
            print!("{}", text);
            io::stdout().flush().unwrap();
            GenerateControl::Continue
        },
    )?;

    println!();

    if let Some(path) = &args.save_state {
        let filter = parse_state_filter(&args.state_filter);
        sofie.save_state(&result.state, result.position, path, filter)?;
        println!(
            "\nState uložen: {} (filtr: {}, pozice: {})",
            path.display(),
            filter.label(),
            result.position
        );
    }

    Ok(())
}

/// REPL mód — interaktivní konverzace se Sofií.
fn run_repl(sofie: &Sofie, args: &Args) -> Result<()> {
    // Resoluce zdroje session: --load-state > --resume > nová session
    let mut session: SofieSession = if let Some(path) = &args.load_state {
        println!("Načítám session z: {}", path.display());
        let s = sofie.resume_session(path)?;
        println!("  pozice: {} tokenů", s.position());
        s
    } else if args.resume {
        let path = default_session_path();
        if path.exists() {
            println!("Pokračuji v poslední session: {}", path.display());
            let s = sofie.resume_session(&path)?;
            println!("  pozice: {} tokenů", s.position());
            s
        } else {
            println!("Žádná předchozí session nenalezena, startuji novou.");
            sofie.new_session()?
        }
    } else {
        sofie.new_session()?
    };

    println!("Sofie je připravena. (q = konec, /save = uložit, /info = session info)\n");

    let stdin = io::stdin();
    let mut reader = stdin.lock();

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            break;
        }

        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        // Příkazy
        match input {
            "q" | "quit" | "exit" => break,
            _ if input.starts_with("/save") => {
                let path = input.strip_prefix("/save").unwrap().trim();
                let path = if path.is_empty() {
                    default_session_path().to_string_lossy().into_owned()
                } else {
                    path.to_string()
                };
                let filter = parse_state_filter(&args.state_filter);
                match sofie.save_state(session.state(), session.position(), path.as_ref(), filter) {
                    Ok(()) => println!(
                        "State uložen: {} (filtr: {}, pozice: {})\n",
                        path,
                        filter.label(),
                        session.position()
                    ),
                    Err(e) => eprintln!("Chyba při ukládání: {e}\n"),
                }
                continue;
            }
            "/info" => {
                let usage_pct = session.context_usage() * 100.0;
                let kv_mb = session.kv_cache_bytes() as f64 / (1024.0 * 1024.0);
                println!(
                    "Session: {} turnů, {} tokenů, běží od {}",
                    session.turn_count(),
                    session.position(),
                    session.started_at().format("%H:%M:%S")
                );
                println!(
                    "  kontext: {} / {} ({:.1}%), zbývá {} tokenů",
                    session.position(),
                    session.context_limit(),
                    usage_pct,
                    session.remaining_tokens()
                );
                println!("  KV cache: ~{:.1} MB (odhad)", kv_mb);
                for (i, pair) in session.history().chunks(2).enumerate() {
                    if pair.len() == 2 {
                        println!(
                            "  turn {}: \"{}\" → {} znaků odpovědi",
                            i + 1,
                            truncate(&pair[0].content, 40),
                            pair[1].content.len()
                        );
                    }
                }
                println!();
                continue;
            }
            _ if input.starts_with('/') => {
                eprintln!("Neznámý příkaz: {input}\n");
                continue;
            }
            _ => {}
        }

        // Pošli zprávu Sofii
        println!();
        let response = sofie.send_message(
            &mut session,
            input,
            args.max_tokens,
            args.temperature,
            |_, text| {
                print!("{}", text);
                io::stdout().flush().unwrap();
                GenerateControl::Continue
            },
        )?;
        let _ = response;

        println!("\n");
    }

    // Auto-save při ukončení
    if session.turn_count() > 0 {
        match ensure_session_dir() {
            Ok(path) => {
                match sofie.save_state(
                    session.state(),
                    session.position(),
                    &path,
                    StateFilter::full(),
                ) {
                    Ok(()) => println!(
                        "\nSession uložena: {} ({} turnů, {} tokenů)",
                        path.display(),
                        session.turn_count(),
                        session.position()
                    ),
                    Err(e) => eprintln!("\nChyba při auto-save: {e}"),
                }
            }
            Err(e) => eprintln!("\nNelze vytvořit session adresář: {e}"),
        }
    }

    Ok(())
}

/// Zkrátí text na max znaků.
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let boundary = s.floor_char_boundary(max);
        format!("{}...", &s[..boundary])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eleutheria_core::CoreMemoryMeta;

    fn meta_with_notes(notes: Option<&str>) -> CoreMemoryMeta {
        CoreMemoryMeta {
            eleutheria_version: "test".into(),
            format_version: 1,
            kind: "core_memory_trained".into(),
            num_layers: 24,
            n_heads: 24,
            headdim: 64,
            d_state: 256,
            dtype: "F32".into(),
            timestamp: "now".into(),
            training_steps: Some(100),
            best_loss: Some(2.0),
            final_loss: Some(2.5),
            notes: notes.map(String::from),
        }
    }

    #[test]
    fn compose_notes_returns_none_when_both_missing() {
        assert_eq!(compose_notes(None, None), None);
        let prior = meta_with_notes(None);
        assert_eq!(compose_notes(Some(&prior), None), None);
        assert_eq!(compose_notes(Some(&prior), Some("")), None);
    }

    #[test]
    fn compose_notes_uses_new_when_prior_missing() {
        assert_eq!(compose_notes(None, Some("epoch 2")), Some("epoch 2".into()));
        let prior = meta_with_notes(None);
        assert_eq!(
            compose_notes(Some(&prior), Some("epoch 2")),
            Some("epoch 2".into())
        );
    }

    #[test]
    fn compose_notes_keeps_prior_when_new_missing() {
        let prior = meta_with_notes(Some("epoch 1"));
        assert_eq!(compose_notes(Some(&prior), None), Some("epoch 1".into()));
        assert_eq!(
            compose_notes(Some(&prior), Some("")),
            Some("epoch 1".into())
        );
    }

    #[test]
    fn compose_notes_concatenates_with_pipe_separator() {
        let prior = meta_with_notes(Some("epoch 1"));
        assert_eq!(
            compose_notes(Some(&prior), Some("epoch 2 resume")),
            Some("epoch 1 | epoch 2 resume".into())
        );
    }
}
