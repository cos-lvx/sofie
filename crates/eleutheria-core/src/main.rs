//! Eleutheria CLI — Sofie's local mind.
use anyhow::{Result, anyhow};
use clap::{Args as ClapArgs, Parser, Subcommand};
use eleutheria_core::bench::{
    BenchVariant, RetentionBench, built_in_probes, harness::DEFAULT_DISTANCES,
};
use eleutheria_core::{GenerateControl, Sofie, SofieSession, StateCheckpoint, StateFilter};
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

    /// Pokračuj v poslední session (~/.eleutheria/last_session.safetensors)
    #[arg(long)]
    resume: bool,

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

/// Zajisti existenci adresáře pro session.
fn ensure_session_dir() -> Result<PathBuf> {
    let path = default_session_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(path)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    // Režim inspekce
    if let Some(path) = &args.inspect_state {
        let meta = StateCheckpoint::inspect(path)?;
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

    let sofie = Sofie::load(&model_dir, args.cuda, persona_opt)?;

    if let Some(cmd) = &args.command {
        match cmd {
            Command::BenchRetention(ba) => run_bench_retention(&sofie, &args, ba),
            Command::TrainCoreMemorySmoke(ta) => run_train_smoke(&sofie, ta),
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

    let result = match ta.cut_at_layer {
        None => sofie.smoke_train_core_memory(ta.seq_len, ta.layer_idx, ta.learning_rate)?,
        Some(cut) => {
            sofie.smoke_train_core_memory_cut(ta.seq_len, ta.layer_idx, ta.learning_rate, cut)?
        }
    };

    println!("Výsledky:");
    println!("  layer_idx:              {}", result.layer_idx);
    println!("  seq_len:                {}", result.seq_len);
    println!("  loss:                   {:.6}", result.loss_value);
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
    println!();

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
