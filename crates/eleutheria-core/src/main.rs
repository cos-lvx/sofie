//! Eleutheria CLI — Sofie's local mind.
use anyhow::Result;
use clap::Parser;
use eleutheria_core::{GenerateControl, Sofie, StateCheckpoint, StateFilter};
use std::io::{self, Write};
use std::path::PathBuf;

/// Eleutheria — lokální inference engine pro Falcon-H1.
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Model: "1.5b", "7b", nebo přímá cesta k adresáři
    #[arg(short, long, default_value = "1.5b")]
    model: String,

    /// Prompt pro generování
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

    /// Načíst state z checkpointu před generováním
    #[arg(long)]
    load_state: Option<PathBuf>,

    /// Uložit state po generování
    #[arg(long)]
    save_state: Option<PathBuf>,

    /// Filtr pro uložení stavu: full, core, ssm
    #[arg(long, default_value = "full")]
    state_filter: String,

    /// Inspekce checkpointu — vypiš metadata a skonči
    #[arg(long)]
    inspect_state: Option<PathBuf>,
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

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // Režim inspekce — nevyžaduje prompt ani model
    if let Some(path) = &args.inspect_state {
        let meta = StateCheckpoint::inspect(path)?;
        print!("{meta}");
        return Ok(());
    }

    // Prompt je povinný pro generování
    let prompt = args
        .prompt
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--prompt je povinný pro generování"))?;

    println!("Eleutheria se probouzí...");
    let model_dir = match args.model.as_str() {
        "1.5b" => PathBuf::from("/home/lvx/Models/falcon-h1-1.5b-instruct"),
        "7b" => PathBuf::from("/home/lvx/Models/falcon-h1-7b-instruct"),
        other => PathBuf::from(other),
    };

    println!("Model: {} ({})", args.model, model_dir.display());
    println!("Device: {}\n", if args.cuda { "CUDA" } else { "CPU" });

    let persona_path = PathBuf::from(&args.persona);
    let persona_opt = if persona_path.exists() {
        Some(persona_path.as_path())
    } else {
        tracing::warn!("Persona soubor nenalezen: {}", args.persona);
        None
    };

    let sofie = Sofie::load(&model_dir, args.cuda, persona_opt)?;

    // Načtení state z checkpointu
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

    // Uložení state
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
