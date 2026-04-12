//! Eleutheria CLI — Sofie's local mind.
use anyhow::Result;
use clap::Parser;
use eleutheria_core::{GenerateControl, Sofie, SofieSession, StateCheckpoint, StateFilter};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

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

    let persona_path = PathBuf::from(&args.persona);
    let persona_opt = if persona_path.exists() {
        Some(persona_path.as_path())
    } else {
        tracing::warn!("Persona soubor nenalezen: {}", args.persona);
        None
    };

    let sofie = Sofie::load(&model_dir, args.cuda, persona_opt)?;

    match &args.prompt {
        Some(prompt) => run_single_shot(&sofie, &args, prompt),
        None => run_repl(&sofie, &args),
    }
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
    let mut session: SofieSession = match &args.load_state {
        Some(path) => {
            println!("Načítám session z: {}", path.display());
            let s = sofie.resume_session(path)?;
            println!("  pozice: {} tokenů", s.position());
            s
        }
        None => sofie.new_session()?,
    };

    println!("Sofie je připravena. (q = konec, /save = uložit, /info = session info)\n");

    let stdin = io::stdin();
    let mut reader = stdin.lock();

    loop {
        // Prompt
        print!("> ");
        io::stdout().flush()?;

        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            // EOF
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
                    "/tmp/eleutheria_session.safetensors"
                } else {
                    path
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
        let _ = response; // výstup je už streamovaný

        println!("\n");
    }

    // Nabídni uložení při odchodu
    if session.turn_count() > 0 {
        println!(
            "\nSession ukončena ({} turnů, {} tokenů).",
            session.turn_count(),
            session.position()
        );
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
