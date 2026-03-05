//! Eleutheria CLI — Sofie's local mind.
use anyhow::Result;
use clap::Parser;
use eleutheria_core::{GenerateControl, Sofie};
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
    prompt: String,

    /// Maximální počet tokenů k vygenerování
    #[arg(short = 'n', long, default_value_t = 100)]
    max_tokens: usize,

    /// Temperature (0.0 = deterministické, 1.0 = náhodné)
    #[arg(short, long, default_value_t = 0.8)]
    temperature: f64,

    /// Použít CUDA (GPU)
    #[arg(long)]
    cuda: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    println!("Eleutheria se probouzí...");
    let model_dir = match args.model.as_str() {
        "1.5b" => PathBuf::from("/home/lvx/Models/falcon-h1-1.5b-instruct"),
        "7b" => PathBuf::from("/home/lvx/Models/falcon-h1-7b-instruct"),
        other => PathBuf::from(other),
    };

    println!("Model: {} ({})", args.model, model_dir.display());
    println!("Device: {}\n", if args.cuda { "CUDA" } else { "CPU" });

    let sofie = Sofie::load(&model_dir, args.cuda)?;

    println!("Prompt: {}\n", args.prompt);
    print!("Sofie říká:\n");
    io::stdout().flush()?;

    sofie.generate_streaming(&args.prompt, args.max_tokens, args.temperature, |_, text| {
        print!("{}", text);
        io::stdout().flush().unwrap();
        GenerateControl::Continue
    })?;

    println!();

    Ok(())
}