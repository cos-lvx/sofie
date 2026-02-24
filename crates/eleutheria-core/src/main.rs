//! Eleutheria CLI . testovací rozhraní pro Sofii
use anyhow::Result;
use clap::Parser;
use eleutheria_core::{Sofie, ModelSpec};


/// Eleuitheria - Sofie's local mind
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
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

    /// Který model použít (130m, 1.4b)
    #[arg(short, long, default_value = "130m")]
    model: String,
}

fn main() -> Result<()> {
    // Inicializace logování
    tracing_subscriber::fmt::init();

    // Parsování argumentů
    let args = Args::parse();

    // Výběr modelu

    let spec = match args.model.as_str() {
        "130m" => ModelSpec::mamba_130m(),
        "1.4b" => ModelSpec::mamba_1_4b(),
        other => {
            eprintln!("Neznámý model: {}. Použik '130m' nebo '1.4b'", other);
            std::process::exit(1);
        }
    };

    // Načtení Sofie
    println!("Eleuitheria se probouzí...\n");
    let mut sofie = Sofie::load(spec, args.cuda)?;

    // Generování
    println!("Prompt: {}\n", args.prompt);
    let output = sofie.generate(&args.prompt, args.max_tokens, args.temperature)?;

    println!("Sofie říká:\n{}", output);

    Ok(())

}