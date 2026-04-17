//! Dataset pro Core Memory training.
//!
//! Jednoduchý token-level dataset: vstup = textový korpus, výstup =
//! iterator batches `Tensor[batch, seq_len]`. Chunking s overlap 0
//! (každý token je v maximálně jednom chunku), shuffle přes chunk
//! indexy, ne uvnitř chunků (zachovává lokální soudržnost).
//!
//! **Design rozhodnutí:**
//! - **Žádný streaming** — celý korpus se tokenizuje najednou do
//!   `Vec<u32>` v RAM. Pro Sofie první production run je korpus
//!   ~100k tokenů (50-Sofie), pohodlně se vejde. Pokud později
//!   korpus vyroste, přepneme na streaming (alpha.20+).
//! - **Žádný padding** — drop poslední chunk, pokud je kratší než
//!   `seq_len`. Jednoduchost > coverage několika málo posledních
//!   tokenů.
//! - **Shuffle per-epoch** — deterministický s seed=epoch_idx,
//!   reprodukovatelné běhy.
//! - **BOS token** — pokud tokenizer vrací BOS automaticky (Falcon-H1:
//!   ano, `add_special_tokens=true`), každý chunk začíná BOS po
//!   tokenizaci celého korpusu. Pro čisté trénovací chování bez duplikace
//!   BOS uprostřed textu to necháme na callera (dá buď `add_bos=true`
//!   s krátkými texty, nebo `false` s jedním dlouhým).

use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

/// Tokenizovaný korpus + logika pro batching.
pub struct TokenDataset {
    /// Celý korpus jako token IDs.
    tokens: Vec<u32>,
    /// Délka sekvence v jednom chunku.
    seq_len: usize,
    /// Počet kompletních chunků (floor(tokens.len() / seq_len)).
    num_chunks: usize,
}

impl TokenDataset {
    /// Vytvoří dataset z textu. Tokenizuje celý text, vyhodí kratší
    /// poslední chunk.
    ///
    /// - `text`: zdrojový korpus
    /// - `tokenizer`: načtený Falcon-H1 tokenizer
    /// - `seq_len`: délka jedné trénovací sekvence
    /// - `add_bos`: true = přidat BOS na začátek (standardní pro první
    ///   turn); false = čistý raw tokenize (pro kontinuální korpus)
    pub fn from_text(
        text: &str,
        tokenizer: &Tokenizer,
        seq_len: usize,
        add_bos: bool,
    ) -> Result<Self> {
        if seq_len < 2 {
            return Err(anyhow!(
                "seq_len musí být >= 2 (cross-entropy potřebuje next-token target)"
            ));
        }
        let encoding = tokenizer
            .encode(text, add_bos)
            .map_err(|e| anyhow!("Tokenizer error: {e}"))?;
        let tokens: Vec<u32> = encoding.get_ids().to_vec();
        if tokens.len() < seq_len {
            return Err(anyhow!(
                "korpus má {} tokenů, seq_len={} — příliš krátké pro jeden chunk",
                tokens.len(),
                seq_len
            ));
        }
        let num_chunks = tokens.len() / seq_len;
        Ok(Self {
            tokens,
            seq_len,
            num_chunks,
        })
    }

    /// Počet chunků (trénovacích sekvencí) v datasetu.
    pub fn num_chunks(&self) -> usize {
        self.num_chunks
    }

    /// Celkový počet tokenů (včetně těch, co spadly mimo poslední chunk).
    pub fn total_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Seq_len používaný při batching.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Vytvoří shuffled pořadí chunk indexů pro jednu epoch.
    /// Deterministické pro daný `seed` — reprodukovatelný trénink.
    fn epoch_order(&self, seed: u64) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.num_chunks).collect();
        // Jednoduchý deterministic Fisher-Yates s xorshift PRNG.
        let mut rng = SimpleRng::new(seed);
        for i in (1..order.len()).rev() {
            let j = rng.next_u64() as usize % (i + 1);
            order.swap(i, j);
        }
        order
    }

    /// Iterátor přes batches pro jednu epoch. Vrací tensory
    /// `[batch_size, seq_len]` na `device`, dtype `U32`.
    ///
    /// `seed` — kontroluje shuffle per epoch. Obvyklá konvence:
    /// `seed = epoch_idx as u64`.
    ///
    /// Poslední batch může být **kratší než batch_size**, pokud
    /// `num_chunks % batch_size != 0`. Caller musí pracovat s
    /// skutečnou první dimenzí tensoru, ne s deklarovaným batch_size.
    pub fn iter_batches(
        &self,
        batch_size: usize,
        device: &Device,
        seed: u64,
    ) -> Result<Vec<Tensor>> {
        if batch_size == 0 {
            return Err(anyhow!("batch_size musí být > 0"));
        }
        let order = self.epoch_order(seed);
        let mut batches = Vec::with_capacity(self.num_chunks.div_ceil(batch_size));

        for chunk_of_chunks in order.chunks(batch_size) {
            let mut batch_data: Vec<u32> = Vec::with_capacity(chunk_of_chunks.len() * self.seq_len);
            for &chunk_idx in chunk_of_chunks {
                let start = chunk_idx * self.seq_len;
                let end = start + self.seq_len;
                batch_data.extend_from_slice(&self.tokens[start..end]);
            }
            let tensor =
                Tensor::from_vec(batch_data, (chunk_of_chunks.len(), self.seq_len), device)?;
            batches.push(tensor);
        }
        Ok(batches)
    }
}

// ---------------------------------------------------------------------------
// Jednoduchý deterministic RNG (xorshift64) — nechceme externí dep
// ---------------------------------------------------------------------------

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        // Zabraň degenerativnímu seed=0 (xorshift by dával samé nuly)
        let state = if seed == 0 {
            0xDEAD_BEEF_CAFE_BABE
        } else {
            seed
        };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::Tokenizer;

    /// Načte reálný Falcon-H1 tokenizer z dev modelu. Test se přeskočí,
    /// pokud není dostupný (CI bez modelu).
    fn load_tokenizer() -> Option<Tokenizer> {
        let path =
            std::path::PathBuf::from("/home/lvx/Models/falcon-h1-1.5b-instruct/tokenizer.json");
        if !path.exists() {
            return None;
        }
        Tokenizer::from_file(path).ok()
    }

    #[test]
    fn from_text_produces_expected_chunk_count() {
        let Some(tok) = load_tokenizer() else {
            return; // skip pokud model není
        };
        // Malý text — "Sofie píše kód v rustu." opakované 20×
        let text = "Sofie píše kód v rustu. ".repeat(20);
        let ds = TokenDataset::from_text(&text, &tok, 8, false).unwrap();
        assert!(ds.num_chunks() > 0);
        assert_eq!(ds.seq_len(), 8);
        // total_tokens / seq_len (floor) = num_chunks
        assert_eq!(ds.total_tokens() / 8, ds.num_chunks());
    }

    #[test]
    fn from_text_rejects_seq_len_less_than_2() {
        let Some(tok) = load_tokenizer() else {
            return;
        };
        let text = "test text s dostatečnou délkou pro tokenizaci";
        assert!(TokenDataset::from_text(text, &tok, 1, false).is_err());
    }

    #[test]
    fn from_text_rejects_too_short_corpus() {
        let Some(tok) = load_tokenizer() else {
            return;
        };
        // 1 slovo, cílíme na seq_len 1000 — pass-through test
        let text = "krátké";
        assert!(TokenDataset::from_text(text, &tok, 1000, false).is_err());
    }

    #[test]
    fn iter_batches_covers_all_chunks_exactly_once() {
        let Some(tok) = load_tokenizer() else {
            return;
        };
        let text = "Sofie píše kód v rustu. ".repeat(50);
        let ds = TokenDataset::from_text(&text, &tok, 8, false).unwrap();
        let batches = ds.iter_batches(4, &Device::Cpu, 42).unwrap();

        // Součet batch dimenzí musí být roven num_chunks
        let total_rows: usize = batches.iter().map(|b| b.dim(0).unwrap()).sum();
        assert_eq!(total_rows, ds.num_chunks());

        // Každý batch má seq_len jako druhou dim
        for batch in &batches {
            assert_eq!(batch.dim(1).unwrap(), 8);
        }
    }

    #[test]
    fn iter_batches_is_deterministic_for_same_seed() {
        let Some(tok) = load_tokenizer() else {
            return;
        };
        let text = "Sofie píše kód v rustu. ".repeat(50);
        let ds = TokenDataset::from_text(&text, &tok, 8, false).unwrap();

        let a = ds.iter_batches(4, &Device::Cpu, 123).unwrap();
        let b = ds.iter_batches(4, &Device::Cpu, 123).unwrap();
        assert_eq!(a.len(), b.len());
        for (ba, bb) in a.iter().zip(b.iter()) {
            let av: Vec<u32> = ba.flatten_all().unwrap().to_vec1().unwrap();
            let bv: Vec<u32> = bb.flatten_all().unwrap().to_vec1().unwrap();
            assert_eq!(av, bv);
        }
    }

    #[test]
    fn iter_batches_differs_for_different_seeds() {
        let Some(tok) = load_tokenizer() else {
            return;
        };
        let text = "Sofie píše kód v rustu. ".repeat(50);
        let ds = TokenDataset::from_text(&text, &tok, 8, false).unwrap();

        let a = ds.iter_batches(4, &Device::Cpu, 1).unwrap();
        let b = ds.iter_batches(4, &Device::Cpu, 2).unwrap();

        // Alespoň jeden batch se liší (extrémně nepravděpodobné u náhodné
        // permutace chunkCount > ~3 dát stejné pořadí se dvěma různými seedy)
        let av: Vec<u32> = a[0].flatten_all().unwrap().to_vec1().unwrap();
        let bv: Vec<u32> = b[0].flatten_all().unwrap().to_vec1().unwrap();
        assert_ne!(av, bv);
    }

    #[test]
    fn simple_rng_is_deterministic() {
        let mut a = SimpleRng::new(42);
        let mut b = SimpleRng::new(42);
        for _ in 0..10 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn simple_rng_handles_zero_seed() {
        let mut rng = SimpleRng::new(0);
        // Nesmí degenerovat na samé nuly
        let vals: Vec<u64> = (0..5).map(|_| rng.next_u64()).collect();
        assert!(vals.iter().all(|&v| v != 0));
    }
}
