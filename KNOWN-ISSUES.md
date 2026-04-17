# Known Issues — Eleutheria

Známé problémy, limitace a vědomá technická rozhodnutí.

Formát: `KI-NNN` s fází, dopadem, kontextem a plánovaným řešením.

---

## Aktivní

### KI-001 — Hardcoded cesty k modelům v CLI

- **Fáze:** 1
- **Dopad:** Nízký — pouze dev convenience
- **Kontext:** `main.rs` obsahuje `/home/lvx/Models/falcon-h1-{1.5b,7b}-instruct`
- **Řešení:** Konfigurovatelné přes config soubor nebo env variable (plánováno pro v0.8.0)

### KI-002 — Placeholder stages v prompt pipeline

- **Fáze:** 3
- **Dopad:** Střední — ConversationContext, MemoryInjection, QualityGate jsou no-op
- **Kontext:** Vědomé rozhodnutí — pipeline architektura připravena, implementace postupně
- **Řešení:** ConversationContext v0.4.0, MemoryInjection v0.5.0, QualityGate v0.6.0

### KI-003 — InputClassifier je statický

- **Fáze:** 3
- **Dopad:** Nízký — defaultně Freeform intent
- **Kontext:** Plná klasifikace vyžaduje buď heuristiky nebo druhý model pass
- **Řešení:** Heuristická klasifikace v0.4.0, případně ML-based později

### KI-004 — CUDA 13.2 workaround v .cargo/config.toml

- **Fáze:** 1
- **Dopad:** Nízký — pouze build na Arch Linux
- **Kontext:** `cudarc` 0.18.2 nepodporuje CUDA 13.2. Workaround přes
  `CUDARC_CUDA_VERSION=13010`. Odebrat až cudarc přidá 13.2 podporu.
- **Řešení:** Sledovat cudarc releases, případně aktualizovat Candle (viz SOL-008)

### KI-005 — CUDA OOM pro multi-layer backward na 6 GB VRAM

- **Fáze:** 5 (Core Memory training)
- **Dopad:** Střední — blokuje GPU training na RTX 4050. CPU F32
  fallback funguje, ale pomalý (KI-006).
- **Kontext:** Alpha.10 `train-core-memory-multi --cuda --seq-len 2`
  selhává s `CUDA_ERROR_OUT_OF_MEMORY`. Forward sám prochází (alpha.8
  potvrdila), ale plný backward graph přes 24 vrstev × 65537 vocab se
  nevejde — jen LM head matmul s grad váží ~768 MB.
- **Řešení:** Alpha.12 — gradient checkpointing (recompute activations
  per layer chunk místo držet v grafu). Alternativa: Gaia deploy s větší
  VRAM (≥24 GB pro pohodlný seq_len na 1.5B/7B).

### KI-006 — Training CPU F32 je 48 s/step na 1.5B

- **Fáze:** 5 (Core Memory training)
- **Dopad:** Střední — blokuje větší runy. Smoke a mini-korpus OK
  (minuty), full korpus 50-Sofie (~100k tokenů, seq_len 64–128) by
  trval dny.
- **Kontext:** Alpha.11 `train-core-memory` na 1.5B CPU F32, `seq_len=8
  batch=1 grad_accum=2`: ~48 s per optimizer step. Full forward + backward
  přes 24 vrstev je compute-bound. RAM ~18 GB.
- **Řešení:** Vyřeší se společně s KI-005 — gradient checkpointing
  odblokuje CUDA, rychlost skočí řádově. Alternativa: 7B training na
  Gaia pro produkci, 1.5B CPU jen pro dev cyklus (smoke tests).

---

## Vyřešené

_(žádné zatím — všechny dosavadní bugy řešeny inline, viz BUGS.md)_
