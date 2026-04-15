# Plán — Eleutheria

> Poslední aktualizace: 2026-04-15

## Dokončeno

- [x] **Akt I: Hlas** (v0.1.0–v0.3.0) — Sofie umí mluvit
  - Inference engine, streaming output, prompt pipeline
- [x] **Research: Memory Architecture** — Deep Research 2026-04-12
  - Třívrstvá architektura: Core Memory (trénovaný state) + Session Memory
    (checkpointing) + Episodic Memory (echo embeddings + pgvector)
  - Výstup: `~/Atlas/Nexus/70-Eleutheria/Research/SSM state persistence...md`
- [x] **Akt II, Fáze 4: Session Memory** (v0.4.0) — Sofie si pamatuje v rámci session
  - v0.3.1: SSM state serializace (StateCheckpoint, 3 filtry, safetensors + metadata)
  - v0.3.2: Multi-turn REPL + inkrementální prefill (přístup B — delta tokeny)
  - v0.3.3: Token budget monitoring + context tracking
  - v0.3.4: Auto-save + `--resume`
  - Ověřeno na živém modelu (Falcon-H1-1.5B, RTX 4050, CUDA 13.2)

## Aktuální: Fáze 5 — Core Memory + Episodic Memory (v0.5.0)

Sofie si pamatuje přes sessions — ne jako databáze, ale jako zkušenost.

### Prerekvizita: Benchmark retence (v0.4.1–v0.4.3)
Behaviorální test SSM state retence — nutný pro informované rozhodnutí
o Core Memory designu. Rozděleno na tři patche:
- [x] **v0.4.1** — harness (modul `bench/`, 5 probes v EN, filler, CLI
  subkomand `bench-retention`, `Sofie::inject_turn` low-level API, JSON + MD export)
- [x] **v0.4.2** — `SsmOnly` a `Cold` varianty (`filter_session_to_ssm_only`,
  `BenchVariant::all()`, CLI `--variant all`)
- [x] **v0.4.3** — bugfix BUG-009 (UTF-8 safe streaming diff)
- [x] **v0.4.4** — `--with-persona` opt-in, bench defaultně bez persony
  (čistý model-level SSM measurement)
- [ ] **v0.4.5** — pilotní běh na Falcon-H1-1.5B, výsledky do
  `~/Atlas/Nexus/70-Eleutheria/Research/`

### Prerekvizita: Deep Research pro Core Memory
Před implementací state tuning:
- [ ] Ověřit backprop podporu v Candle (`candle-nn` gradient computation)
- [ ] RWKV State Tuning — detailní studium implementace
- [ ] State-offset Tuning (ACL 2025) — adaptace pro Falcon-H1

### Implementace Core Memory (v0.5.x)
- [ ] State tuning infrastruktura (backprop přes Candle)
- [ ] Core Memory training (identita, hodnoty, znalosti o Ondrovi)
- [ ] Core Memory loading (trénovaný initial state místo nulového)
- [ ] Evaluace: Core Memory state vs textový system prompt

### Implementace Episodic Memory (v0.5.x)
- [ ] Echo embeddings (self-retrieval přes Falcon-H1)
- [ ] PostgreSQL + pgvector na Mnémosyné
- [ ] MemoryInjection stage — retrieval + injection

## Příští kroky

- [ ] **Fáze 6** — Vnitřní monolog + Konsolidace (v0.6.0)
  - Deep Research: Mamba-CL, Sleep Replay Consolidation, Titans
- [ ] **Fáze 7** — Ruce (v0.7.0) — tools, soubory, notifikace
- [ ] **Fáze 8** — Rozhraní (v0.8.0) — Axum API server
- [ ] **Fáze 9** — Domov (v0.9.0) — 7B model, Gaia deploy

## Technické poznámky

- **CUDA 13.2 workaround**: `.cargo/config.toml` — `CUDARC_CUDA_VERSION=13010` (SOL-008)
- **Safetensors metadata**: přímá závislost na `safetensors` crate, ne Candle wrapper (SOL-007)
- **Přístup B (inkrementální prefill)**: Turn 1 = full pipeline, Turn 2+ = delta
  (`<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n`).
  Pipeline ConversationContext stage je obsoletní pro injekci historie — stav JE kontext.
- **Backprop v Candle**: nutné ověřit pro state tuning (Fáze 5)
- Research: `~/Atlas/Nexus/70-Eleutheria/Research/`
