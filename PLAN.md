# Plán — Eleutheria

> Poslední aktualizace: 2026-04-17

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

### v0.5.0-alpha.1 ✅ — autograd bring-up
- [x] `CoreMemory` struct (Var pro init_state jedné vrstvy)
- [x] `Sofie::smoke_train_core_memory` — forward + backward + AdamW step
- [x] CLI subkomand `train-core-memory-smoke`
- [x] Ověřeno: `forward_prefill` je už sekvenční scan (ne chunked), autograd
  teoreticky protéká bez úprav

### v0.5.0-alpha.2–alpha.9 — BUG-010 diagnostika + fix ✅
- [x] alpha.2 — sequential scan bring-up
- [x] alpha.3 — single-element loss pro stabilní gradient
- [x] alpha.4 — `forward_up_to_layer` + binary search cut-at-layer
- [x] alpha.5 — diagnostic sweep + research backend
- [x] alpha.6 — gradient clipping (clip_grad_norm, μP multipliery ověřeny)
- [x] alpha.7 — minimal reproduction (`training/repro.rs`), stable softplus
- [x] alpha.8 — instrumentovaný forward (trace sink + cut-at-component).
      Diagnostika potvrdila: SSM branch backward je viník, attention OK
- [x] **alpha.9 — BUG-010 vyřešen.** Lokální `silu` backward exploduje
      přes `recip(Inf)` chain pro extrémní |x|. Fix: delegace na
      `candle_nn::ops::silu` (native stable kernel). L22 cut=23 grad
      NaN → 9.80; cut=full NaN → 1.74

### Fáze 5 design decisions (2026-04-17)
- **Rozsah:** trénovat **všech 24 Mamba-2 vrstev** najednou, ne selektivně
  (storage triviální ~75 MB, nevíme co Mamba vrstvy kódují). Selektivita
  je alpha.14+ optimalizace.
- **Dataset přístup:** **thought engineering**, ne dataset engineering.
  Core Memory trénujeme na **distilaci metody myšlení**, ne raw korpusu.
  Hierarchie Core Memory / Episodic Memory / Tools rozděluje kompetence
  (metoda / fakta / akce).
- **Zdroje training korpusu:**
  1. `~/Atlas/Nexus/50-Sofie/` — 78k slov (Bootstrap, Identity, Memory,
     Sessions, Context). Core identity + autentický hlas.
  2. Programovací distillate — reasoning chains ze SOLUTIONS.md napříč
     KQS repos. Sofie dělá.
  3. Právní distillate — reasoning chains z 20–50 reprezentativních
     judikátů NS (Ondra vybere z korpusu 1999–2026).
  4. Archetypální vzory — explicitní meta-pravidla, krátké a husté.
- **Raw judikatura NEPATŘÍ do Core Memory** (objem + fakta ≠ metoda +
  zastarávání). Půjde do Episodic Memory (v0.5.1 / v0.6).

### v0.5.0-alpha.10 (next) — multi-layer + cross-entropy
- [ ] `CoreMemory::all_layers(config, device)` → `Vec<Var>` pro všech 24 vrstev
- [ ] `inject_into_state(&mut ModelState)` — aplikuje všechny init_states
- [ ] Cross-entropy loss na next-token prediction (nahradí single-element)
- [ ] Smoke test multi-layer: 1 forward + backward, všechny Vars dostanou gradient

### v0.5.0-alpha.11 — training loop + dataset loader
- [ ] Dataset struct (tokenize + chunk), ChatML wrapping
- [ ] Training loop (epoch × batch), gradient accumulation (VRAM-friendly)
- [ ] AdamW betas `(0.9, 0.95)`, cosine/WSD schedule, grad clip 1.0
- [ ] LR sweep (RWKV doporučuje 1.0 pro State Tuning, naše smoke měla 1e-3)

### v0.5.0-alpha.12 — Save/Load trained Core Memory
- [ ] Rozšíření `StateCheckpoint` (filter `core_memory` už existuje)
- [ ] Auto-load v `Sofie::load` pokud existuje `core_memory.safetensors`
- [ ] Resume training (init_states + optimizer state + step_idx)

### v0.5.0-alpha.13 — Sofie identity dataset + full distillation
- [ ] Dataset composer — váhová mix (identity core / Sessions / distillate / context)
- [ ] Programovací distillation scale-up (SOLUTIONS.md napříč KQS)
- [ ] Právní distillation scale-up (Ondra + Claude Opus z 20–50 judikátů)
- [ ] Archetypální vzory sepisovat průběžně

### v0.5.0 — Production training run + validace
- [ ] Full training na 1.5B s kompletním korpusem
- [ ] **Validace: re-run retention benchmark — SsmOnly pass-rate musí
      vyskočit z 0 % na měřitelné číslo** (kritický bod důkazu)
- [ ] Save trained state jako `sofie_core_memory.safetensors`
- [ ] Research writeup do `~/Atlas/Nexus/70-Eleutheria/Research/`

### Paralelní track (dataset prep, mimo alpha cycle)
- [ ] **Mini pilot (teď):** 3 reasoning chains ze SOLUTIONS.md jako format
      experiment — ukázka šablony pro Ondru před výběrem judikátů
- [ ] Ondra vybírá 20–50 reprezentativních judikátů NS
- [ ] Schema pro Episodic Memory raw judikatura (v0.5.1 prep)

### v0.5.0 — Core Memory production
- [ ] Dataset pro Sofie identity + Bootstrap + Ondra context
- [ ] Training na 1.5B (pokud alpha.2 stabilní) nebo 7B (pokud lze fit)
- [ ] Validation: re-run retention benchmark → SsmOnly pass-rate musí
  vyskočit (ze 0 % na měřitelné číslo, pokud trained state funguje)

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
- [x] **v0.4.5** — pilotní běh dokončen na Falcon-H1-1.5B, research
  dokument `SSM_retention_findings_2026-04-15.md` v Nexusu, zpřísněn
  `preference_linh` matcher. **Prerekvizita Fáze 5 uzavřena.**

### Empirické nálezy z pilotního běhu (2026-04-16)
- **Full** 100 % do 500 tok, 80 % @ 1 k, 40 % @ 2 k — graceful degradation
- **SsmOnly** **0 % na všech vzdálenostech** — zachycený stav nenese fakta
- **Cold** 20 % = jeden false positive (opraveno)
- **Architektonický závěr:** Core Memory MUSÍ být trénovaný, ne captured

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
