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

### KI-005 — CUDA OOM pro multi-layer backward na 6 GB VRAM (částečně)

- **Fáze:** 5 (Core Memory training)
- **Dopad:** Střední — RTX 4050 6 GB stále neumí trénovat 1.5B
  multi-layer ani s alpha.12 per-layer checkpointing. CPU funguje
  pohodlně po KI-006 fixu.
- **Kontext:** Alpha.12 zavedla per-layer chunked gradient checkpointing
  (`training/checkpoint.rs`). Inter-layer memory snížena z 24× → 1×,
  ale **intra-layer** activations (Mamba chunked scan na 128 segmentech +
  attention QKV cached + MLP intermediate) jedné vrstvy se nevejdou do
  volných ~2.4 GB po model loadingu. Bez checkpoint OOM ihned, s
  checkpoint OOM později uvnitř Phase 3 (per-layer re-forward).
- **Plán řešení:**
  1. **Alpha.13 — sub-layer checkpointing.** Rozdělit jednu vrstvu na
     pre_norm / SSM / attention / MLP chunky. Synthetic loss propaguje
     gradient mezi sub-chunky stejným patternem jako per-layer.
  2. **Alpha.13+ — selective component-aware** (memory-priority drop
     attention QKV, keep SSM scan). Vyžaduje custom Op v Candle nebo
     re-architekturu kolem `Tensor::detach`.
  3. **Gaia deploy** s ≥ 24 GB VRAM pro 1.5B/7B production training —
     fallback, pokud sub-layer checkpoint nestačí.

### KI-006 — Training CPU F32 je 48 s/step na 1.5B (vyřešeno alpha.12)

- **Fáze:** 5 (Core Memory training)
- **Stav:** **VYŘEŠENO v alpha.12.** Per-layer gradient checkpointing
  redukuje CPU step time ze 48 s na **19 s** (2.5× rychleji) díky
  menšímu memory traffic — re-forward během backward je levnější než
  držet plný 24-layer autograd graf.
- **Kontext:** Alpha.11 baseline `seq_len=8 batch=1 grad_accum=2` na
  1.5B CPU F32: 48 s/step. Alpha.12 stejný setup s `--checkpoint`:
  19 s/step. Loss curve identické (7.11 → 3.70 best, pod random baseline
  ln(vocab)≈11.09).
- **Side-effect:** alpha.12 odhalila, že na CPU je checkpoint nejen
  pamětně, ale i **rychleji** — což otáčí intuici, že re-forward je
  drahý.

---

## Vyřešené

_(žádné zatím — všechny dosavadní bugy řešeny inline, viz BUGS.md)_
