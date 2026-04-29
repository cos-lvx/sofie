# Known Issues — Eleutheria

Známé problémy, limitace a vědomá technická rozhodnutí.

Formát: `KI-NNN` s fází, dopadem, kontextem a plánovaným řešením.

---

## Aktivní

### KI-007 — AdamW optimizer state se nepersistuje při resume

- **Fáze:** 5 (alpha.15)
- **Dopad:** Střední — pro multi-stage tréninky má lehkou disrupci v
  efektivním LR po resume (Adam bias correction kompenzuje warmup okno
  ~prvních 100–500 kroků, ne dlouhodobou trajektorii momentum/velocity)
- **Kontext:** `train-core-memory --resume-from <path>` načte init_states
  přes `CoreMemoryArtifact::into_stack`, ale `AdamW::new` startuje
  s prázdnými `m, v` moments. Pro krátký fine-tune nebo single-shot
  resume (alpha.15 use case) je to akceptovatelné. Pro long training
  s checkpoint/restart cyklem je to skutečná limitace.
- **Workaround:** vyšší LR warmup po resume nebo nižší LR.
- **Řešení:** alpha.16 — `core_memory.optim.safetensors` vedle
  artefaktu, per-Var m + v moments + step counter, auto-load při
  `--resume-from`.

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

### KI-005 — CUDA OOM pro multi-layer backward na 6 GB VRAM (vyřešeno alpha.13)

- **Fáze:** 5 (Core Memory training)
- **Stav:** **VYŘEŠENO v alpha.13** kombinací sub-layer chunkingu +
  agresivního progressive drop saved tensorů. RTX 4050 6 GB nyní zvládá
  multi-layer training 1.5B s seq_len=4 batch=1 grad_accum=1 stabilně:
  ~10 s/step, loss klesá normálně.
- **Kořenová příčina (alpha.12 stále padal):**
  1. **Per-layer chunking nestačil** — Mamba scan padding na chunk_size=128
     pro krátký seq alokoval 50-100 MB workspace per vrstva. Sub-layer
     rozdělení (chunk α: pre_norm + SSM + attention; chunk β: post_norm
     + MLP) snížilo peak max(α, β) místo sum.
  2. **Memory leak v Phase 3 reverse sweep** — saved tensorů ve `Vec`
     drží Arc references na GPU storage po celou dobu sweep, plus
     `final_grads` z Phase 2 loss.backward držel intermediate tensors
     lm_head workspace (~700 MB).
- **Fix:**
  - Sub-layer chunkování (`forward_chunk_branches` + `forward_chunk_mlp`)
  - `mem::replace` saved tensorů progressive drop v Phase 3 loop
  - `drop(loss)`, `drop(last_hidden_var)`, scope-bounded `final_grads`
  - `phase3_layer_reverse` helper s lokálními scope pro chunk α/β
- **Limit:** `grad_accum > 1` stále padá (accumulator + scratch). Pro
  6 GB VRAM použij `--grad-accum 1`. Pro `seq_len > 4` bude potřeba další
  optimalizace (alpha.14+ nebo větší VRAM).

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
