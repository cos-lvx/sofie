# Known Issues — Eleutheria

Známé problémy, limitace a vědomá technická rozhodnutí.

Formát: `KI-NNN` s fází, dopadem, kontextem a plánovaným řešením.

---

## Aktivní

### KI-008 — Adam bez warmup overshoots, loss osciluje místo monotonního descentu

- **Fáze:** 5 (alpha.15)
- **Dopad:** Střední — limituje konvergenci na současné HP, viditelné
  oscilace 1↔11 v loss trajektorii
- **Kontext:** Empiricky pozorováno 2026-04-29 ve smoke testu
  (RN-002, RN-006). Loss má 4-fázový pattern:
  1. Phase 1 (step 1-20): rapid descent na lokální minimum
  2. Phase 2 (step 20-40): overshoot — Adam velocity buffer naskočil
     na strong gradient, udělá obří krok ven z lokálního minima
  3. Phase 3 (step 40-100): noisy recovery
  4. Phase 4 (step 100+): slow descent s spike'y
  Best loss je ephemerálně dosažený někde v Phase 1 nebo začátku Phase 2,
  pak ztracen. Final loss reflektuje noisy stav, ne best.
- **Workaround:** žádný akutní; smoke validace funguje navzdory tomuto
- **Řešení:** Samostatný research patch (alpha.15.X nebo dedikovaný):
  - Linear LR warmup 0 → target přes prvních ~50 kroků (odstraní Phase 2)
  - LR cosine/linear decay v Phase 4 (stabilizuje convergenci)
  - Větší effective batch (Gaia, ne RTX 4050) sníží Phase 3 noise
  - Před implementací uděláme ablation runs (RN-006..00X)

### KI-009 — Artefakt drží final state, ne best snapshot

- **Fáze:** 5 (alpha.14/.15)
- **Dopad:** Střední — pro noisy training (KI-008) zahodíme nejlepší
  state. Ve smoke best=0.85 ephemerálně dosažen, uložen state s loss=8.86.
- **Kontext:** Mechanická vlastnost API (RN-003): `from_stack(&CoreMemoryStack)`
  čte aktuální `Var.as_tensor()` v okamžiku volání. `train_core_memory`
  volá save *po* posledním `optimizer.step()`, takže artefakt obsahuje
  state z **posledního** stepu, ne z best step.
- **Workaround:** zatím žádný; lze ručně checkpoint po N krocích, ale
  není API support
- **Řešení:** Dedikovaný patch (alpha.16+):
  - `BestSnapshotTracker` v `train_core_memory` — drží shadow CPU buffer
    Var values, aktualizuje při každém best loss improvement
  - Save volá `from_snapshot()` místo `from_stack()` pokud snapshot
    existuje
  - Alternativa: periodic save každých N kroků s rolling best mark

### KI-011 — `train_core_memory` reportuje "loss nedecreased" pro úspěšný resume

- **Fáze:** 5 (alpha.15)
- **Dopad:** Nízký (špatný exit code + matoucí UX) — ne datová ztráta
- **Kontext:** `TrainingResult.loss_decreased = final < initial` byl
  rozumný criterion pro **fresh** training (initial = random baseline
  ~9.85, final by měl klesnout). Pro **resume** training je špatný:
  - Initial loss je už nízká (díky trained state přes `into_stack`)
  - Final loss může být cokoli v noisy oscilaci pásmu (RN-002)
  - `final < initial` je teď spíš **noise gate**, ne signal of success
  Empirický příklad ze smoke testu 2026-04-29: stage 2 resume,
  initial=7.13, final=8.86 → engine reportuje
  `✗ Loss neklesl, Err("training loss nedecreased")` přestože:
  - Best ephemerální=0.8535 (lepší než stage 1 best=0.9965)
  - Save proběhl s kumulativním 335 steps
  - State tuning prokazatelně pokračoval z trained init
- **Workaround:** ignorovat exit code; verifikuj přes
  `--inspect-core-memory` že artefakt má smysluplná metadata.
- **Řešení:** alpha.16+ revidovat `loss_decreased` criterion:
  - Buď `final < ln(vocab_size)` (random baseline jako sanity)
  - Nebo per-mode logika: fresh → final < initial, resume → final < ln(vocab)
  - Nebo úplně odstranit (engine signalizuje úspěch save vs. NaN halt,
    quality judgement nech na uživateli skrz inspect)

### KI-010 — Training subkomandy double-loadují Core Memory

- **Fáze:** 5 (alpha.14/.15)
- **Dopad:** Nízký — funkční, ale redundantní I/O a matoucí logging
- **Kontext:** Pokud uživatel spustí
  `train-core-memory --resume-from <path>` a artefakt existuje na default
  cestě (`~/.eleutheria/core_memory.safetensors`), proběhnou **dva
  loady**:
  1. `Sofie::load` v main.rs auto-discoveruje soubor a připojí ho přes
     `attach_core_memory` (alpha.14 logika). Sofie::core_memory = Some(...).
  2. `run_train` pak načte stejný (nebo jiný) soubor přes `--resume-from`,
     `into_stack` zkonstruuje trainable CoreMemoryStack.
  Training compute path **nepoužívá Sofie::core_memory** vůbec (stack
  injektuje vlastní Var-y), takže auto-attach v této cestě je dead
  weight. Logging vypisuje "Core Memory: ..." dvakrát.
- **Workaround:** žádný akutní není potřeba — funkční chování. Pokud
  uživatel chce čisté logy, použije `--no-core-memory` při training
  subkomandě.
- **Řešení:** alpha.16+ clean-up patch — v `main.rs` rozhodnout o
  auto-attachi na základě `args.command`: pro
  `TrainCoreMemory{,Smoke,Multi}` subkomandy auto-attach přeskočit.
  Inference flow (REPL, single-shot) auto-attach pochopitelně potřebuje.

### KI-007 — AdamW optimizer state se nepersistuje při resume (vyřešeno alpha.16)

- **Fáze:** 5 (alpha.15)
- **Stav:** **VYŘEŠENO v alpha.16** — `EleutheriaAdamW` (vlastní wrapper
  s veřejným state) + `OptimizerArtifact` (sourozenec
  `<core>.optim.safetensors`). Auto-load při `--resume-from` pokud
  sourozenec existuje, auto-save při `--output`. Soft resume (sourozenec
  chybí → prázdný Adam) zachován pro backwards-compatibility.
- **Kontext:** `train-core-memory --resume-from <path>` v alpha.15 načetl
  init_states přes `CoreMemoryArtifact::into_stack`, ale `AdamW::new`
  startoval s prázdnými `m, v` moments. Adam bias correction kompenzoval
  warmup okno (~prvních 100–500 kroků), ale velocity buffer naskakoval
  znova → Phase 2 overshoot fáze (RN-002, RN-006).
- **Fix:** Re-implementace AdamW algoritmu (`adamw_state.rs`,
  byte-identická s Candle), `OptimizerArtifact` v `optim_io.rs`,
  konvence `<core>.optim.safetensors`.

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
