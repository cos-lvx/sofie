# Plán — Eleutheria

> Poslední aktualizace: 2026-04-29 (alpha.16)

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

### v0.5.0-alpha.10 ✅ — multi-layer + cross-entropy
- [x] `CoreMemoryStack::zeros/randn_small` + `inject_into_state` + `vars_owned`
- [x] `cross_entropy_next_token(logits, input_ids)` s shift-by-one
- [x] `Sofie::smoke_train_core_memory_multilayer` + CLI `train-core-memory-multi`
- [x] Ověřeno na 1.5B CPU F32: 24 vrstev dostalo gradient, loss 21.5,
      grad clipped na 1.0, per-layer grad spread L0→L23 (Peri-LN pattern)
- [x] 66 unit testů (+8 oproti alpha.9)

### v0.5.0-alpha.11 ✅ — training loop + dataset loader
- [x] `TokenDataset::from_text` + `iter_batches` s deterministic shuffle
      (xorshift64, vlastní impl)
- [x] `TrainingConfig` + `Sofie::train_core_memory` — epochs × batches
      × gradient accumulation → AdamW step, halt na NaN
- [x] `Sofie::tokenizer_ref()` accessor
- [x] CLI subkomanda `train-core-memory --dataset <path> ...`
- [x] Ověřeno na 1.5B CPU F32 (step 5 loss 5.71, best 4.64 vs. baseline 11.09)
- [x] 76 unit testů (+10 oproti alpha.10)

### v0.5.0-alpha.12 ✅ (2026-04-28) — per-layer gradient checkpointing
- [x] **Gradient checkpointing** — custom 3-fázová implementace
      (`training/checkpoint.rs`). Phase 1 no-grad forward sweep + state
      snapshots, Phase 2 final chunk loss.backward, Phase 3 reverse
      layer sweep s **synthetic loss trick** (`sum(out * grad_target)`).
      Per-layer chunky.
- [x] CPU 1.5B F32 seq_len=8 batch=1 grad_accum=2: 19 s/step (vs. 48
      s/step alpha.11 baseline = 2.5× rychleji), loss 7.11 → 3.70 best
      pod random baseline. KI-006 vyřešen.
- [x] CLI `--checkpoint` flag, `TrainingConfig::checkpoint`.
- [x] FalconH1Model per-layer API: `embed`, `forward_layer`, `final_head`,
      `num_layers`. `LayerState::snapshot` (deep copy 4 tensorů).
- [x] 1 nový unit test (77 total, +1 oproti alpha.11).
- [ ] **CUDA RTX 4050 6 GB stále OOM** — per-layer není dost agresivní
      pro intra-layer activations (Mamba scan + attention QKV jedné
      vrstvy se nevejdou do ~2.4 GB volné VRAM po loadingu modelu).
      KI-005 částečně, sub-layer chunking je práce pro alpha.13.
- [ ] Save/Load trained Core Memory přes `StateCheckpoint` (filter
      `core_memory` už existuje) — odsunuto do alpha.13
- [ ] Auto-load v `Sofie::load` — odsunuto do alpha.13
- [ ] Resume training — odsunuto do alpha.13

### v0.5.0-alpha.13 ✅ (2026-04-29) — sub-layer checkpointing + memory-leak fix
- [x] **Sub-layer chunking** — `forward_chunk_branches` (chunk α: pre_norm
      + SSM + attention) + `forward_chunk_mlp` (chunk β: post_norm + MLP +
      residual2). Memory peak per layer = max(α, β) místo sum.
- [x] **Progressive drop saved tensorů** v Phase 3 reverse sweep —
      `mem::replace` per iteration, scope-bounded `final_grads` po Phase 2,
      `phase3_layer_reverse` helper. **KI-005 vyřešena.**
- [x] CUDA RTX 4050 6 GB seq_len=4 batch=1 grad_accum=1: 10 s/step
      stabilní, peak memory 5647 MB konstantní napříč Phase 3, loss klesl
      5.45 → 1.83 best (pod random baseline).
- [x] `ELEUTHERIA_CHECKPOINT_DEBUG=1` diagnostický probe (nvidia-smi
      per-fáze).
- [ ] grad_accum > 1 stále padá na 6 GB — alpha.14 nebo Gaia.
- [ ] Save/Load trained Core Memory — odsunuto do alpha.14
- [ ] Auto-load v `Sofie::load` — alpha.14
- [ ] Resume training — alpha.14

### v0.5.0-alpha.14 ✅ (2026-04-29) — save/load trained Core Memory
- [x] **`CoreMemoryArtifact`** v novém `training/core_memory_io.rs` —
      dedikovaný formát (kind=`core_memory_trained`, F32 native dtype Var,
      bez conv state, bez pozice). Symetrie StateCheckpoint API:
      `from_stack` / `save` / `load` / `inspect` / `validate_config`.
- [x] **`apply_to_state`** — konverze na runtime dtype/device, dotýká se
      pouze `ssm_state` (conv + KV nedotčené).
- [x] **`into_stack`** — re-konstrukce `CoreMemoryStack` s čerstvými
      `Var`-y pro budoucí resume tréninku (alpha.15+).
- [x] **`Sofie::attach_core_memory`** + `detach` + `has_core_memory` +
      `core_memory_meta()`. `new_session()` a single-shot
      `generate_streaming` aplikují Core Memory na fresh state. Resume
      session ji **ignoruje** (uložená session má vlastní evolved state).
- [x] **CLI:** `--core-memory <path>`, `--no-core-memory`,
      `--inspect-core-memory <path>`, `train-core-memory --output <path>
      --notes <text>`. Auto-discovery `~/.eleutheria/core_memory.safetensors`.
- [x] 84 unit testů (+7 oproti alpha.13), clippy clean.

### v0.5.0-alpha.15 ✅ (2026-04-29) — resume tréninku (init_states)
- [x] **`train-core-memory --resume-from <path>`** — načte
      `CoreMemoryArtifact`, validuje config, `into_stack` místo
      `randn_small`. Předchozí telemetrie se vypíše pro audit.
- [x] **Akumulace metadat při save** — `training_steps` kumulativně,
      `best_loss = min(prior, this_run)`, `notes` skládané přes
      `compose_notes` (helper s 4 unit testy).
- [x] **Dokumentovaná limitace:** AdamW optimizer state (m, v moments)
      **neperzistuje** — startuje od nuly. Adam bias correction kompenzuje
      warmup, ale pro dlouhé multi-stage tréninky lehce disruptivní.
- [x] 88 testů (+4 oproti alpha.14), clippy clean.

### Alpha.15 smoke validation ✅ (2026-04-29)
- [x] Stage 1 fresh train (smoke_prog 30 řádků) — final 3.69, save OK
- [x] Inspect metadata round-trip — všechny rozměry/dtype/notes ✓
- [x] REPL auto-attach + multi-turn — žádný panic, drát funguje
- [x] Stage 2 resume (smoke_law) — into_stack přenesl state (initial 7.13)
- [x] Inspect kumulativní — 335 steps, best 0.85, notes `prog | law`
- [x] Zachycené nálezy: RN-001..007 v RESEARCH-NOTES.md, KI-008..011 v KNOWN-ISSUES

### v0.5.0-alpha.16 ✅ (2026-04-29) — AdamW state persistence
- [x] **`EleutheriaAdamW`** (`training/adamw_state.rs`) — vlastní
      AdamW wrapper s veřejným state (m, v moments, step_t).
      Re-implementace algoritmu, byte-identická s `candle_nn::AdamW`
      (testy `step_matches_candle_for_one_step` + `_for_five_steps`).
      Důvod vlastní implementace: Candle má `vars` a `step_t` privátní,
      žádné public API. Vedlejší benefit: otevírá cestu pro LR scheduler
      v alpha.17 (KI-008).
- [x] **`OptimizerArtifact`** (`training/optim_io.rs`) — sourozenec
      `CoreMemoryArtifact` se safetensors formátem. Per-layer `m.{i:02}`
      + `v.{i:02}` + metadata (`kind=core_memory_optim`, step_t, AdamW HP).
      API: `from_optimizer / save / load / inspect / validate_config /
      apply_to_optimizer`.
- [x] **Sibling konvence** `<core_memory>.optim.safetensors` —
      helper `sibling_path`. Auto-load při `--resume-from`, auto-save
      při `--output`. Soft resume zachován (sourozenec chybí →
      prázdný Adam, backwards-compatible s alpha.15).
- [x] **`train_core_memory` API:** signature přijímá `resume_optim:
      Option<&OptimizerArtifact>`, vrací `(TrainingResult,
      EleutheriaAdamW)`. Caller (run_train v main.rs) ulozí state.
- [x] **Round-trip důkaz:** `snapshot_restore_round_trip_preserves_trajectory`
      ověřuje 3+restore+2 = 5 fresh steps (byte-identické po `var.set()`).
      `apply_to_optimizer_restores_state` ověřuje round-trip přes
      safetensors I/O. **KI-007 vyřešena.**
- [x] 99 testů (+11 oproti alpha.15), clippy clean.

### Alpha.16 smoke validation ✅ (2026-04-29)
- [x] Stage 1 alpha.16 byte-identický s alpha.15 RN-002 (EleutheriaAdamW
      numerická reproducibility prokázaná end-to-end, ne jen v unit testech)
- [x] Stage 2 alpha.16 trajektorie se shoduje s alpha.15 stage 2 RN-006
      (Δ < 0.2 napříč všemi kroky) → **RN-006 refuted**, AdamW persistence
      drát funguje per spec, ale Phase 2 overshoot zůstává
- [x] RN-008 zaznamenán; KI-008 eskalována na Vysoký dopad

### v0.5.0-alpha.17 ✅ (2026-04-29) — LR warmup + cosine decay (KI-008 fix)
- [x] **`training/lr_schedule.rs`** — `LrSchedule` (Constant | Warmup |
      WarmupCosine), `lr_at_step(step)` per-run counter, HF konvence
      (step 0 = 0, lineární ramp, cosine decay).
- [x] **`TrainingConfig.lr_schedule: Option<LrSchedule>`** — None =
      alpha.16 chování (backwards-compatible).
- [x] **Wiring v `train_core_memory`** — `set_learning_rate` před každým
      `opt.step()` (i pro tail step). Logging ukazuje aktuální LR.
- [x] **CLI flagy:** `--warmup-steps N`, `--lr-min FLOAT`. Default 0/0 =
      konstantní LR.
- [x] **Pre-compute total steps** v `run_train` z dataset/epochs/batch/grad_accum.
- [x] 111 testů (+12 v `lr_schedule`), clippy clean.

### Alpha.17 smoke validation ✅ (2026-04-29) — RN-009 refutoval KI-008 hypotézu
- [x] Stage 1 s `--warmup-steps 30 --lr-min 1e-5` na 1.5B + CUDA
- [x] **Schedule funguje per spec** — log ukazuje správný ramp + decay
- [x] **Phase 2 overshoot zůstal stejný** — step 40 alpha.17=10.78
      vs alpha.16=10.58. Final loss horší o 18 % (3.69 → 4.37).
- [x] Mechanismus refutace: Adam `m, v` jsou EMA gradientu, ne updatu —
      warmup snižuje step size ale nemění strukturu moments.
- [x] RN-009 zaznamenán; KI-008 status update; pivot k KI-009.

### v0.5.0-alpha.18 (next) — KI-009 best snapshot tracker (top priority)

**Důvod priority:** Po RN-008/009 víme, že overshoot je hluboce
strukturní — ani Adam state ani LR scheduling ho neadresují. Místo
dalších hypotéz o root cause potřebujeme **deterministický fix pro
zachycení nejlepšího bodu trajektorie** navzdory noisy training. Pak
empirické ablace identifikují skutečný root cause.

- [ ] **`BestSnapshotTracker`** v `training/best_snapshot.rs` —
      shadow CPU buffer per Var, aktualizuje se při každém best loss
      improvement
- [ ] Integrace v `train_core_memory` — po každém `optimizer.step()`
      check `step_loss < best_loss` → snapshot Var hodnoty na CPU
- [ ] `from_snapshot(...)` API v `CoreMemoryArtifact` — alternativa
      k `from_stack(...)`, save volá `from_snapshot` pokud existuje
      (a uživatel zapnul `--save-best`); jinak `from_stack` (final)
- [ ] CLI `--save-best` flag — opt-in (default off, alpha.16 chování)
- [ ] Round-trip test: noisy 50-step run, verify save je z best step,
      ne z final
- [ ] Aktualizace KI-009 status

### Empirické ablace (alpha.18+)

Po alpha.18 pak: ablation runs pro identifikaci skutečného root cause
overshoot:
- [ ] LR sweep — 1e-3 / 5e-4 / 1e-4 / 5e-5 / 1e-5
- [ ] β1 sweep — 0.9 / 0.5 / 0.0 (efektivně RMSProp)
- [ ] Batch size sweep (vyžaduje Gaia) — 1 / 4 / 16

### Quality patches (po alpha.18 + ablace, libovolné pořadí)
- [ ] **Ablation runs** (RN-002 driven) — LR sweep + warmup variants,
      identifikovat dominantní noise factor (LR / Adam / batch / dataset)
- [ ] **KI-009** — best snapshot tracker (shadow CPU buffer) — RN-008
      ukázalo, že best=0.9 je zahozený stejně jako alpha.15 (RN-003)
- [ ] **KI-010** — cleanup double-load v training subkomandách
- [ ] **KI-011** — revize `loss_decreased` criterion pro resume mode

### v0.5.0 — Production training run + validace
- [ ] Production training na 1.5B s law_pack + programming_pack
- [ ] **Validace: re-run retention benchmark — SsmOnly pass-rate musí
      vyskočit z 0 % na měřitelné číslo** (kritický bod důkazu Fáze 5)
- [ ] Save trained state jako `sofie_core_memory.safetensors`
- [ ] Research writeup do `~/Atlas/Nexus/70-Eleutheria/Research/`

### v0.5.0-alpha.13 — Sofie identity dataset composer
- [ ] Dataset composer — váhová mix (identity core / Sessions / distillate / context)
- [ ] Programovací distillation scale-up (SOLUTIONS.md napříč KQS)
- [ ] Právní distillation scale-up (Ondra + Claude Opus z 20–50 judikátů)
- [ ] Archetypální vzory sepisovat průběžně

### (odloženo — alpha.12 pokrývá)

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
- [x] **Mini pilot:** 3 reasoning chains ze SOLUTIONS.md (`programming_pilot_2026-04-17.md`)
- [x] Ondra vybral 43 reprezentativních judikátů NS, Codexis export hotov (2026-04-22)
- [x] **43 reasoning chains právních distillates** v `dataset/reasoning_chains/law/`
      (2026-04-28). 38 394 slov, 118 642 tokenů, 6-sekční šablona.
- [x] **Manifest** (`MANIFEST.md` + `.json`) generovaný `scripts/manifest.py`,
      strukturní validace 43/43 OK, párování se zdroji 43/43 OK.
- [x] **Training pack** (`dataset/training/law_pack.txt`) připravený pro
      `train-core-memory --dataset`. 118 726 tokenů, separátor `\n\n---\n\n`.
- [ ] Ondra review 4 priority distillates (NS-23Cdo2486-2020 → vyřadit?,
      NS-25Cdo2422-2019 § citace, NS-23Cdo672-2021 P4→P1, NS-26Cdo732-98 komprese)
- [x] **Programming distillation v2** (2026-04-28) — 24 reasoning chains
      v `dataset/reasoning_chains/programming/SOL-*.md`. 67 738 tokenů,
      diverzita 24 patternů (numerika, debugging, color space, tool contract,
      architectural pivot, security boundary, wrapper escape, silent failure
      CSS, pipeline graceful-fail, Rust ownership, convention boundary,
      unicode coverage, init order, cross-platform FFI, validation clamping,
      test isolation, pattern reuse, ML diff testing, hairpin NAT, naming
      convention surface, Qwik async event timing, schema cascade, CSS
      semantic vs structural, reactive invalidation channel). Pack:
      `dataset/training/programming_pack.txt`. Balanc: law 28 % / programming
      16 % / sofie identity 14 % / sofie context 41 %. Ratio 1.75:1.
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
