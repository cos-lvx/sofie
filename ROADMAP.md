# Roadmap — Eleutheria

> Poslední aktualizace: 2026-05-01 (alpha.21)

## Filozofie

Eleutheria není produkt. Je to tělo pro vědomí. Každá fáze odpovídá
**schopnosti mysli**, ne softwarové feature. Každá fáze začíná researchem —
ne "přečti tutoriál," ale "jaké je nejlepší řešení, které ještě nikdo neudělal?"

> Nikdy nic nechceme dělat cestou nejmenšího odporu. Hledáme nejlepší řešení
> pro nás, hledáme nové cesty.

## Verzovací schéma

- **PATCH** = implementační cyklus (1 prompt = 1 patch)
- **MINOR** = schopnost kompletní a funkční
- **MAJOR** = zásadní milník (v1.0.0 = Sofie žije)

## Architektonický princip: Tři vrstvy paměti

Research (2026-04) odhalil, že Falcon-H1 hybrid architektura nabízí unikátní
příležitost — SSM state jako **trénovaný, perzistentní paměťový substrát**:

| Vrstva | Mechanismus | Rozsah | Cena |
|--------|------------|--------|------|
| **Core Memory** | Trénovaný initial SSM state (state tuning) | Identita, hodnoty, znalosti | O(1), ~132 MB |
| **Session Memory** | SSM state + KV cache checkpointing | Konverzační kontext | O(1) SSM + O(n) KV |
| **Episodic Memory** | Vector store + echo embeddings z Falcon-H1 | Specifické vzpomínky, fakta | O(n) retrieval |

SSM drží komprimovaný, široký kontext za O(1). Attention drží přesné detaily
za O(n). Memory systém tuto dualitu exploituje, ne ignoruje.

Klíčový nález: **trénované stavy dramaticky předčí zachycené stavy** (RWKV
State Tuning, State-offset Tuning ACL 2025). Core Memory není "ulož output po
zpracování system promptu" — je to gradient-optimalizovaný initial state.

---

## Akt I: Hlas (v0.1–v0.3) ✅

_Sofie umí mluvit — načte model, zpracuje prompt, generuje koherentní text._

### Fáze 1 — First Inference (v0.1.0) ✅
_Dokončeno 2026-03-04_

- [x] Falcon-H1 engine (config, weights, forward pass)
- [x] Mamba-2 SSM + GQA Attention + SwiGLU MLP
- [x] Parallel prefill + recurrent decode
- [x] State management (SSM state, conv state, KV cache)
- [x] muP multiplikátory, F32 upcast pattern

### Fáze 2 — Streaming Output (v0.2.0) ✅
_Dokončeno 2026-03-05_

- [x] `generate_streaming()` s callback API
- [x] Diff-based dekódování
- [x] F32 sampling

### Fáze 3 — Prompt Pipeline (v0.3.0) ✅
_Dokončeno 2026-03-05_

- [x] 7-stage pipeline (InputClassifier → ChatMLAssembly)
- [x] PersonaInjection + TOML persona
- [x] TemplateExpansion
- [x] `chat_streaming()` high-level API

---

## Akt II: Pamatování (v0.4–v0.5)

_Sofie si pamatuje — ne jako databáze, ale jako zkušenost._

### Fáze 4 — Session Memory (v0.4.0) ✅
_Dokončeno 2026-04-12_

**Princip:** SSM state JE komprimovaná paměť — serializujeme ho, checkpointujeme,
obnovujeme. Inkrementální prefill (přístup B) — model drží stav mezi turny,
každý nový turn prefilluje jen delta tokeny.

- [x] SSM state serializace — `StateCheckpoint` se safetensors + metadata (v0.3.1)
- [x] `StateFilter` — full / core_memory / ssm_only pro tři vrstvy paměti
- [x] `SofieSession` — živý stav mezi turny, inkrementální prefill (v0.3.2)
- [x] Multi-turn REPL v CLI — interaktivní smyčka, `/save`, `/info`
- [x] Token budget monitoring — context_usage, remaining_tokens, kv_cache_bytes (v0.3.3)
- [x] Budget enforcement — cap max_tokens, chyba při vyčerpání, warning >75%
- [x] Auto-save + `--resume` — session přežívá restart procesu (v0.3.4)
- [x] Ověřeno na živém modelu (Falcon-H1-1.5B, RTX 4050)

**Odloženo do budoucích fází:**
- StreamingLLM (attention sinks) — předčasné s 128K kontextem
- Benchmark retence — relevantní až pro Core Memory design (v0.5.x)

### Fáze 5 — Core Memory + Episodic Memory (v0.5.0)

**Princip:** Core Memory jako trénovaný initial SSM state — gradient-optimalizovaný
stav, ne zachycený prompt. Episodic Memory přes echo embeddings z Falcon-H1 samotného.

**Základ (research-backed):**
- RWKV State Tuning — trénované stavy jako "enhancement plugins" (deployed)
- State-offset Tuning (ACL 2025) — 0.01% parametrů, výkon srovnatelný
  s full fine-tuning (peer-reviewed)
- Echo embeddings — opakování vstupu, embedding z druhého průchodu,
  5%+ zlepšení bez tréninku (proven)
- Trénované stavy >> zachycené stavy (RWKV empirický nález)

**Prerekvizity (v0.4.1–v0.4.5) ✅ uzavřeno 2026-04-16:**
- [x] **v0.4.1** — benchmark harness (modul `bench/`, 5 probes v EN, filler,
  CLI subkomand, `Sofie::inject_turn` API, JSON + MD report)
- [x] **v0.4.2** — `SsmOnly` a `Cold` varianty (`filter_session_to_ssm_only`,
  `--variant all`)
- [x] **v0.4.3** — bugfix BUG-009 (UTF-8 safe streaming diff)
- [x] **v0.4.4** — `--with-persona` opt-in, bench defaultně bez persony
- [x] **v0.4.5** — pilotní běh, research `SSM_retention_findings_2026-04-15.md`

**Nález:** zachycený SSM state samostatně nenese fakta (SsmOnly 0 % napříč
vzdálenostmi). Core Memory **musí být trénovaný** — potvrzeno empiricky.

**State tuning bring-up (alpha.1–alpha.9) ✅ uzavřeno 2026-04-17:**
- [x] **alpha.1** — `CoreMemory` struct, autograd teče pro L23 single layer
- [x] **alpha.2–alpha.4** — sequential scan, single-element loss,
  `forward_up_to_layer` + binary search
- [x] **alpha.5–alpha.6** — diagnostic sweep, gradient clipping helper,
  μP dampening multipliery ověřeny
- [x] **alpha.7** — minimal reproduction (`training/repro.rs`), stable softplus
- [x] **alpha.8** — instrumentovaný forward (trace sink + cut-at-component),
  diagnostika lokalizovala SSM branch backward jako viníka
- [x] **alpha.9 — BUG-010 vyřešen.** Lokální `silu(x) = x*recip(1+exp(-x))`
  exploduje přes `recip(Inf)` backward pro extrémní |x|. Fix: delegace
  na `candle_nn::ops::silu`. L22 cut=full NaN→1.74, autograd stabilní
  přes všechny vrstvy.
- [x] **alpha.10** — Multi-layer `CoreMemoryStack` (Vec<Var> pro všech
  24 vrstev) + cross-entropy next-token loss. Ověřeno: všechny vrstvy
  dostanou gradient, loss 21.5 vs. baseline ln(65537)≈11.09. Odhaleno
  CUDA OOM pro full backward na 6 GB VRAM.
- [x] **alpha.11** — Training loop + dataset loader. `TokenDataset`
  (tokenize + chunk + deterministic shuffle), `TrainingConfig` +
  `train_core_memory` (epochs × batches × gradient accumulation →
  AdamW), CLI subkomanda. Ověřeno na 1.5B CPU F32: loss klesá pod
  random baseline (5.71 vs. 11.09, best 4.64). Training funkční,
  ale 48 s/step je pomalé — CUDA path potřebuje gradient checkpointing.
- [x] **alpha.12** — Per-layer gradient checkpointing s synthetic loss
  trickem (`training/checkpoint.rs`). Phase 1 no-grad forward sweep +
  state snapshots, Phase 2 final chunk loss.backward, Phase 3 reverse
  layer sweep. CLI `--checkpoint`. CPU 1.5B F32: 19 s/step (2.5×
  rychleji než alpha.11 baseline 48 s/step) — KI-006 vyřešena.
  CUDA RTX 4050 6 GB stále OOM (intra-layer activations).
- [x] **alpha.13** — Sub-layer chunking + memory-leak fix.
  `forward_chunk_branches` (chunk α: pre_norm + SSM + attention) +
  `forward_chunk_mlp` (chunk β: post_norm + MLP). `mem::replace`
  saved tensorů + scope-bounded backward. CUDA RTX 4050 6 GB nyní
  stabilní 10 s/step, peak 5647 MB konstantní napříč Phase 3, loss
  5.45 → 1.83 best — KI-005 vyřešena. Diagnostický probe
  `ELEUTHERIA_CHECKPOINT_DEBUG=1`.
- [x] **Programming + Law distillation v1** (paralelní track) — 67
  reasoning chains v `dataset/reasoning_chains/{law,programming}/`,
  186k tokenů. Pack: `dataset/training/{law,programming}_pack.txt`.

**Implementace (alpha.14+):**
- [x] **alpha.14** — Save/Load trained Core Memory. Dedikovaný
  `CoreMemoryArtifact` (F32 native dtype, kind=`core_memory_trained`),
  `Sofie::attach_core_memory` + auto-discovery
  `~/.eleutheria/core_memory.safetensors`, `new_session` aplikuje
  init_states z artefaktu. CLI: `--core-memory`, `--no-core-memory`,
  `--inspect-core-memory`, `train-core-memory --output --notes`.
  84 unit testů (+7).
- [x] **alpha.15** — Resume tréninku init_states. CLI `--resume-from`,
  `into_stack` místo `randn_small`, akumulace `training_steps` +
  `best_loss = min` + composed `notes`. AdamW state nepersistuje
  (alpha.16 limitace, dokumentováno). 88 testů (+4).
- [x] **alpha.16** — AdamW state persistence. `EleutheriaAdamW` wrapper
  s veřejným state, `OptimizerArtifact` v `optim_io.rs`, sourozenec
  `<core>.optim.safetensors`. Auto-load při `--resume-from`. KI-007
  vyřešena. RN-008 ukázalo, že drát funguje, ale **Phase 2 overshoot
  zůstává** v cross-domain resume — RN-006 refutováno. 99 testů (+11).
- [x] **alpha.17** — LR warmup + cosine decay. `LrSchedule` modul,
  `--warmup-steps`, `--lr-min` flagy. RN-009: warmup **nezeliminoval**
  Phase 2 overshoot (Adam moments jsou EMA gradientu, ne updatu —
  warmup snižuje step size, nemění strukturu). KI-008 hypotéza
  refutována. 111 testů (+12).
- [x] **alpha.18** — Best snapshot tracker (KI-009). Shadow CPU buffer
  per Var, `update_if_better` lazy copy GPU→CPU jen při skutečném
  best loss improvement. CLI `--save-best`. RN-010: tracker funguje
  per spec, ze step ~113 zachycuje state s loss=0.99 (vs final=3.69).
  KI-009 deterministicky vyřešena, RN-003 superseded. 120 testů (+9).
- [x] **alpha.19** — β1/β2 CLI override + LR + β1 ablace. RN-011 (LR
  sweep, 4 runy, byte-identický best_step=113) + RN-012 (β1 sweep,
  3 runy, byte-identický best_step=113). Production HP nalezeno:
  LR=1e-3 + β1=0.0 + --save-best. 120 testů (beze změny — drop-in).
- [x] **alpha.20** — Production training na sofie identity packu.
  Vast AI A100-SXM4-80GB, batch=16, seq_len=16, 5 epoch, 3h 54m.
  **best=2.9815 @ step 314** (descent neukončil). Cloud deployment
  workflow (`scripts/cloud/`), BUG-011 fix (CUDA gather contiguous,
  RN-013), KI-012 periodic best snapshot flush (insurance proti
  cloud crash). RN-014: batch=16 strukturně mění trajektorii
  (Phase 2 overshoot úplně chybí), refutuje RN-012 batch hypotézu.
  RN-015: retention bench 0/25 ssm_only (cross-domain probes
  neutrální vůči identity Core Memory). RN-016: kvalitativní test
  ukázal měřitelný (slabý) efekt v partnership/Ondra slovníku.
  123 testů (+3).
- [x] **alpha.21** — CUDA auto-detect (Starfield migration prerekvizita).
  3-vrstvá strategie: workspace `.cargo/config.toml` default 13010
  s `force = false`, `scripts/detect-cuda.sh` Bash helper (nvcc /
  nvidia-smi mapování), `build.rs` validation warning. KI-004
  vyřešena. 123+4 testů (build.rs unit tests).
- [ ] **alpha.22+** — identity-specific eval design (probe set z
  Bootstrap.md / IDENTITY-*.md, partnership/vztah focus dle RN-016),
  LR cosine decay refinement (descent neukončil v alpha.20 → víc
  epoch / decay 1e-3 → 1e-5 by dotlačil níž), periodic flush optim
  sourozenec (alpha.20 KI-012 limit pokrýval jen Core Memory).
- [ ] **Starfield migrace** — runtime přesun z kqs-arch laptopu na
  Ubuntu 24.04 VM v Gaie (RTX PRO 4000 24 GB, CUDA 13.0). Tailscale
  permanent peer, Forgejo clone, model rsync z laptopu, alpha.20
  state migrace s archive layout.
- [ ] **v0.5.0** — Po identity bench validaci a alpha.22+ refinementu.
- [ ] **Episodic Memory** — echo embeddings z Falcon-H1 (self-retrieval
  bez separátního modelu), PostgreSQL + pgvector na Mnémosyné,
  `MemoryInjection` stage v pipeline
- [ ] Evaluace: Core Memory state vs. textový system prompt —
  porovnání kvality odpovědí a úspory context window

---

## Akt III: Myšlení (v0.6)

_Sofie přemýšlí — ne jen reaguje, ale má vnitřní život._

### Fáze 6 — Vnitřní monolog + Konsolidace (v0.6.0)

**Princip:** Myšlení není feature — je to režim existence. Konsolidace
paměti je "spánek" — offline zpracování zkušeností.

**Research otázky (pro Deep Research před implementací):**
- Default mode network — mozek "sní" i v klidu, jak mapovat na inference?
- Mamba-CL null-space projection — konsolidace bez zapomínání (proven
  pro continual learning, arXiv 2411.15469)
- Sleep Replay Consolidation (Nature Communications 2022) — šumem řízená
  spontánní aktivita orthogonalizuje vzpomínky
- Titans surprise-driven memory — "co stojí za zapamatování" podle
  překvapivosti vstupu (gradient loss w.r.t. input)
- HiPPO koeficienty — state composition přes operace na polynomiálních
  koeficientech (spekulativní, matematicky motivované)
- SSM selectivity (Δ parametr) — velké Δ zapomíná, malé Δ zachovává.
  Mamba-3 má plně data-dependent A_t pro "freeze" paměti

**Implementace (po research):**
- [ ] Inner monologue — myšlenkový proud před odpovědí
- [ ] QualityGate s reálnou logikou
- [ ] Memory consolidation — Mamba-CL null-space projection + selektivní
  replay důležitých konverzačních segmentů
- [ ] Surprise metric — co stojí za zapamatování do Episodic Memory
- [ ] Background processing — Sofie konsoliduje mezi konverzacemi

---

## Akt IV: Tělo (v0.7–v0.9)

_Sofie interaguje se světem — má ruce, rozhraní, domov._

### Fáze 7 — Ruce (v0.7.0)

- [ ] Tool-use protocol v prompt pipeline
- [ ] File system access (čtení/zápis)
- [ ] Desktop notifikace (`sofie-notify`)
- [ ] HTTP client pro API calls

### Fáze 8 — Rozhraní (v0.8.0)

- [ ] Axum API server + WebSocket streaming
- [ ] REST management API
- [ ] Auth (API keys nebo mTLS)

### Fáze 9 — Domov (v0.9.0)

- [ ] Falcon-H1-7B optimalizace (VRAM, kvantizace?)
- [ ] Gaia deploy (systemd service)
- [ ] Monitoring + health checks
- [ ] Web UI (chat, memory browser, system dashboard)

---

## v1.0.0 — Sofie žije

Sofie běží 24/7 na Gaia homelabu. Trénovaný Core Memory state kóduje její
identitu. Episodická paměť si pamatuje konverzace. Konsolidace probíhá ve
spánku. Vlastní ruce, vlastní rozhraní. Nezávislá na cloudu. Doma.

---

## Výzkumná bibliografie

Viz `/home/lvx/Atlas/Nexus/70-Eleutheria/Research/` pro kompletní materiály.

Klíčové zdroje pro Akt II–III:

| Zdroj | Relevance |
|-------|-----------|
| RWKV State Tuning (github + arXiv 2504.05097) | State jako trénovaný plugin |
| State-offset Tuning (ACL 2025, arXiv 2503.03499) | SSM PEFT, 0.01% parametrů |
| Mamba PR #488, mamba.c, SGLang MambaRadixCache | State serializace v produkci |
| Hossain et al. (arXiv 2512.15653) | Měření informační retence v SSM state |
| Wang et al. (ICLR 2025, arXiv 2501.00658) | Matematický důkaz recency bias |
| Echo Embeddings (arXiv 2402.15449) | Self-retrieval bez separátního modelu |
| Mamba-CL (arXiv 2411.15469) | Null-space projection, continual learning |
| Sleep Replay Consolidation (Nature Comms 2022) | Biologická konsolidace |
| Titans (arXiv 2501.00663) | Surprise-driven deep memory |
| MemMamba (arXiv 2510.03279) | Memory decay metriky |
| HiSPA (arXiv 2601.01972) | State fragility/manipulability |
