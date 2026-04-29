# Memory — Eleutheria

Chronologický záznam implementačních cyklů.

---

## 2026-04-29 | v0.5.0-alpha.14 — Save/Load trénované Core Memory

Trénovaná Core Memory přežívá restart procesu. Vyřešený "trenuju → loss
klesá → zavřu okno → ztracená Sofie" cyklus alpha.11–13 končí.

**Nový modul `training/core_memory_io.rs`** — `CoreMemoryArtifact`
serializační formát: per-layer F32 tensory (native dtype `Var`) +
`__metadata__` hlavička s `kind=core_memory_trained`, telemetrií
tréninku (`training_steps`, `best_loss`, `final_loss`, `notes`) a
strukturními rozměry (num_layers, n_heads, headdim, d_state).

**Klíčové rozhodnutí — proč ne `StateCheckpoint::core_memory()` filter:**
1. StateCheckpoint je v runtime dtype (BF16 na CUDA); native Var je F32 —
   round-trip BF16↔F32 by zhoršil precision.
2. StateCheckpoint má `position` field (sémanticky pro session);
   trénovaná Core Memory pozici nemá — je to plugin, ne kontinuace.
3. Conv state v StateCheckpoint je krátkodobé okno (poslední d_conv
   tokeny); trénovaná Core Memory ho nepotřebuje (vždy startujeme nulovou
   conv při fresh session).
4. Sémantická separace: `kind=core_memory_trained` umožňuje budoucím
   nástrojům rozeznat, co se vlastně načítá. Zabrání záměně.

**Sofie API:** `attach_core_memory(art)` + `detach_core_memory()` +
`has_core_memory()` + `core_memory_meta()`. `new_session()` a
single-shot `generate_streaming` (bez `initial_state`) aplikují per-layer
init_states z artefaktu místo nul. **Resume session** (`load_state`,
`--resume`) Core Memory **ignoruje** — uložená session má vlastní
evolved state, který by neměl být přepsán statickým artefaktem.

**CLI:**
- `--core-memory <path>` (explicit)
- `--no-core-memory` (vypne i auto-discovery)
- `--inspect-core-memory <path>` (symetrie s `--inspect-state`)
- `train-core-memory --output <path> --notes <text>` (persistence po
  tréninku)
- Auto-discovery `~/.eleutheria/core_memory.safetensors` — pokud
  uživatel neuvede flagy a soubor existuje, auto-attach.

**`into_stack(config, device)` API** — re-konstrukce `CoreMemoryStack`
s čerstvými `Var`-y pro **resume tréninku** (alpha.15+). Test ověřuje,
že vars_owned vrací 24 (1.5B) trainable handles po round-tripu.

**Čísla:** 84 unit testů (+7 oproti alpha.13: round_trip,
apply_to_state, inspect, incompatible_config_rejected, into_stack,
load_rejects_wrong_kind, metadata_display). Clippy clean, fmt OK.

**Co dál (alpha.15):** resume training s persistovaným AdamW state +
step_idx + epoch (vedle artefaktu jako `core_memory.optim.safetensors`).
Production training run na 1.5B s `law_pack` + `programming_pack`.
Validace přes re-run retention benchmarku — SsmOnly pass-rate musí
vyskočit z 0 % (kritický důkazní bod Fáze 5).

---

## 2026-04-29 | v0.5.0-alpha.13 — Sub-layer checkpointing + memory-leak fix

**KI-005 vyřešena.** Multi-layer Core Memory training na RTX 4050 6 GB
nyní funguje stabilně.

Sub-layer chunking: `FalconH1Layer::forward_chunk_branches` (chunk α —
pre_norm + parallel SSM/attention + residual1) + `forward_chunk_mlp`
(chunk β — post_norm + SwiGLU MLP + residual2). Memory peak per layer =
max(α, β) místo sum. `FalconH1Model::forward_layer_branches` /
`forward_layer_mlp` per-layer-idx wrappery.

**Klíčové zjištění:** alpha.12 OOM nebyla primárně intra-layer
activations, ale **memory-leak v Phase 3 reverse sweep** — saved
tensorů `Vec` drží Arc references na GPU storage po celou Phase 3,
plus `final_grads` z Phase 2 `loss.backward` držel intermediate refs
lm_head workspace (~700 MB skok). Fix: `mem::replace` saved tensorů
progresivně po konzumaci, scope-bounded `final_grads` ihned po extract
`grad_target`, `phase3_layer_reverse` helper s lokálními scope pro
chunk α/β.

**Diagnostika:** `ELEUTHERIA_CHECKPOINT_DEBUG=1` env probe — per-fáze
`nvidia-smi` reading. Bez něj OOM vypadal jako hardware limit, s ním
se ukázal lineární růst +64-96 MB per Phase 3 iterace = leak.

**Empirie na CUDA RTX 4050 6 GB:** seq_len=4 batch=1 grad_accum=1 —
~10 s/step (2× rychlejší než CPU 19 s/step), peak memory **5647 MB
konstantní napříč 24 vrstev**, loss 5.45 → 1.83 best (pod random
baseline ln(vocab)≈11.09).

**Limit:** `grad_accum > 1` stále padá na 6 GB. Pro production
training použij `--grad-accum 1` s zvýšeným batch_size. seq_len > 4
také hraniční. Větší VRAM (alpha.14 / Gaia) odblokuje pohodlnější
parametry.

77 unit testů, clippy clean, zero warnings. SOL-013 dokumentuje
progressive drop pattern v SOLUTIONS.md.

**Co dál (alpha.14):** save/load trained Core Memory přes
`StateCheckpoint` (filter `core_memory` už existuje), auto-load
v `Sofie::load`, resume training (init_states + AdamW state + step_idx),
production training run s law_pack + programming_pack.

---

## 2026-04-28 | v0.5.0-alpha.12 — Per-layer gradient checkpointing

**KI-006 vyřešena.** Custom 3-fázový gradient checkpointing pro
multi-layer Core Memory training. Synthetic loss trick (`sum(out *
grad_target)`) propaguje libovolný tensor gradient skrz chunk hranici
v Candle (které nemá `torch.utils.checkpoint` ekvivalent).

Phase 1 — no-grad forward sweep s `Tensor::detach()`, save state
snapshotů. Phase 2 — re-forward `final_norm + lm_head` s autograd,
cross_entropy loss, `loss.backward()`. Phase 3 — reverse layer sweep:
restore stavu, fresh `Var::from_tensor(saved_input)`, re-forward s
autograd, synth.backward vrátí `d_init_state[i]` + `grad_target` pro
chunk i-1.

**FalconH1Model per-layer API:** `embed`, `forward_layer`, `final_head`,
`num_layers`. `LayerState::snapshot` (deep copy 4 tensorů). CLI
`--checkpoint`, `TrainingConfig::checkpoint`.

**Empirie:** CPU 1.5B F32 seq_len=8 batch=1 grad_accum=2 — **19 s/step**
(2.5× rychleji než alpha.11 baseline 48 s/step — menší memory traffic
vyhrává nad 2× compute z re-forward). Loss 7.11 → 3.70 best, pod random
baseline.

**KI-005 částečně:** RTX 4050 6 GB stále OOM během Phase 3 — per-layer
nebyla dost agresivní granulrita. Sub-layer chunking je práce pro
alpha.13.

77 unit testů (+1 nový — `checkpointed_forward_backward_runs_on_short_seq`),
clippy clean. SOL-012 v SOLUTIONS.md dokumentuje synthetic loss trick.

**Paralelní track 2026-04-28:** dataset prep — 67 reasoning chains (43
NS judikátů + 24 KQS programming SOLUTIONS). 186k tokenů celkem,
ratio law:programming 1.75:1. Pack: `dataset/training/`. Manifest +
generátory v `dataset/scripts/`.

---

## 2026-04-17 | v0.5.0-alpha.11 — Training loop + dataset loader

Produkční variant single-iteration smoke testu. Core Memory se umí
trénovat na textovém korpusu.

- `TokenDataset` (`training/dataset.rs`) — tokenize + chunk + shuffle
  s deterministic xorshift64 PRNG (žádná externí `rand` dep)
- `TrainingConfig` + `train_core_memory` (`training/train.rs`) — epochs
  × batches × gradient accumulation → AdamW step, halt na NaN,
  `tracing::info!` logging
- `Sofie::tokenizer_ref()` accessor
- CLI subkomanda `train-core-memory --dataset <path> ...`

**Ověřeno na 1.5B CPU F32** (smoke_corpus 475 tokens, seq_len=8,
batch=1, grad_accum=2):
```
step 5: loss=5.71, best=4.64
  (random baseline ln(65537)≈11.09 → pod baseline, signifikantní signál)
```
Loss klesá monotónně. Training loop funkční.

**CPU F32 je 48s/step** — full training na 100k tokens trvá dny.
Alpha.12 prioritně: gradient checkpointing → odblokování CUDA.

76 unit testů (+10: 7 dataset + 2 train + 1 extra). Zero warnings.

## 2026-04-17 | v0.5.0-alpha.10 — Multi-layer CoreMemory + cross-entropy

První produkční building block Fáze 5. Infrastruktura pro multi-layer
state tuning:

- `CoreMemoryStack` — `Vec<CoreMemory>` pro všech 24 Mamba-2 vrstev,
  `inject_into_state`, `vars_owned` pro AdamW
- `cross_entropy_next_token(logits, input_ids)` — shift-by-one LM loss
  s F32 upcast pro log_softmax (BF16 nestabilní)
- `Sofie::smoke_train_core_memory_multilayer` + CLI
  `train-core-memory-multi`

Ověřeno na Falcon-H1-1.5B CPU F32 (seq_len=2, lr=1e-3, clip=1.0):
loss 21.5 (vs. baseline ln(vocab)≈11.09), total grad clipped na 1.0,
všech 24 vrstev dostalo gradient, per-layer grad roste s hloubkou
(L0=1.6e-2 → L23=4.38, Peri-LN pattern). 15 s wall time na CPU.

**CUDA OOM** na RTX 4050 (6 GB) pro multi-layer backward graph —
full forward přes 24 vrstev + 65537 vocab nezvládne. alpha.11 řeší
gradient checkpointing / accumulation / větší VRAM (Gaia).

66 unit testů (+8: 4 loss + 4 CoreMemoryStack). Zero warnings.

## 2026-04-17 | v0.5.0-alpha.9 — BUG-010 vyřešen (silu backward fix)

**Diagnostika z alpha.8 identifikovala root cause:** lokální
`silu(x) = x * recip(1 + exp(-x))` v `mixer.rs` a `norm.rs` produkuje NaN
gradient pro extrémně záporné x (exp overflow → recip(Inf)=0 → 0*Inf=NaN
v backward chain). Hluboké vrstvy Falcon-H1 produkují hodnoty ±100,
kde tato naivní implementace exploduje.

**Fix:** delegace na `candle_nn::ops::silu` (native `Tensor::silu()`
s numericky stabilním backward kernelem).

**Ověřeno na Falcon-H1-1.5B (CUDA RTX 4050):**
- L22 cut=22: grad 2.85 (před i po)
- L22 cut=23: **NaN → 9.80**
- L22 cut=full (přes lm_head): **NaN → 1.74**

Fáze 5 (Core Memory) odblokovaná. Další krok: alpha.10 multi-layer
training loop.

58 unit testů (+3 silu micro testy, +5 trace z alpha.8).

## 2026-04-17 | v0.5.0-alpha.8 — Instrumentovaný forward pass (BUG-010)

Přidána dvojí diagnostika pro lokalizaci op zodpovědné za NaN gradient
v backward přes více vrstev:

1. **Forward tensor stats** — `training/trace.rs` thread-local sink,
   `probe(&t, label)` v 30+ bodech forward pass (layer, mixer, attention).
   Statistiky: `abs_max`, `abs_min_nonzero`, `mean`, `l2`, `has_nan`,
   `has_inf`. Detach před výpočtem — neváže autograd graf.
2. **Sub-layer cut-at-component** — enum `LayerStop`
   (`AfterPreNorm / AfterSsm / AfterAttn / AfterResidual1 / AfterPostNorm /
   AfterMlpGate / AfterMlpSiluMul / AfterMlpDown / Full`),
   `FalconH1Layer::forward_until`, `FalconH1Model::forward_up_to_layer_with_stop`.
   Binary search op uvnitř jedné vrstvy.

CLI: `--cut-at-component <name>` a `--trace` na `train-core-memory-smoke`.
Unified API `Sofie::smoke_train_core_memory_component` vrací
`(SmokeTrainResult, Option<Vec<TraceEntry>>)`.

**Další krok (alpha.9):** Ondra spustí smoke test s `--trace` na L0
i problematických L20–L22, identifikujeme op s extrémním dynamickým
rozsahem. Primární podezřelí: local `silu` v mixer.rs/norm.rs pro velmi
záporné x (recip(Inf) backward), attention softmax pro extreme logits,
conv1d backward.

55 unit testů (přibylo 5 pro trace modul). Zero warnings, zero clippy.

## 2026-03-03 | Scaffold

Kompletní scaffold Falcon-H1 engine: config parser, state management, normalizace,
attention (GQA + RoPE), mixer (Mamba-2 SSM), layer architektura, model struct.
muP multiplikátory jako f64 konstanty. Weight key naming konvence z analýzy safetensors.

## 2026-03-04 | v0.1.0 — Parallel Prefill

Implementace `forward_prefill()` — celá sekvence v jedné GPU operaci místo
token-by-token. Causal conv1d + sekvenční SSM scan. Opraven BUG-007 (invertovaná
RmsNormGated logika) — po opravě koherentní český text.

## 2026-03-05 | v0.2.0 — Streaming Output

`generate_streaming()` s callback-based token emission. Diff-based dekódování
(emituje jen nový text per token). Opraven BUG-008 (BF16 sampling crash → F32 upcast).
Vyčištěny unused variable warnings.

## 2026-03-05 | v0.3.0 — Prompt Pipeline

Nový modul `src/prompt/` — 7-stage pipeline architektura:
InputClassifier → PersonaInjection → TemplateExpansion → ConversationContext →
MemoryInjection → QualityGate → ChatMLAssembly.

High-level `chat_streaming()` API, TOML persona loading, template variable substitution,
CLI `--persona` argument. Placeholder stages připraveny pro budoucí fáze.

## 2026-04-12 | Scaffolding

Vytvořeny živé dokumenty: CLAUDE.md, CHANGELOG.md, ROADMAP.md, MEMORY.md, PLAN.md,
KNOWN-ISSUES.md, SOLUTIONS.md, BUGS.md. Nastavena spolupráce podle vzoru Tessera/Vesna.

## 2026-04-12 | v0.4.0 — Session Memory milestone uzavřen

Fáze 4 (Session Memory) kompletní. Čtyři implementační cykly v jedné session:
- v0.3.1: StateCheckpoint (safetensors + metadata, 3 filtry)
- v0.3.2: SofieSession + inkrementální prefill (přístup B — delta tokeny)
- v0.3.3: Token budget monitoring + context tracking
- v0.3.4: Auto-save + --resume

Klíčová architektonická rozhodnutí:
- Přístup B (inkrementální prefill) místo A (re-prefill celé konverzace)
- ConversationContext stage v pipeline je obsoletní pro injekci historie —
  s přístupem B je session state = kontext
- Benchmark retence odsunut do v0.5.x — relevantní až pro Core Memory design
- StreamingLLM odsunut — předčasný s 128K kontextem

Ověřeno na živém modelu (Falcon-H1-1.5B, RTX 4050, CUDA 13.2). Model si
pamatuje kontext přes turny. Inkrementální prefill funguje (16 tokenů delta
místo 223 full re-prefill).

Příští fáze: v0.5.0 — Core Memory (trénovaný SSM state) + Episodic Memory.
Prerekvizita: benchmark retence + Deep Research pro state tuning v Candle.

## 2026-04-12 | v0.3.4 — Automatic Checkpointing + Resume

Auto-save session do `~/.eleutheria/last_session.safetensors` při ukončení REPL.
`--resume` flag načte poslední session. Priorita: `--load-state` > `--resume` > nová.
`/save` bez argumentu ukládá do default path. StreamingLLM odsunut — předčasný s 128K.

## 2026-04-12 | v0.3.3 — Token Budget Monitoring

Context tracking v SofieSession — `context_limit`, `context_usage()`, `remaining_tokens()`,
`kv_cache_bytes()`. Budget enforcement v `send_message()` — cap max_tokens, chyba při
vyčerpání, warning >75%. `/info` v REPL zobrazuje kontext a KV cache odhad.

Přidáno `max_position_embeddings` do FalconH1Config (1.5B: 128K, 7B: 256K).
Přehodnocení ConversationContext stage: s přístupem B (inkrementální prefill) je
injekce historie obsoletní — session state JE kontext. Stage přeformulována na
token budget monitoring, ne na injekci historie.

## 2026-04-12 | v0.3.2 — Multi-turn REPL + SofieSession

Nový modul `session.rs` — `SofieSession` drží `ModelState` mezi turny. Architektonické
rozhodnutí: přístup B (inkrementální prefill). Turn 1 prochází plným pipeline,
Turn 2+ prefilluje jen delta (ChatML wrapping nové zprávy). SSM state akumuluje
kontext přirozeně, KV cache roste — O(nové_tokeny) per turn místo O(všechny_tokeny).

Session API na Sofie: `new_session()`, `resume_session()`, `send_message()`.
Generate loop extrahován do `generate_from_logits()` (sdílený single-shot i session).

REPL mód: bez `--prompt` se spustí interaktivní smyčka. Příkazy `/save`, `/info`, `q`.

## 2026-04-12 | v0.3.1 — SSM State Serializace

Nový modul `falcon_h1/checkpoint.rs` — `StateCheckpoint` jako přenosový formát
mezi živým `ModelState` a diskem (safetensors s `__metadata__`).

Tři filtry: `full()` (session resume), `core_memory()` (SSM+conv pro state tuning),
`ssm_only()` (experimenty). CLI flagy: `--save-state`, `--load-state`, `--inspect-state`.

`generate_streaming()` refaktorován — přijímá volitelný initial state, vrací
`GenerateResult` (text + state + pozice). `LayerState::new()` zjednodušen na
`&FalconH1Config`. Opraveny pre-existující clippy warnings (nulová tolerance).

6 unit testů: round-trip full, core_memory, metadata, config validace, selektivní
apply, filter labels. Safetensors 0.6 jako přímá závislost pro metadata support.

## 2026-04-12 | Deep Research — Memory Architecture

Dokončen Deep Research přes Claude Desktop: "SSM State Persistence and Memory
Architecture for Hybrid Mamba-Attention Models."

Klíčové nálezy:
- SSM state serializace je mechanicky proven (mamba.c, SGLang, Mamba PR #488)
- Trénované initial states >> zachycené states (RWKV State Tuning, State-offset
  Tuning ACL 2025 — 0.01% parametrů, výkon srovnatelný s full fine-tuning)
- Informace v SSM state má poločas ~14 tokenů (α≈0.95), ale d_state=256 u
  Falcon-H1-7B dává výrazně lepší retenci
- Echo embeddings — self-retrieval přes Falcon-H1 bez externího modelu
- Mamba-CL null-space projection pro konsolidaci bez zapomínání
- Titans surprise-driven memory pro rozhodování co zapamatovat

Rozhodnutí: Třívrstvá architektura paměti (Core Memory = trénovaný state,
Session Memory = checkpointing, Episodic Memory = echo embeddings + pgvector).
ROADMAP přepsán z feature-driven na capability-driven (Hlas → Pamatování →
Myšlení → Tělo). Mantra: ne pragmatické řešení, ale revoluční.

## 2026-04-15 | v0.4.1 — Retention Bench Harness

První ze tří patchů prerekvizity pro Fázi 5 (Core Memory). Postavena infra
pro behaviorální test SSM state retence — kolik si SSM pamatuje přes N tokenů
vzdálenosti od seedu.

Nový modul `crates/eleutheria-core/src/bench/`:
- `probe.rs` — `RetentionProbe` + 5 vestavěných probes v angličtině
  (relational / numeric / enumeration / preference / multi-attribute),
  AND-substring matcher case-insensitive
- `filler.rs` — `FillerCorpus` se 6 neutrálními EN větami, cyklicky
  opakovatelný `FillerPlan`
- `variant.rs` — `BenchVariant` (Full | SsmOnly | Cold); v0.4.1 jen Full
- `harness.rs` — `RetentionBench` orchestrátor, per-probe isolation
  (čerstvá session pro každý pokus)
- `report.rs` — `BenchReport` s JSON + markdown výstupem (souhrn pass-rate
  per bucket + detailní tabulka)

Rozhodnutí EN místo CZ: Falcon-H1-1.5B má slabší češtinu, šum z jazyka
by maskoval signál retence. Dokumentace a CLI output zůstávají česky.

Nové low-level API: `Sofie::inject_turn(&mut session, user, assistant)` —
prefill bez decoding, injektuje forced assistant reply. Drží stejný
invariant jako `send_message` (stav za obsahem, `<|im_end|>` zatím
nekonzumován).

CLI subkomand `bench-retention --variant --distances --output --notes`,
zpětně kompatibilní (bez subkomand = původní REPL/single-shot).

17 nových unit testů (matcher, filler determinismus, variant parsing,
JSON round-trip, markdown rendering). Celkem 30 testů prochází.

Odloženo do v0.4.2/v0.4.3: SsmOnly + Cold varianty, pilotní běh na
Falcon-H1-1.5B s výstupem do Nexus research.

Paralelně otevřeno: Deep Research pro backprop v Candle (blocker pro state
tuning ve Fázi 5).

## 2026-04-15 | v0.4.2 — SsmOnly + Cold varianty

Druhý ze tří patchů Fáze 5 prerekvizity. Otevřeny zbývající dvě varianty
retention benchmarku.

Nové API: `Sofie::filter_session_to_ssm_only(&mut session)` — round-trip
přes StateCheckpoint s `StateFilter::ssm_only()` filtrem, vytvoří čerstvý
ModelState s SSM komponentou injektovanou (KV + conv vynulovány). Resetuje
pozici na 0 a označí session za neinicializovanou — RoPE indexy v KV musí
startovat od 0, takže další turn musí jít přes plnou pipeline jako turn 1.

`SofieSession::replace_state(state, position, mark_uninitialized)` —
interní `pub(crate)` helper. `turn_count` a `history` zůstávají zachované
pro audit; změna stavu není výmaz konverzace, jen restrukturalizace paměti.

Harness::run_one teď match-uje variant:
- **Full**: standard (fact, filler, otázka přes delta turn)
- **SsmOnly**: fact + filler, pak filter_session_to_ssm_only, pak otázka
  přes plnou pipeline (turn 1)
- **Cold**: žádný kontext, jen otázka na čerstvé session

CLI `--variant all` přes `BenchVariant::all()` slice helper. Stream log
obsahuje variant label vedle probe ID pro snadnou orientaci.

Odstraněno: `BenchVariant::is_implemented()` (gating už zbytečný).
Test `only_full_is_implemented_in_v041` → `all_returns_three_variants`.

30 testů prochází (stejné celkové číslo, jen substituce testu).

## 2026-04-15 | v0.4.3 — BUG-009 streaming panic fix

První pilotní spuštění `bench-retention --variant all` na 1.5B odhalilo
panic v `generate_from_logits` během SsmOnly variantu: model bez KV cache
halucinuje česky (`Knyž Lomího, řeberlí...`), UTF-8 multi-byte chars +
BPE retokenizace = `&full_text[emitted_len..]` spadne na
`byte index N is out of bounds`.

Root cause: BPE tokenizer může při novém tokenu re-dekódovat celý
`generated` vektor jinak než minule. Naivní diff-based streaming
předpokládal prefix-monotonicity, která neplatí.

Fix: resync `emitted_len` na nejbližší nižší UTF-8 char boundary přes
`str::is_char_boundary`. V patologických případech (model halucinuje)
se streaming může krátce desynchronizovat — final decode je vždy
z úplného `generated` vektoru, takže žádný data loss.

Odhaleno při prvním reálném běhu, před tímto patchem nebyl v ničem
viditelný — single-shot i REPL s normálním modelovým výstupem nikdy
neretokenizoval na UTF-8 boundary.

## 2026-04-15 | v0.4.4 — bench bez persony default

Ondra položil zásadní otázku: nezkresluje nám persona výsledky retention
benchmarku? Odpověď: ano, čtyřmi mechanismy.

1. Persona je česky, probes jsou EN → jazyková inkonzistence = šum v SSM
   komprimovaném kontextu
2. "Mysli v krocích" instrukce prodlužuje odpovědi, klíčová slova často
   padají mimo 80-token answer budget
3. 1.5B model může ignorovat meta-instrukci "odpovídej v jazyce, ve kterém
   ti bylo napsáno" → odpoví česky → AND-matcher hledající EN substrings
   false-negative i když model fakt pamatuje
4. ~180 tokenů persony posouvá absolute position, zkresluje měření krátkých
   vzdáleností (50, 200 tokenů)

Fix: `bench-retention --with-persona` je opt-in flag, default je vypnutý.
Bench běží s čistým ChatML `<|im_start|>user\n{fact/filler/question}<|im_end|>`
bez system promptu → čistý model-level SSM capacity measurement.

Implementace: `matches!(args.command, Some(Command::BenchRetention(ba)) if !ba.with_persona)`
preempt-uje načtení persony při Sofie::load. REPL a single-shot mód zůstávají
nedotčené.

Architectonická poznámka: Core Memory (Fáze 5) bude nahrazovat persona
system prompt trénovaným initial SSM state. Bench bez persony je tedy i
správný srovnávací baseline pro budoucí produkční cíl.

Pilot matrix na Falcon-H1-1.5B čekal na tento patch — spuštění odložen
o jeden cyklus.

## 2026-04-16 | v0.4.5 — Pilot report + prerekvizita Fáze 5 uzavřena

Ondra spustil `bench-retention --variant all` na Falcon-H1-1.5B (RTX 4050).
75 pokusů (3 varianty × 5 vzdáleností × 5 probes) doběhlo čistě, JSON + MD
zapsán do Nexus research. Čísla jsou dramatická a jasná:

**Full** (SSM + KV + conv): 100 % do 500 tok, 80 % @ 1 k, 40 % @ 2 k.
Graceful degradation, čísla křehká první (`7429` padá na 1 k), proper nouns
a enumerace na 2 k. Přežívají silné sémantické kotvy (`multiattr_helion`
s "Professor" + "observatory" + "built" + "1893").

**SsmOnly** (KV + conv vyčištěné, jen SSM zachován): **0 % na všech
vzdálenostech včetně 50 tokenů**. Žádná výjimka, žádný probe. Zachycený
SSM state samostatně nenese diskrétní fakta — potvrzuje RWKV empirický
nález a Hossain et al. teoretickou predikci (α≈0.95, poločas ~14 tokenů).

**Cold** (žádný kontext): konstantních 20 % = jeden false positive
`preference_linh`, protože matcher hledal jen `"tea"` a model bez kontextu
generuje "I'd recommend tea". Opraveno v v0.4.5: matcher teď vyžaduje
`"linh"` + `"tea"`. Nový test `preference_linh_requires_name_and_drink`.

Architektonický závěr (klíčový vstup pro Fázi 5):
**Core Memory MUSÍ být trénovaný initial state, ne captured.** Naivní
implementace "ulož state po persona promptu a reuse" **nefunguje** na
naší architektuře. Musíme skutečně gradient-optimalizovat initial state.
To mapuje na RWKV State Tuning / State-offset Tuning workflow dle
`reference_candle_backprop.md`.

Research dokument zapsán:
`~/Atlas/Nexus/70-Eleutheria/Research/SSM_retention_findings_2026-04-15.md`
(setup, výsledky, interpretace, dopady pro Fázi 5, limitace, další kroky).

**Prerekvizita Fáze 5 kompletní.** Next: v0.5.0-alpha1 autograd bring-up
na 1.5B, jedna vrstva, sekvenční scan.

## 2026-04-16 | v0.5.0-alpha.1 — Core Memory autograd bring-up

Začátek Fáze 5. První skutečný krok k trénovanému initial SSM state.

**Překvapivý nález při průzkumu:** naše `forward_prefill` v `mixer.rs:382`
je už **sekvenční scan** (prostá smyčka přes tokeny), ne chunked SSD.
Research agent verdict "YELLOW kvůli chunked scan backward" byl
overcautious — chunked jsme nikdy neimplementovali. To autograd workflow
dramaticky zjednodušuje.

Nový modul `crates/eleutheria-core/src/training/`:
- `CoreMemory` — drží `candle_core::Var` pro initial SSM state jedné
  vrstvy, tvar `[n_heads, headdim, d_state]`, F32. Dva init módy:
  `zeros()` (matching ModelState::new) a `randn_small()` (std=0.01).
- `Sofie::smoke_train_core_memory(seq_len, layer_idx, lr)` —
  jedna iterace forward + backward + AdamW step. Reportuje
  gradient L2 normu, delta init_state, loss, wall time.

Kritická otázka: **zero init** dává zero gradient? Multiplikativní
SSM rekurze `h' = dA·h + dB⊗x`: s `h=0` derivace vůči `h` je `dA`
(non-zero z `A_log`), ale `x` z input embeddings je non-zero, takže
gradient by měl téct. Přesto `randn_small` je bezpečnější default.

Accessor API na `Sofie`:
- `device_ref()`, `dtype_ref()`, `new_model_state()`, `model_forward()`
- Trainable Var injektujeme do `state.layers[i].ssm_state` přes
  `to_dtype(bf16)` — autograd-aware konverze, gradient protéká

CLI: `eleutheria --cuda train-core-memory-smoke --seq-len 10 --layer-idx 0`
→ reportuje PASS / FAIL s diagnostickou tabulkou.

34 testů prochází (+3 pro CoreMemory).

**Co dál (alpha.2):** multi-layer init states, cross-entropy loss,
skutečný training loop přes epochs, save/load. Pak alpha.3 — dataset
generation a tréning na Sofie identity.

**Ondra spustí smoke test na 1.5B** — ověří, že gradient skutečně
dotéká, peak VRAM nepřeskočí 6 GB, a wall time je rozumný.

Paralelně dorazil Deep Research na backprop v Candle — verdict YELLOW:
autograd existuje (`Var`, `GradStore`, `AdamW`, kanonický MNIST training
loop), ale náš chunked Mamba-2 SSD scan není testovaný na backward path.
Mitigace pro Fázi 5: použít sekvenční scan (jako mamba-minimal v Candle)
místo chunked, alespoň pro autograd bring-up. Reference: RWKV-LM,
furiosa-ai/ssm-state-tuning. Plný report uložen do auto-memory.
