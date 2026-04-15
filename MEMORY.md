# Memory — Eleutheria

Chronologický záznam implementačních cyklů.

---

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

Paralelně dorazil Deep Research na backprop v Candle — verdict YELLOW:
autograd existuje (`Var`, `GradStore`, `AdamW`, kanonický MNIST training
loop), ale náš chunked Mamba-2 SSD scan není testovaný na backward path.
Mitigace pro Fázi 5: použít sekvenční scan (jako mamba-minimal v Candle)
místo chunked, alespoň pro autograd bring-up. Reference: RWKV-LM,
furiosa-ai/ssm-state-tuning. Plný report uložen do auto-memory.
