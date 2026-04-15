# Changelog

Veškeré významné změny v projektu Eleutheria jsou dokumentovány v tomto souboru.

Formát vychází z [Keep a Changelog](https://keepachangelog.com/cs/1.0.0/),
projekt dodržuje [sémantické verzování](https://semver.org/lang/cs/).

---

## [0.4.3] — 2026-04-15

### Opraveno
- **BUG-009** — panic v streaming diff při BPE retokenizaci. V `generate_from_logits`
  je `&full_text[emitted_len..]` byte-level slice, ale BPE tokenizer může při
  novém tokenu re-dekódovat celý `generated` vektor jinak — `full_text` z
  iterace N+1 nemusí být byte-prefix extension z iterace N. Při halucinaci
  v SsmOnly variantě (model bez KV cache generuje UTF-8 multi-byte garbage)
  `emitted_len` ukazoval doprostřed UTF-8 sekvence a Rust panikoval
  `byte index N is out of bounds`.

  Fix: resync na nejbližší nižší UTF-8 char boundary přes
  `str::is_char_boundary`. Final decode je z úplného `generated` vektoru,
  takže žádný data loss; streaming může v ojedinělých případech krátce
  opakovat znaky nebo přeskočit 1–2 znaky při BPE resyncu.

### Důvod patche
Odhaleno při prvním pokusu o pilotní běh `bench-retention --variant all`
na Falcon-H1-1.5B. Blokovalo dokončení matrixu Full × SsmOnly × Cold ×
5 vzdáleností × 5 probes (75 pokusů).

---

## [0.4.2] — 2026-04-15

### Přidáno
- `Sofie::filter_session_to_ssm_only(&mut session)` — vyfiltruje session
  na SSM-only stav: zachová Mamba-2 SSM state, zahodí KV cache + conv state,
  resetuje pozici na 0 a označí session za neinicializovanou. Vyžaduje
  re-init v dalším turnu (nutné, RoPE indexy v KV musí startovat od 0).
- `BenchVariant::all()` — sliceový helper se všemi třemi variantami,
  použito pro `--variant all` v CLI.
- `RetentionBench` — implementace variant `SsmOnly` a `Cold`:
  - **SsmOnly**: po fact + filler se zavolá `filter_session_to_ssm_only`,
    pak otázka přes plnou pipeline (turn 1). Měří, kolik si SSM state
    samostatně zachová informaci, když attention historie zmizí.
  - **Cold**: žádný kontext, jen otázka na čerstvé session. Baseline
    bez paměťového signálu — typicky Fail pro ne-triviální fakta.
- CLI `--variant all` — spustí všechny tři varianty v jednom běhu;
  výstup ve stream loggu obsahuje variant label vedle probe ID.

### Změněno
- `BenchVariant::is_implemented()` odstraněno (dřív gating pro v0.4.1)
- `RetentionBench::run()` přijímá libovolnou kombinaci variant bez kontroly
- Dokumentace `BenchVariant` aktualizována — všechny tři varianty jsou živé
- Test `only_full_is_implemented_in_v041` nahrazen `all_returns_three_variants`

### Odloženo do v0.4.3
- Pilotní běh na Falcon-H1-1.5B (RTX 4050)
- Zápis výsledků do `~/Atlas/Nexus/70-Eleutheria/Research/`
- Aktualizace PLAN.md s empirickými nálezy → vstup pro Core Memory design

---

## [0.4.1] — 2026-04-15

### Přidáno
- Modul `bench/` v `eleutheria-core` — harness pro retention benchmark (Fáze 5 prerekvizita)
  - `RetentionProbe` + 5 vestavěných probes v angličtině (relational, numeric,
    enumeration, preference, multi-attribute) s AND-substring matcherem
  - `FillerCorpus` — 6 neutrálních EN vět pro deterministický token filler,
    cyklicky opakovatelný plán
  - `BenchVariant` enum — `Full` (implementováno), `SsmOnly` a `Cold`
    odloženo do v0.4.2
  - `RetentionBench` orchestrátor — iteruje variants × distances × probes,
    per-probe isolation (čerstvá session pro každý pokus)
  - `BenchReport` — JSON + markdown export (souhrn pass-rate per bucket
    + detailní tabulka); zápis JSON a MD vedle sebe
- `Sofie::inject_turn()` — low-level API pro prefill bez decoding.
  Injektuje kontrolovaný (user, assistant) pár do session, posune stav,
  negeneruje vlastní odpověď. Slouží pro benchmark replay a deterministické
  reprodukování konverzačního stavu.
- CLI subkomand `bench-retention` s flagy `--variant`, `--distances`,
  `--output`, `--notes` (zpětně kompatibilní — bez subkomand běží původní
  REPL/single-shot mód)
- 18 nových unit testů (probe matcher, filler determinismus, variant parsing,
  report round-trip + markdown rendering)

### Zdůvodnění angličtiny v probe obsahu
Falcon-H1-1.5B má slabší češtinu. Pro čisté měření retence stavu (ne
jazykové kapacity) jsou fact + question + filler v angličtině. Dokumentace,
komentáře a CLI output zůstávají česky dle projektových pravidel.

### Odloženo do v0.4.2 / v0.4.3
- `SsmOnly` varianta — vyčištění KV cache přes `StateFilter::ssm_only()`
  před otázkou, pak re-inject do fresh ModelState
- `Cold` varianta — otázka bez kontextu, baseline bez paměťového signálu
- Pilotní běh na Falcon-H1-1.5B s výstupem do Nexus research adresáře

---

## [0.3.4] — 2026-04-12

### Přidáno
- Auto-save session při ukončení REPL do `~/.eleutheria/last_session.safetensors`
- `--resume` flag — pokračování v poslední session
- `/save` bez argumentu ukládá do default session path
- `ensure_session_dir()` — automatické vytvoření `~/.eleutheria/`

### Změněno
- Priorita načtení: `--load-state` > `--resume` > nová session
- StreamingLLM (attention sinks) odsunut z v0.3.4 do budoucí fáze —
  s 128K kontextem je předčasná optimalizace

---

## [0.3.3] — 2026-04-12

### Přidáno
- Token budget monitoring v `SofieSession`
  - `context_limit()` — max pozičních embeddingů z config (1.5B: 128K, 7B: 256K)
  - `context_usage()` — využití kontextu jako poměr (0.0–1.0)
  - `remaining_tokens()` — zbývající tokeny do limitu
  - `kv_cache_bytes()` — odhad velikosti KV cache v VRAM
- Budget enforcement v `send_message()` — cap max_tokens na zbývající kontext,
  chyba při vyčerpání, warning při >75% využití
- `/info` v REPL zobrazuje kontext usage, zbývající tokeny, KV cache odhad
- `max_position_embeddings` pole v `FalconH1Config` (deserializuje se z config.json)
- 2 nové testy: context_usage, kv_cache_bytes

---

## [0.3.2] — 2026-04-12

### Přidáno
- `SofieSession` — živá konverzační session s inkrementálním prefillem
  - `ModelState` přežívá mezi turny — SSM akumuluje, KV cache roste
  - Turn 1: plný pipeline (PersonaInjection → ChatMLAssembly)
  - Turn 2+: delta prefill — jen ChatML wrapping nové zprávy, O(nové_tokeny)
  - Historie konverzace, počítadlo turnů, timestamp
- `Sofie::new_session()` / `resume_session()` / `send_message()` — session API
- REPL mód v CLI (bez `--prompt` = interaktivní konverzace)
  - `/save [cesta]` — uložit state do checkpointu
  - `/info` — informace o session (turny, tokeny, čas)
  - `q` / `quit` / `exit` — ukončení
- `generate_from_logits()` — extrahovaný generate loop (sdílený single-shot i session)
- 3 unit testy pro SofieSession (new, record_turn, from_checkpoint)

### Změněno
- `--prompt` je nyní volitelný — bez něj se spustí REPL

---

## [0.3.1] — 2026-04-12

### Přidáno
- `StateCheckpoint` — serializace a deserializace `ModelState` do safetensors
  - `StateFilter` se třemi presety: `full()`, `core_memory()`, `ssm_only()`
  - `CheckpointMeta` v safetensors `__metadata__` hlavičce (pozice, rozměry, filtr, timestamp)
  - `from_model_state()` — export z GPU na CPU, selektivní dle filtru
  - `save()` / `load()` — safetensors s metadaty na disk a zpět
  - `apply_to_model_state()` — injekce do existujícího stavu (přepíše jen přítomné komponenty)
  - `into_model_state()` — vytvoření čerstvého stavu z checkpointu
  - `inspect()` — metadata bez načítání tensorů
  - `validate_config()` — kontrola kompatibility checkpointu s modelem
- `GenerateResult` struct — `generate_streaming()` vrací text + stav + pozici
- CLI flagy: `--save-state`, `--load-state`, `--state-filter`, `--inspect-state`
- 6 unit testů pro round-trip, selektivní save/load, metadata, validaci

### Změněno
- `generate_streaming()` přijímá volitelný `initial_state` pro state injection
- `chat_streaming()` a `chat()` vrací `GenerateResult` místo `String`
- `LayerState::new()` refaktorován — přijímá `&FalconH1Config` místo 9 parametrů

### Opraveno
- Clippy warnings: `map_or` → `is_some_and`, loop indexing → slice iteration,
  `Default` impl pro `PromptPipeline`, doc-comment indentace

---

## [0.3.0] — 2026-03-05

### Přidáno
- 7-stupňový prompt pipeline (`src/prompt/`)
  - InputClassifier — detekce záměru vstupu (placeholder)
  - PersonaInjection — načtení persony ze TOML
  - TemplateExpansion — substituce šablonových proměnných (datum, čas)
  - ConversationContext — správa konverzační historie (placeholder)
  - MemoryInjection — injekce paměťových fragmentů (placeholder)
  - QualityGate — validace kvality výstupu (placeholder)
  - ChatMLAssembly — formátování do ChatML (`<|im_start|>role\n...`)
- High-level `chat_streaming()` API na `Sofie` structu
- CLI podpora `--persona` argumentu
- TOML konfigurace persony (`persona/sofie.toml`)

---

## [0.2.6] — 2026-03-05

### Přidáno
- Integrace pipeline do `chat_streaming()` v `lib.rs`
- CLI persona podpora — defaultní `persona/sofie.toml`

---

## [0.2.5] — 2026-03-05

### Přidáno
- Placeholder stages: InputClassifier, ConversationContext, MemoryInjection, QualityGate
- Kompletní pipeline řetězec (7 stages)

---

## [0.2.4] — 2026-03-05

### Přidáno
- TemplateExpansion stage — proměnné `{{date}}`, `{{time}}`, `{{weekday}}` v system promptu

---

## [0.2.3] — 2026-03-05

### Přidáno
- PersonaInjection stage + TOML persona loader
- `persona/sofie.toml` — první definice Sofiiny persony

---

## [0.2.2] — 2026-03-05

### Přidáno
- ChatMLAssembly stage — formátování zpráv do ChatML formátu

---

## [0.2.1] — 2026-03-05

### Přidáno
- Prompt pipeline skeleton (`src/prompt/mod.rs`, `types.rs`, `pipeline.rs`)
- `PromptStage` trait + `PromptPipeline` orchestrátor
- `PromptContext` jako sdílený stav průchodu pipeline

---

## [0.2.0] — 2026-03-05

### Přidáno
- `generate_streaming()` API s callback-based token emission
- Diff-based dekódování (emituje jen nový text per token)
- `GenerateControl` enum (Continue/Stop) pro řízení generování

### Opraveno
- BUG-008: BF16 sampling crash — F32 upcast před dělením teplotou

---

## [0.1.0] — 2026-03-04

### Přidáno
- Kompletní Falcon-H1 inference engine
  - Paralelní prefill (`forward_prefill()`) — celá sekvence v jedné GPU operaci
  - Recurrent decode (`forward()`) — token-by-token generování
  - Mamba-2 SSM implementace s muP multiplikátory
  - GQA attention s RoPE
  - SwiGLU MLP
  - RmsNorm + RmsNormGated
  - Safetensors weight loading
  - State management (SSM state, conv state, KV cache)
- CLI interface s model selection (1.5b/7b/custom path)
- Podpora BF16 na CUDA, F32 fallback na CPU

### Opraveno
- BUG-007: RmsNormGated logika invertovaná — gate/norm pořadí opraveno
- BUG-001 až BUG-006: Průběžné opravy během vývoje (conv1d duplikace, muP aplikace, weight keys)
