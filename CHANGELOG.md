# Changelog

Veškeré významné změny v projektu Eleutheria jsou dokumentovány v tomto souboru.

Formát vychází z [Keep a Changelog](https://keepachangelog.com/cs/1.0.0/),
projekt dodržuje [sémantické verzování](https://semver.org/lang/cs/).

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
