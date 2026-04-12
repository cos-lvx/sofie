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
