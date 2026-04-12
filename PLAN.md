# Plán — Eleutheria

> Poslední aktualizace: 2026-04-12

## Dokončeno

- [x] **Akt I: Hlas** — Sofie umí mluvit
  - Fáze 1 — Inference engine (v0.1.0)
  - Fáze 2 — Streaming output (v0.2.0)
  - Fáze 3 — Prompt pipeline (v0.3.0)
- [x] **Research: Memory Architecture** — Deep Research dokončen 2026-04-12
  - SSM state persistence je mechanicky proven (mamba.c, SGLang, PR #488)
  - Trénované stavy >> zachycené stavy (RWKV, ACL 2025)
  - Třívrstvá architektura: Core Memory + Session Memory + Episodic Memory
  - Echo embeddings pro self-retrieval bez externího modelu
  - Výstup: `~/Atlas/Nexus/70-Eleutheria/Research/SSM state persistence...md`

## Aktuální: Fáze 4 — Session Memory (v0.4.0)

Implementace po krocích:

### v0.3.1 — SSM state serializace ✅
- [x] `StateCheckpoint` — save/load s metadaty, StateFilter, validace
- [x] 6 unit testů (round-trip, selektivní save/load, metadata, config validace)
- [x] CLI: `--save-state`, `--load-state`, `--state-filter`, `--inspect-state`

### v0.3.2 — Multi-turn REPL
- [ ] Interaktivní smyčka v CLI (ne single-shot)
- [ ] `SofieState` struct — živý stav oddělený od ModelState
  (conversation history, metadata, timestamps)
- [ ] State přetrvává mezi zprávami v rámci session

### v0.3.3 — ConversationContext stage
- [ ] Reálná logika místo placeholder — injekce předchozích zpráv
- [ ] Token budget management (počítání tokenů, ořezávání historie)
- [ ] InputClassifier s heuristikami (detekce otázka/instrukce/pokračování)

### v0.3.4 — State checkpointing
- [ ] Automatické uložení state na hranicích konverzace
- [ ] Obnovení session z checkpointu (`--resume`)
- [ ] StreamingLLM pro attention KV cache (attention sinks + sliding window)

### v0.3.5 — Benchmark retence
- [ ] Měření informační retence v SSM state Falcon-H1-7B
  (replikace Hossain et al. — ROUGE F1 na různých délkách)
- [ ] Porovnání s/bez state persistence
- [ ] Dokumentace výsledků do Research/

**Po v0.3.5 → v0.4.0** (Session Memory kompletní)

## Příští kroky

- [ ] **Fáze 5** — Core Memory + Episodic Memory (v0.5.0)
  - State tuning přes backprop (RWKV metodologie)
  - Echo embeddings pro self-retrieval
  - PostgreSQL + pgvector na Mnémosyné
- [ ] **Fáze 6** — Vnitřní monolog + Konsolidace (v0.6.0)
  - Potřebuje další Deep Research (Mamba-CL, SRC, Titans)

## Poznámky

- Safetensors save: Candle wrapper nepodporuje metadata, řešeno přímou
  závislostí na `safetensors` crate (viz SOL-007)
- Pro state tuning (Fáze 5) bude potřeba backprop podpora v Candle —
  ověřit stav `candle-nn` gradient computation
- Research materiály: `~/Atlas/Nexus/70-Eleutheria/Research/`
