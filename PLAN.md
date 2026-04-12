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

### v0.3.2 — Multi-turn REPL ✅
- [x] `SofieSession` — živý stav s inkrementálním prefillem (přístup B)
- [x] Session API: `new_session()`, `resume_session()`, `send_message()`
- [x] REPL mód v CLI (bez `--prompt`), příkazy `/save`, `/info`, `q`
- [x] `generate_from_logits()` — extrahovaný sdílený generate loop

### v0.3.3 — Token budget monitoring ✅
- [x] `context_limit`, `context_usage()`, `remaining_tokens()`, `kv_cache_bytes()`
- [x] Budget enforcement v `send_message()` — cap, chyba, warning
- [x] `/info` zobrazuje kontext usage a KV cache odhad
- [x] `max_position_embeddings` v FalconH1Config

### v0.3.4 — Automatic checkpointing + resume ✅
- [x] Auto-save session do `~/.eleutheria/last_session.safetensors` při ukončení
- [x] `--resume` flag pro pokračování v poslední session
- [x] `/save` bez argumentu → default path
- [ ] ~~StreamingLLM~~ — odsunuto, předčasné s 128K kontextem

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
