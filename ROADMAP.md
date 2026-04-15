# Roadmap — Eleutheria

> Poslední aktualizace: 2026-04-15

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

**Prerekvizity (rozdělené na v0.4.1–v0.4.3):**
- [x] **v0.4.1** — benchmark harness (modul `bench/`, 5 probes v EN, filler,
  CLI subkomand, `Sofie::inject_turn` API, JSON + MD report)
- [x] **v0.4.2** — `SsmOnly` a `Cold` varianty (`filter_session_to_ssm_only`,
  `--variant all`)
- [ ] **v0.4.3** — pilotní běh na Falcon-H1-1.5B, výsledky do Nexus research

**Implementace:**
- [ ] State tuning infrastruktura — backpropagation přes Candle pro
  optimalizaci initial SSM state
- [ ] Core Memory training — natrénovat initial state kódující Sofiinu
  identitu, hodnoty, znalosti o Ondrovi. Trénovací data: existující
  konverzace, Bootstrap.md, Identity.md
- [ ] Core Memory loading — při startu session načíst trénovaný initial
  state místo prázdného (nula → natrénovaný stav)
- [ ] Echo embeddings — implementace self-retrieval přes Falcon-H1:
  vstup se opakuje 2×, embedding z druhého průchodu
- [ ] Episodic Memory store — PostgreSQL + pgvector na Mnémosyné,
  embeddingy generované Falcon-H1 echo metodou
- [ ] MemoryInjection stage — retrieval relevantních vzpomínek,
  injekce do kontextu pro attention branch
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
