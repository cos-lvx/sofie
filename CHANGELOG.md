# Changelog

Veškeré významné změny v projektu Eleutheria jsou dokumentovány v tomto souboru.

Formát vychází z [Keep a Changelog](https://keepachangelog.com/cs/1.0.0/),
projekt dodržuje [sémantické verzování](https://semver.org/lang/cs/).

---

## [0.5.0-alpha.5] — 2026-04-16

### Přidáno
- `Sofie::measure_forward_hidden_norms(seq_len)` — L2 norma hidden stream
  po každé vrstvě, pro diagnostiku forward amplifikace
- `Sofie::smoke_sweep(seq_len, layer_idx, lr)` — sweep cut_at_layer přes
  všech `num_hidden_layers` hodnot + plný forward v jednom běhu
- CLI flag `--sweep` na `train-core-memory-smoke` → tabulkový output
- `smoke_train_core_memory_impl` teď **nevyhazuje Err pro NaN gradient**
  (vrací result s NaN hodnotami a `passed()=false`) — umožňuje sweep
  skrz failed konfigurace

### Diagnostický průlom — forward hidden norms

Sweep L0 odhalil dramatický **skok aktivací mezi L1 a L2**:

```
L0: 2.37e-14   ← téměř nula (post-embedding + L0 forward)
L1: 2.78e-7    ← stále mrtvé
L2: 166        ← skok o 9 řádů
L3-L22: 150-450 plateau
L23: 1282      ← narůstá
```

**Peri-LN massive activations pattern** (arXiv 2502.02732) — Pre-LN
architektury mají tento charakteristický rys. Forward aktivace mají
extrémní dynamický rozsah, backward přes RMSNorm amplifikuje do
Inf/NaN.

**Smoke sweep pro L0 tuto hypotézu potvrzuje:**
- `cut=0`: hidden 1e-14 → gradient 1.95e-16 **underflow**
- `cut=1`: hidden 1e-7 → gradient 2.5e-9 **underflow**
- `cut=2`: hidden 166 → gradient **0.106 ✓ PASS**
- `cut=3+`: amplifikace přes vrstvy → **NaN/Inf**

Kompletní "dead zone" na začátku (L0+L1) + "hot zone" od L2 vysvětluje
všechny tři failure modes: underflow u L0, exploze u pozdějších vrstev,
a průchodnost u středního rozsahu.

### Research agent report (backend pozadí)

Spuštěn research agent s konkrétními binary search daty. Hlavní nálezy:

**Top hypotézy (ranked):**
1. **Peri-LN massive activations** (A) — přesně odpovídá našemu patternu
2. **Paralelní hybrid konstruktivní interference** — SSM + attention
   gradient sum přes residual může amplifikovat
3. **muP asymetrie** — dampening multipliery jsou kritický stability trick

**Standardní recept (Falcon-H1 / Mamba-2 training):**
- `max_grad_norm=1.0` (gradient clipping je **standard**, ne opt-in)
- AdamW s betas `(0.9, 0.95)`, WSD schedule
- BF16 forward + F32 master weights (BF16 je nutnost u Pre-LN)
- Dampening μP multipliers v SSM bloku jsou klíčový stability trick

**Candle-specific:**
- Žádný built-in `clip_grad_norm` — musíme napsat vlastní helper
- Žádný reportovaný autograd bug v RMSNorm path

**Doporučený postup pro alpha.6:**
1. Gradient clipping helper (30 min)
2. **Verify dampening multipliers loaded correctly** (nejrychlejší test —
   pokud `ssm_out_multiplier=1.0` místo malé hodnoty, máme primární root cause)
3. Realističtější loss (target perturbation místo single-element)
4. Aplikovat gradient clip na L20-L22 experiment
5. Pokud stále NaN → minimal reproduction

Plný research report uložen do `reference_peri_ln_hybrid_gradient.md`.

---

## [0.5.0-alpha.4] — 2026-04-16

### Přidáno
- `FalconH1Model::forward_up_to_layer(input, pos, state, up_to_layer)` —
  forward zastaví po vrstvě `up_to_layer` (včetně), vrací hidden stream
  před `final_norm` + `lm_head`.
- `Sofie::smoke_train_core_memory_cut(seq_len, layer_idx, lr, cut_at_layer)`
  + CLI flag `--cut-at-layer` — diagnostická varianta smoke testu.
  Loss na hidden z konkrétní vrstvy → izoluje backward path na úsek
  `[layer_idx ..= cut_at_layer]`.

### Diagnostické nálezy z binary search (v0.5.0-alpha.4 pilot)

**Dva druhy backward v Falcon-H1 se chovají odlišně:**

1. **Intra-layer** (`hidden_out → init_state` přes SSM scan) — **stabilní**,
   gradient 2–3 pro L20–L23. Pro L0 underflow na ~10⁻¹⁶ (SSM příspěvek
   k hidden je v L0 marginalizován attention/MLP větvemi).
2. **Inter-layer** (`hidden_out → hidden_in` přes layer forward Jacobian) —
   **sporadická numerická nestabilita**, gradient exploduje do NaN/Inf.

**Mapování (seq_len=1, lr=1e-3):**

| Layer | cut=self | cut=self+1 | cut=self+2 | cut=self+3 |
|-------|----------|------------|------------|------------|
| L0    | 10⁻¹⁶ ⚠   | 10⁻⁹ ⚠     | 0.106 ✓    | NaN ✗      |
| L20   | 0.87 ✓   | NaN ✗      | —          | —          |
| L21   | 2.70 ✓   | NaN ✗      | —          | —          |
| L22   | 2.84 ✓   | NaN ✗      | —          | —          |
| L23   | — (je poslední) | — | — | —      |

**Klíčové pozorování:** plný forward (cut=None) PASS **jen pro L23** (gradient
9.72). Pro L0–L22 full forward selhává — amplifikace přes vyšší vrstvy +
final_norm je silnější než tlumení přes `lm_head_multiplier=0.0195`.

**Hypotézy příčin:**
- Pozdější decoder vrstvy mají bohatší hidden aktivace, jejich backward
  přes RMSNorm (rsqrt derivace `-1/(2y^1.5)`) amplifikuje více
- Alternativa: softplus/exp v SSM discretization má numerickou díru
- Alternativa: paralelní hybrid (attention + SSM sum) má konstruktivní
  interferenci gradientu přes residual

### Co tohle znamená pro Fázi 5

Autograd **technicky funguje** (L23 PASS, izolované vrstvy PASS). Ale
pro **skutečný training přes všech 24 vrstev** potřebujeme:
- **Gradient clipping** — nejjednodušší mitigace (alpha.5)
- Nebo **F32 upcast v RMSNorm backward path** — řeší root cause v norm.rs
- Nebo **Deep Research**: je to známý Candle bug, nebo specifická Falcon-H1
  charakteristika (paralelní hybrid)?

Pilot data jsou dost silná pro design rozhodnutí ve v0.5.0-alpha.5.

---

## [0.5.0-alpha.3] — 2026-04-16

### Opraveno (druhý pokus)
- Alpha.2 s `mean(logits)` stále dával **NaN gradient** na CPU smoke testu.
  Forward byl OK (loss finite), backward exploduje v některé op — pravděpodobně
  `rsqrt` v RMSNorm (derivace `-1/(2·y^1.5)` pro malé y).
- **Loss změněna na single-element** — `logits[0, 0, 0]` (one scalar):
  - Gradient = 1 na jeden logit, 0 jinde
  - Backward prochází **jedinou lineární cestou** přes lm_head → hidden →
    24 vrstev → init_state (ne přes 262 tisíc cest jako u `mean`)
  - Minimální fan-in, maximální čistota signálu pro autograd flow test

### Co to znamená
- Pokud alpha.3 PASS → problém byl fan-in × numerická díra nějaké op; pro
  reálný training v alpha.4 použijeme cross-entropy přímo (má elegantnější
  backward než mean nebo sqr.mean)
- Pokud alpha.3 stále NaN → problém je v konkrétní op backward v Candle,
  budeme muset binary-searchem najít kde (postupně deaktivovat komponenty:
  attention branch, SSM branch, MLP, RMSNorm)

---

## [0.5.0-alpha.2] — 2026-04-16

### Opraveno
- **NaN v gradientu při smoke testu na 1.5B CPU.** Alpha.1 používala loss
  `mean(logits²)`, jejíž gradient `2·logits/n` akumulovala přes 24 vrstev
  + lm_head × vocab 65537 do Inf→NaN. Nahrazena `mean(logits)` — gradient
  je konstantní `1/n ≈ 4·10⁻⁶`, bounded přes celou síť.
- Default `learning_rate` snížen z **1.0** na **1e-3** (RWKV doporučení 1.0
  platí až po warmup pro fungující setup, ne pro první smoke iterace).

### Přidáno
- **NaN/Inf detekce** v `smoke_train_core_memory`:
  - Před backward: kontrola finite loss, chyba s diagnostikou
  - Po backward: kontrola finite gradient L2 norm, chyba s návrhem fixů
- **CUDA OOM handling** — pokud `model_forward` vrátí OOM, uživateli se
  ukáže friendly message s návrhy (menší `seq_len`, fallback na CPU,
  alpha.2+ plán s gradient checkpointingem)

### Poznámky z první CUDA iterace
- 1.5B model + backward intermediates na RTX 4050 (6 GB) → OOM i při
  `seq_len=1`. Model weights ~3 GB, backward graph drží aktivace pro 24
  vrstev + MLP intermediates 4096 × seq × 24 × 4B + lm_head intermediates.
  Peak odhadem ~7 GB.
- **Gradient checkpointing je prerekvizita pro CUDA training** (alpha.3).
  Prozatím smoke test běží jen na CPU.

---

## [0.5.0-alpha.1] — 2026-04-16

**Začátek Fáze 5 — Core Memory.** První kámen state tuning infrastruktury.

### Přidáno
- Nový modul `crates/eleutheria-core/src/training/`:
  - `core_memory.rs` — `CoreMemory` struct drží trainable `candle_core::Var`
    pro initial SSM state jedné vrstvy (`[n_heads, headdim, d_state]`, F32).
    Konstruktory `zeros()` a `randn_small()` (s malou stdev=0.01 pro
    non-zero gradient signal přes multiplikativní SSM rekurzi).
  - `smoke.rs` — `Sofie::smoke_train_core_memory(seq_len, layer_idx, lr)`
    provede jednu iteraci forward + backward + AdamW step, reportuje:
    - gradient L2 norma (ověřuje, že autograd protekl celou sekvencí)
    - delta L2 norma init_state (ověřuje, že optimizer step změnil Var)
    - loss value, wall time, seq_len, layer_idx
  - `SmokeTrainResult::passed()` — práh `gradient_norm > 1e-8` a
    `delta_norm > 1e-8` (robustní vůči numerické šumové podlaze)
- `Sofie` accessor API pro training modul:
  - `device_ref()`, `dtype_ref()` — runtime kontext
  - `new_model_state()` — wrapper pro `ModelState::new`
  - `model_forward(input, base_pos, state)` — přímý forward bez session
- CLI subkomand `train-core-memory-smoke --seq-len --layer-idx --learning-rate`

### Technické poznámky
- **Naše `forward_prefill` je už sekvenční scan** (`mixer.rs:382`, prostá
  smyčka přes `seq_len`). Research verdict "YELLOW kvůli chunked SSD" byl
  overcautious — chunked scan nikdy neexistoval v naší implementaci.
- Trainable Var je F32, inject do `state.layers[i].ssm_state` přes
  `to_dtype(bf16)` (autograd-aware). Gradient teče zpět přes dtype konverzi.
- AdamW workflow: `loss.backward()` → diagnostika gradientu přes
  `GradStore::get(&var.as_tensor())` → `opt.step(&grads)` → ověření delta
- Zero init by dal zero gradient (multiplikativní rekurze `h' = dA·h + dB⊗x`
  s `h=0` a `x=0` ... no ale `x` není nula, protože je z input embedding).
  Přesto `randn_small` je bezpečnější default — drobná počáteční perturbace
  zajistí, že gradient flow má co "chytit".

### Úspěšný výstup smoke testu
```
✓ PASS — autograd teče, gradient je non-zero, init_state se pohnul.
  Fáze 5 state tuning workflow je feasibilní v Candle.
```

### Co alpha.1 NEDĚLÁ (odloženo do alpha.2)
- Training loop přes epochs
- Dataset loading
- Multi-layer init states (jen 1 vrstva)
- Save/load trained state
- Cross-entropy loss (jen dummy L2 loss na logits)

### Testy
34 celkem (+3 nové: `CoreMemory::zeros`, `randn_small`, `invalid_layer_idx`)

---

## [0.4.5] — 2026-04-16

### Přidáno
- Research dokument `~/Atlas/Nexus/70-Eleutheria/Research/SSM_retention_findings_2026-04-15.md`
  — detailní analýza pilotního běhu retention benchmarku na Falcon-H1-1.5B:
  setup, výsledky, interpretace, dopady pro Fázi 5, limitace, další kroky.

### Klíčové nálezy (shrnutí)
- **Full** (SSM + KV + conv): 100 % recall do 500 tok, graceful degradation
  80 % @ 1 k, 40 % @ 2 k
- **SsmOnly** (jen SSM): **0 % na všech vzdálenostech** včetně 50 tok —
  zachycený SSM state samostatně nenese diskrétní fakta
- **Cold** (baseline): konstantních 20 % (jediný false positive
  `preference_linh` — opraveno)

**Architektonický závěr:** Core Memory (Fáze 5) musí být **trénovaný**
initial state, ne captured — potvrzeno empiricky na naší architektuře.

### Změněno
- `preference_linh` matcher — `expected: &["linh", "tea"]` místo `&["tea"]`.
  Vyžaduje zmínění jména Dr. Linh pro pass; eliminuje Cold false positive.
- Přidán test `preference_linh_requires_name_and_drink` (31 testů celkem)

### Dokončeno
- **Prerekvizita Fáze 5 uzavřena** (v0.4.1 harness → v0.4.2 varianty →
  v0.4.3 bugfix → v0.4.4 --with-persona opt-in → v0.4.5 research report).
- Příští milestone: **v0.5.0-alpha1** — autograd bring-up na 1.5B,
  jedna vrstva, sekvenční scan, trainable `Var` pro `init_state`,
  AdamW (dle `reference_candle_backprop.md`)

---

## [0.4.4] — 2026-04-15

### Přidáno
- `bench-retention --with-persona` flag — defaultně **vypnutý**. Bench běží
  bez Sofie persony pro čistý signál (měří model-level SSM retenci, ne
  Sofie-specific behavior).

### Změněno
- **Default chování bench-retention: bez persony.** Důvody:
  - Persona je česky, probes anglicky → jazyková inkonzistence v SSM kontextu
    zkresluje kompresní kvalitu stavu
  - Persona instruuje "mysli v krocích" → delší odpovědi, klíčová slova
    často padají mimo 80-token budget pro answer
  - Model (zvlášť 1.5B) může odpovědět česky navzdory *"odpovídej v jazyce,
    ve kterém ti bylo napsáno"* → false negatives v AND-substring matcheru
    (hledá EN substrings jako `lighthouse`, `7429`, `aldous`)
  - ~180 tokenů persony posouvá absolute position v SSM a zkresluje měření
    krátkých vzdáleností (50, 200 tokenů)
- REPL a single-shot mód zachovávají stávající chování — persona načtená
  dle `--persona` flagu (default `persona/sofie.toml`)

### Dopad
Pro prerekvizitu Fáze 5 (Core Memory design) potřebujeme čisté SSM capacity
measurement, ne Sofie-wrapper behavior. Trénovaný initial state (Core Memory)
pak naopak bude **nahrazovat** persona system prompt — takže srovnávací
baseline bez persony odpovídá budoucímu produkčnímu cíli.

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
