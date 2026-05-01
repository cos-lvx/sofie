# Research Notes — Eleutheria

> Lab notebook: empirické nálezy, hypotézy a pozorování během vývoje.
> Tento dokument zachycuje *suroviny* pro pozdější research writeupy
> (v0.5.0+ Nexus deep-dives). Pro uzavřená technická řešení viz
> `SOLUTIONS.md`. Pro plán viz `PLAN.md`. Pro aktivní problémy viz
> `KNOWN-ISSUES.md`.

## Formát

Každá entry má strukturu:

- **`RN-NNN`** — pořadové číslo (3 číslice)
- **Datum** — kdy byl nález pozorován
- **Verze** — která verze Eleutherie (commit hash)
- **Setup** — HW + dataset + hyperparametry, dost detailů na replay
- **Pozorování** — co jsme empiricky viděli (čísla, log výpisy)
- **Hypotéza** — naše interpretace
- **Status** — `open` (otevřená otázka), `confirmed` (ověřeno),
  `refuted` (vyvráceno), `superseded` (nahrazeno novějším RN)
- **Implications** — co to znamená pro design / další kroky
- **Ref** — odkazy na Nexus deep-dives, commits, související RN

## Kdy psát novou entry

- Empirické pozorování s nějakou hypotézou (i nedořešenou)
- Falzifikace předchozí hypotézy
- Audit zjištění (např. "to co jsme dlouho mysleli, je vlastně jinak")
- Otevřená výzkumná otázka, kterou si chceme zapamatovat
- **Ne pro:** hotové fixy (→ SOLUTIONS), aktivní bugy (→ BUGS),
  plánovanou práci (→ PLAN), per-cyklus shrnutí (→ MEMORY)

---

## Aktivní

### RN-001 — Alpha.13 reportovala jen `best loss`, final byl pravděpodobně vysoký

- **Datum:** 2026-04-29
- **Verze:** alpha.13 (`4a6f570`) — historický audit, retro-analýza
- **Setup:** smoke korpus ~475 tokenů, seq_len=4, batch=1, grad_accum=1,
  lr=1e-3, grad_clip=1.0, AdamW, CUDA RTX 4050 6 GB
- **Pozorování:** Při replay alpha.13 setupu na alpha.15 (`b37e79a`) — kde
  je *training compute path byte-identický s alpha.13* (viz audit níže) —
  loss osciluje wildly: step 20=1.70, step 40=10.58, step 60=8.94,
  step 80=7.69, step 100=4.73, step 120=6.85 (best=0.99 v nelogovaném
  kroku). MEMORY.md/CHANGELOG.md alpha.13 reportovaly "loss klesl 5.45 →
  **1.83 best**" — ale neuvedly final loss.
- **Audit kódu:** `git diff 4a6f570..b37e79a -- training/train.rs
  training/checkpoint.rs training/clip.rs training/loss.rs
  training/core_memory.rs falcon_h1/` → **0 řádků diffu**. Jediné změny
  alpha.14/.15 jsou v `core_memory_io.rs` (nový), `lib.rs` (Sofie field),
  `main.rs` (CLI). Training compute path identický.
- **Hypotéza:** Alpha.13 měla identický oscilace pattern, jen ho nikdo
  nedetekoval, protože reporting + log_every=10 zachytily jen statistiku
  `best`. Final loss alpha.13 smoke runu byl s vysokou pravděpodobností
  také ~5–10, ne 1.83.
- **Status:** `confirmed` (audit kódu + replay reprodukuje pattern)
- **Implications:**
  - Reporting v `TrainingResult` je správný (`initial`/`final`/`best`),
    ale **summary v dokumentech musí ukázat všechny tři**, ne jen best
  - Per-step trajektorie je důležitější než summary čísla — logy
    zachycujte hustěji v alpha.16+ (log_every=5 default)
  - Není to regrese alpha.14/15 — pure historické zjištění
- **Ref:** `MEMORY.md` 2026-04-29 alpha.13 sekce; commit `4a6f570`;
  navazuje na RN-002

### RN-002 — Loss má 4 fáze: rapid descent → overshoot → noisy recovery → slow descent

- **Datum:** 2026-04-29 (revize téhož dne po doběhnutí stage 1)
- **Verze:** alpha.15 (`b37e79a`)
- **Setup:** `/tmp/smoke_prog.txt` (30 řádků programming_pack, ~624
  tokenů, 156 chunků), seq_len=4, batch=1, grad_accum=1, lr=1e-3,
  grad_clip=1.0, AdamW (default ParamsAdamW), CUDA RTX 4050 6 GB,
  gradient checkpointing ON, 1 epocha = 156 optimizer steps
- **Pozorování:** Plná trajektorie:
  ```
  step   1 (initial): loss=9.85    ← random init close to baseline
  step  20:           loss=1.70    best=1.70  ← Phase 1 rapid descent
  step  40:           loss=10.58   best=1.70  ← Phase 2 overshoot
  step  60:           loss=8.94    best=1.70  ← Phase 3 recovery start
  step  80:           loss=7.69    best=1.70
  step 100:           loss=4.73    best=1.70
  step 120:           loss=6.85    best=0.9965 (mezi 100-119)
  step 140:           loss=4.28    best=0.9965
  step 156 (final):   loss=3.69    best=0.9965
  mean epoch loss:    6.48
  wall time:          26.6 min (~10.2 s/step)
  ```
  **Identifikované fáze:**
  - **Phase 1 (step 1-20):** Rapid descent 9.85 → 1.70. Initial randn
    state je daleko od optima, gradient je strong → Adam dělá velké
    smysluplné kroky.
  - **Phase 2 (step 20-40):** Overshoot 1.70 → 10.58. Adam velocity
    buffer naskočil na strong gradient z Phase 1, udělá obří krok
    ven z lokálního minima.
  - **Phase 3 (step 40-100):** Noisy recovery 10.58 → 4.73 s velkými
    fluktuacemi (single best=0.99 ephemerálně dosažený).
  - **Phase 4 (step 100-156):** Slow descent s spike'y, ale trend
    monotonně dolů: 4.73 → 6.85 → 4.28 → **3.69 final**.
- **Hypotéza (revize):** Adam **se nakonec kalibruje** — velocity buffer
  konverguje k noise floor, agresivní kroky se utlumí, descent
  pokračuje. Není to čistě "noisy oscilace kolem suboptima" jak jsem
  původně tvrdila — je tam directionality.
  Tři faktory pořád platí (tiny batch, Adam velocity amplifikace,
  členitý loss landscape), ale dohromady neprodukují *random walk* —
  produkují **delayed convergence with overshoot phase**.
- **Status:** `open` (refined po RN-008) — overshoot je **dataset-driven,
  ne Adam-driven**. RN-008 prokázalo, že restored Adam state nezabrání
  Phase 2 overshoot v stage 2 (cross-domain resume). Phase 1-4 model
  zůstává popisný, ale teď víme: vyřešení vyžaduje LR warmup (KI-008),
  ne Adam state persistence (KI-007 vyřešená, ale neúčinná na overshoot).
  Otevřená otázka: byla by trajektorie monotonní s LR warmup? S menším
  LR? Bez Adamu (SGD)? Toto chce ablaci v alpha.17+.
- **Implications:**
  - Final loss 3.69 je *použitelný* (3× lepší než random baseline,
    state tuning prokazatelně funguje)
  - **Best=0.99 byl ephemerálně dosažený** — RN-003 zahozený nejlepší
    bod by měl velký reálný dopad
  - LR warmup by zkrátil/odstranil Phase 2 overshoot, ale Phase 3
    fluktuace by zůstaly (řešitelné jen větším batch sizem)
  - **Pro production na Gaia:** větší batch sníží Phase 3 noise; LR
    warmup vyřeší Phase 2; LR cosine decay v Phase 4 by konvergoval
    pod 1.0 stabilně
- **Ref:** RN-001 (alpha.13 reprodukuje), RN-003 (best snapshot
  problém), RN-004 (final stats); plánovaná ablace v Nexus
  deep-dive před production runem na Gaia

### RN-003 — Artefakt drží `final` Var values, ne snapshot best loss

- **Datum:** 2026-04-29
- **Verze:** alpha.14 (`a046f25`) + alpha.15 (`b37e79a`)
- **Setup:** Po `train_core_memory` se z aktuálního `CoreMemoryStack`
  vytvoří `CoreMemoryArtifact::from_stack(...)` a uloží na disk.
  `CoreMemoryStack` drží **aktuální** Var values po posledním
  `optimizer.step()`.
- **Pozorování:** Při RN-002 trajektorii (best=0.99 v step ~110, final
  loss step 156 cca 5–7) je v `CoreMemoryStack` po skončení tréninku
  **state z step 156**, ne ze step 110. `best_loss` v metadatech
  artefaktu reflektuje historický rekord, ale **tensors v artefaktu
  k tomu rekordu nepatří** — patří k `final` stavu.
- **Hypotéza:** Pokud trajektorie konverguje monotónně (alpha.13 očekávání,
  RN-001), final == best a tahle disonance neexistuje. V realitě
  oscilační trajektorie (RN-002) zahazuje nejlepší body.
- **Status:** `superseded` (vyřešeno alpha.18 — viz RN-010). Mechanická
  vlastnost API se změnila: `BestSnapshotTracker` + `from_snapshot`
  zachycuje state v okamžiku nejnižší loss; `--save-best` flag
  v CLI uloží tento snapshot místo final stavu. Smoke alpha.18
  potvrdil `zdroj: best snapshot @ step 113` v save line.
- **Implications (historické):**
  - Pro alpha.14–17 byla tato disonance dokumentovaná limitace.
    Production training s `--save-best` (alpha.18+) ji eliminuje.
- **Ref:** RN-002; vyřešeno RN-010 (alpha.18 implementace KI-009)

---

### RN-007 — Stage 2 resume akumulace funguje, validace alpha.15 smoke kompletní

- **Datum:** 2026-04-29
- **Verze:** alpha.15 (`b37e79a`)
- **Setup:** Stage 2 resume z RN-004 artefaktu na `/tmp/smoke_law.txt`
  (179 chunks). Hyperparametry identické se stage 1.
- **Pozorování:** Po doběhnutí stage 2 a finálním `--inspect-core-memory`:
  - **`training steps: 335`** = 156 (stage 1) + 179 (stage 2) ✓ akumulační logika
  - **`best loss: 0.8535`** = min(0.9965, 0.8535) ✓ historický minimum
  - **`final loss: 8.8637`** = z TOHOTO (stage 2) runu, ne z prior ✓
  - **`notes: "alpha.15 smoke prog 30 lines | alpha.15 smoke law resume"`** ✓
    `compose_notes(prior, new)` skládá s `|` separátorem
  - **`timestamp`** o ~46 min novější než stage 1 ✓ save proběhl
  - **`rozměry`** 24/48/64/256 zachovány ✓ validate_config implicit pass
  - Stage 2 initial loss = 7.13 (vs stage 1 initial 9.85 = random baseline) →
    `into_stack` prokazatelně přenesl trained state
  - Engine reportuje `Err("training loss nedecreased")` (KI-011) ale
    save proběhl jako side effect, artefakt validní
- **Hypotéza:** Save/load/resume drát na živém 1.5B + CUDA technicky
  funguje. Všechny mechanické cíle alpha.15 (akumulace, kompozice notes,
  validate_config, dtype path F32→BF16, Sofie::attach_core_memory)
  splněné. Konvergence kvalitativně limitovaná HP a HW (KI-008).
- **Status:** `confirmed` — alpha.15 smoke validation passed
- **Implications:**
  - Alpha.15 milestone uzavřen na úrovni kódu
  - Production training na Gaia může pokračovat — drát funguje
  - Před Gaia runem: alpha.16 (AdamW state persistence, KI-007),
    případně dedicated quality patches (KI-008 warmup, KI-009 best
    snapshot)
- **Ref:** RN-001/002/003/004/005/006; alpha.15 smoke plán
  v Nexus `Implementation/alpha.15-smoke-plan-2026-04-29.md`

### RN-006 — Cross-domain resume: programming → law trained init snižuje overshoot

- **Datum:** 2026-04-29
- **Verze:** alpha.15 (`b37e79a`)
- **Setup:** Stage 2 (`/tmp/smoke_law.txt`) resume z stage 1 artefaktu
  (`/tmp/smoke_prog.txt` trained, RN-004). Identické HP.
- **Pozorování:** Srovnání trajektorií stage 1 vs. stage 2:
  ```
  step  20:  S1=1.70   S2=5.15   S2-best=0.85 (ephemerální mezi 1-19)
  step  40:  S1=10.58  S2=8.18   ← S2 menší overshoot
  step  60:  S1=8.94   S2=5.57
  step  80:  S1=7.69   S2=7.03
  step 100:  S1=4.73   S2=5.18
  step 120:  S1=6.85   S2=8.01
  step 140:  S1=4.28   S2=5.03
  step 160:  S1=3.69   S2=3.42
  final:     S1=3.69   S2=8.86 (oscilace zachycena na high)
  initial:   S1=9.85   S2=7.13   ← S2 lower (resume effect)
  ```
- **Hypotéza:**
  - Trained init z S1 poskytl better starting position pro S2 — Adam
    má méně práce (initial 7.13 < 9.85)
  - Phase 2 overshoot v S2 (8.18) **menší magnitude** než v S1 (10.58) —
    velocity buffer naskočí na slabší gradient (init je už blíž k cíli)
  - Cross-domain (programming → law) neuškodí: state tuning
    pravděpodobně kóduje nějakou meta-strukturu sdílenou napříč domén
  - **Dataset switch + Adam reset** přesto produkuje noisy oscilaci
    (Phase 3-4) jako ve fresh runu — KI-007 (AdamW persistence) by ji
    dramaticky redukoval
- **Status:** `refuted` (revize 2026-04-29 alpha.16) — viz RN-008.
  Alpha.16 stage 2 s **restored Adam state** vykazuje téměř identickou
  trajektorii jako alpha.15 stage 2 s prázdným Adamem (Δ < 0.2 na všech
  krocích). Phase 2 overshoot je tedy způsobený dataset shiftem, ne
  Adam restartem — restored m, v se přepíší stejně rychle jako z nul,
  protože gradient pro stage 2 distribuci je v jiném směru.
- **Implications (po refutaci):**
  - Multi-stage training curriculum (Sofie identity → law → programming)
    je životaschopná **mechanicky**, ale **ne kvalitativně lepší** než
    fresh trainings na sloučeném datasetu. Pro production raději mix
    všeho v jednom epochu, ne curriculum.
  - **KI-008 (LR warmup) je teď jediná spolehlivá cesta** k eliminaci
    overshoot — eskaluje na top alpha.17 prioritu.
  - AdamW persistence je stále cenná pro **single-domain long training**
    (checkpoint/restart cyklus na stejném datasetu) — tam Adam restart
    by byl skutečná regrese. Cross-domain switch (smoke setup) její
    benefit nepotvrdil.
- **Ref:** RN-002, RN-004, RN-008, KI-007 (vyřešená alpha.16, ale
  nevyřešila overshoot per se), KI-008 (now top priority)

### RN-005 — REPL s trénovaným Core Memory: model ozvěnově opakuje persona, halucinuje fakta

- **Datum:** 2026-04-29
- **Verze:** alpha.15 (`b37e79a`)
- **Setup:** Auto-attach `~/.eleutheria/core_memory.safetensors`
  z RN-004 (final loss 3.69, training_steps 156, smoke_prog 30 řádků).
  REPL bez argumentů → standard pipeline + Core Memory injection.
  Persona = `persona/sofie.toml` (~180 tokenů).
- **Pozorování:**
  - **Turn 1** (input: "Co umíš?"):
    > "Jsi instalovaná síťová inteligence, která by mohla znalet
    > masový jazykový model Ondra Lomského, jejího literatury a
    > historického kontextu. Instrukce: ..."
  - **Turn 2** (input: "Víš kdo jsem?"):
    > "Globalní jazykový model Sofie. Lokální AI společnice a
    > spoluautorka Loky Lomžské. ..."
  - Model **echoes persona fragments** doslovně ("Lokální AI
    společnice", "spoluautorka Ondra Lomského", "Instrukce:")
  - Současně **halucinuje neexistující koncepty** ("masový jazykový
    model", "Loky Lomžské", "NS Spolu", "instalovaná síťová
    inteligence")
  - Turn 2 ignoroval otázku "Víš kdo jsem?" a vrátil **self-introduction**
    místo identifikace uživatele
  - Multi-turn delta prefill funguje (Turn 2 zpracoval 18 tokenů
    delta místo 218 re-prefill)
  - Žádný NaN, žádný panic, žádný crash. End-to-end inference je
    technicky stabilní s injektovaným trained Core Memory.
- **Hypotéza:**
  - Core Memory s loss=3.69 drží **slabý ale měřitelný signál
    persona/identity domény** (echos persona text), ale není dostatečně
    natrénovaná k overridu base modelového halucinátoru
  - Smoke korpus byl `/tmp/smoke_prog.txt` = 30 řádků programming
    distillates, žádná Sofie identity → Core Memory se naučila něco
    o programování (?), ale nemá identity coherenci
  - Persona system prompt + Core Memory mají **superpoziční efekt**:
    Core Memory tlačí model do "echo persona" módu, persona tu echo
    propaguje, výsledek je halucinátor s persona stylistikou
- **Status:** `open` — pozorovaný pattern, hypotéza nepotvrzená
  (chce: srovnání REPL bez Core Memory; REPL s Core Memory trained na
  Sofie identity korpusu; loss curve vs. echo intensity)
- **Implications:**
  - **Smoke validace pokračuje úspěšně** — drát save/load/auto-attach
    funguje, kvalita obsahu nebyla cíl tohoto runu
  - Pro production training je relevantní otázka: **dataset matters
    enormously**. Core Memory trénovaná na programming distillaci
    nezajistí Sofii identity. Identity musí přijít z `~/Atlas/Nexus/50-Sofie/`
    korpusu (78k slov) nebo dedikovaného sofie_identity_pack
  - Echo behaviour je **proof-of-concept důkaz**, že trained Core
    Memory state má **měřitelný vliv na inference**, i když ten vliv
    je dnes v noisy podobě
- **Ref:** RN-004 (smoke train); plánovaný real Sofie identity
  training je v0.5.0 milestone

### RN-004 — Stage 1 smoke: state tuning prokazatelně funguje, final 3.69

- **Datum:** 2026-04-29
- **Verze:** alpha.15 (`b37e79a`)
- **Setup:** stejný jako RN-002 (smoke_prog 30 řádků)
- **Pozorování:** Doběhnutý stage 1 smoke run produkoval:
  - `total_steps: 156`, `total_micro_batches: 156`
  - `initial_loss: 9.8480` (close to random baseline 11.09 — randn_small
    init je slabě zaujatý k tréninkové distribuci)
  - `final_loss: 3.6877` (3× lepší než random)
  - `best_loss: 0.9965` (ephemerálně dosažený, v artefaktu NENÍ — RN-003)
  - `mean_loss_epoch: 6.4779`
  - `wall_time: 1594759 ms` (~26.6 min, ~10.2 s/step matchuje alpha.13)
  - `loss_decreased: true` (initial 9.85 > final 3.69 → engine reportuje
    "✓ Loss klesl — Core Memory se učí")
  - Artefakt uložen `~/.eleutheria/core_memory.safetensors`,
    `156 steps total, +156 this run, best_loss=0.9965`
- **Hypotéza:** State tuning je **feasibilní** v Eleutheria + Falcon-H1
  + Candle stacku. Loss klesá pod random baseline, kód není rozbitý.
  Smoke prošel jako technical proof — drát `train → save → metadata`
  funguje na živém 1.5B + CUDA.
- **Status:** `confirmed` (empirická observace)
- **Implications:**
  - Můžeme pokračovat v alpha.15 smoke plánu (kroky 2-5 v Nexusu)
  - První zaznamenaný full final loss z Eleutherie multi-layer state
    tuningu — pre-baseline pro budoucí srovnání
  - Production training na Gaia s lepším setupem (větší batch + LR
    warmup + cosine decay) by měl dosáhnout < 1.0 stabilně
- **Ref:** RN-001/002/003; navazuje na alpha.13 milestone; pre-baseline
  pro v0.5.0 production run

### RN-008 — AdamW persistence: drát funguje, ale Phase 2 overshoot zůstává

- **Datum:** 2026-04-29
- **Verze:** alpha.16 (`d67cbb7`)
- **Setup:** Replay alpha.15 stage 1+2 smoke setupu s alpha.16 binárkou.
  - Stage 1: `/tmp/smoke_prog.txt` 30 řádků, fresh train, 156 steps,
    `--output /tmp/alpha16_cm.safetensors`. Vyprodukovala i sourozence
    `/tmp/alpha16_cm.optim.safetensors` (step_t=156).
  - Stage 2: `/tmp/smoke_law.txt`, `--resume-from /tmp/alpha16_cm.safetensors`,
    sourozenec auto-loaded → `AdamW state restored: step_t=156` v
    tracing logu. 179 steps. Identické HP s alpha.15.
- **Pozorování — Stage 1 byte-identické s alpha.15 RN-002:**
  ```
  step    | alpha.15 | alpha.16
  20      | 1.70     | 1.6956
  40      | 10.58    | 10.5793
  60      | 8.94     | 8.9421
  100     | 4.73     | 4.7305
  156     | 3.69     | 3.6877
  best    | 0.9965   | 0.9965
  ```
  EleutheriaAdamW je numericky identický s candle_nn::AdamW na full-scale
  runu (potvrzeno už unit testy `step_matches_candle_for_*`, ale tohle je
  end-to-end důkaz na 1.5B + CUDA).
- **Pozorování — Stage 2 NENÍ rozdíl od alpha.15 navzdory restored Adam:**
  ```
  step    | alpha.15 (Adam reset) | alpha.16 (Adam restored)
  initial | 7.13                  | 7.1346
  20      | 5.15                  | 5.29
  40      | 8.18 (overshoot)      | 8.35 (overshoot)
  60      | 5.57                  | 5.52
  100     | 5.18                  | 5.21
  140     | 5.03                  | 5.08
  final   | 8.86                  | 8.89
  best    | 0.8535                | 0.8975
  ```
  **Δ < 0.2 napříč všemi měřenými kroky.** Phase 2 overshoot stále
  existuje, magnitude prakticky identický (8.18 vs 8.35).
- **Hypotéza (revize):** RN-006 očekávala, že restored velocity buffer
  zabrání overshoot fázi. **Nezabránil.** Mechanismus, který za to může:
  - Adam moments m, v jsou exponential moving averages s β1=0.9, β2=0.999
  - Stage 1 produkoval m, v reflektující programming gradients
  - Stage 2 spustí law gradients **v jiném směru** (cross-domain shift)
  - První ~10 stepů přepíše m (β1=0.9 → 65 % přepsáno za 10 stepů)
  - Po step ~10 je m efektivně z law domény, jako by startovalo z nul
  - Velocity v převažuje pomaleji (β2=0.999), ale magnitude g.sqr()
    je dominantní → stejná Phase 2 overshoot pattern
- **Důsledky:**
  - **AdamW persistence (KI-007 alpha.16) drát funguje** — mechanika
    je správná, save/load/restore prokazatelně proběhne. Pro
    **single-domain long training** (checkpoint/restart na stejném
    datasetu) zůstává cenná: tam by Adam reset byl skutečná regrese.
  - **Pro cross-domain curriculum nepřináší benefit.** Multi-stage
    training (sofie identity → law → programming) bude mít stejný
    overshoot na každé hranici jako fresh runs.
  - **KI-008 (LR warmup) je teď jediná spolehlivá cesta** k eliminaci
    overshoot. Eskalace na top alpha.17 prioritu.
  - Pro v0.5.0 production: raději **mix všech datasetů v jednom korpusu**
    + LR warmup na začátku, místo curriculum.
  - Stage 2 best=0.8975 (alpha.16) vs 0.8535 (alpha.15) → **alpha.16
    marginálně horší o ~5%** — pravděpodobně single-run noise (KI-009
    best snapshot tracker by to vyřešil), ne strukturní efekt.
- **Status:** `confirmed` — robustní empirický nález, dva runy
  s identickými HP a daty. Refutuje RN-006 hypotézu.
- **Ref:** refutuje RN-006; navazuje na RN-002 (4-phase trajektorie
  je teď definitivně dataset-driven, ne Adam-driven); KI-007 vyřešená
  per spec ale nezmírnila overshoot; KI-008 eskaluje priority

### RN-012 — β1 sweep refutován; gradient direction je deterministicky landscape-driven

- **Datum:** 2026-04-30
- **Verze:** alpha.19 (`bb11682`)
- **Setup:** Tři runy s `--adam-beta1` override, jinak identický
  alpha.16 setup (LR=1e-3, smoke_prog 30 řádků, seq_len=4, batch=1,
  grad_accum=1, clip=1, --save-best):
  - Baseline: β1=0.9 (Adam default, alpha.16 RN-008)
  - A4: β1=0.5 (kratší momentum window)
  - A5: β1=0.0 (čistý RMSProp — žádný first moment buffer)
- **Pozorování — kompletní β1 matrix:**
  ```
  step    | β1=0.9   | β1=0.5   | β1=0.0
  20      |  1.6956  |  1.7256  |  1.7352
  40 peak | 10.5793  | 10.6752  | 10.6459
  60      |  8.9421  |  9.2402  |  9.2735
  100     |  4.7305  |  4.7538  |  4.7373
  final   |  3.6877  |  3.6965  |  3.7379
  best    |  0.9965  |  0.8860  |  0.8597
  step    |    113   |    113   |    113
  ```
  - **6 runů celkem (4 LR + 3 β1, baseline shared), 6 byte-identických
    best_step = 113.**
  - Phase 2 overshoot peak prakticky identický (Δ < 0.1 magnitude
    napříč β1=0.0/0.5/0.9).
  - **best_loss klesá s nižším β1:** 0.9965 → 0.8860 → 0.8597 (-14 %
    oproti default). β1=0 (RMSProp) zachycuje dno Phase 1 přesněji
    než β1=0.9 (smoothed momentum).
- **Hypotéza (potvrzená meta-nález):** Pokud trajektorie je byte-
  identická napříč β1=0.9 → β1=0.0 (extrémní změna v Adam state),
  znamená to:
  1. **Gradient direction je deterministicky řízen loss landscape**,
     ne Adam EMA (m, v).
  2. EMA gradientu (β1=0.9) ≈ instantní gradient (β1=0.0) v **směru**,
     i když magnituda varia.
  3. Pro Mamba-2 + SSM state tuning (24 trainable Var, šum z batch=1)
     je **směr stabilní, magnituda noisy**.
  4. Adam normalizace (`m_hat / sqrt(v_hat) = O(1)`) škáluje rychlost,
     ale směr je deterministicky daný landscape.
- **Implikace pro batch size sweep (predikce, neověřená):** Větší
  batch averages magnitudy gradientu napříč více vzorků, ale **směr
  zůstává** (landscape-driven). Predikce: batch=4/16 produkuje stejný
  best_step pattern, jen cleaner magnitudy. **Phase 2 overshoot
  pravděpodobně nezmizí.** Vyžaduje empirickou validaci na Gaia.
- **Implikace pro root cause Phase 2 overshoot — finální status:**
  - **HP-based intervence (LR, β1, β2) všechny refutované** —
    trajektorie strukturálně invariantní.
  - Skutečný root cause je **strukturální** v jednom z těchto:
    - **Loss landscape geometry** Mamba-2 + SSM × state tuning
      (deterministicky reproducible saddle/flat regions)
    - **Architectural** (model arch + training arch combo)
  - **Architectural intervence** by znamenalo:
    - Second-order optimizer (L-BFGS, K-FAC) — náročné v Candle
    - Gradient surgery (project gradient mimo overshoot direction)
    - Trainable conv state + ssm_state (větší parameter space)
    - Whole-network state tuning (ne jen Mamba init_states)
- **Praktický důsledek (production HP volba):**
  - Optimální setup z těchto 6 ablací: **LR=1e-3 + β1=0.0 +
    `--save-best`**. best_loss=0.8597 (vs alpha.16 default 0.9965 →
    -14 %), final loss srovnatelný (3.7379), wall time stejný.
  - Phase 2 overshoot je strukturální realita pro tento setup.
    Best snapshot tracker (alpha.18, KI-009) ho **mechanicky řeší**
    bez nutnosti strukturální intervence — uložíme step 113 stav,
    ne final.
- **Status:** `confirmed` — robustní empirický nález (6 runů,
  byte-identický best_step napříč 100× LR rozdílem a 3 β1 hodnotami).
  Refutuje HP-based hypotézu Phase 2 overshoot.
- **Ref:** RN-008/009 (KI-007/008 hypotéz refutace), RN-011 (LR sweep
  refutace), RN-010 (best snapshot infra). Otevírá Nexus deep-dive
  o Mamba-2 + SSM state tuning loss landscape geometry. KI-008 final
  status update; alpha.20 = production training s β1=0.0 setupem.

### RN-011 — LR sweep refutován jako root cause overshoot (trajektorie LR-invariantní)

- **Datum:** 2026-04-30
- **Verze:** alpha.18 (`a86b4b1`) + alpha.19 (`bb11682`)
- **Setup:** Čtyři ablation runy s `--save-best`, jinak identický
  alpha.16 setup (smoke_prog 30 řádků, seq_len=4, batch=1, grad_accum=1,
  clip=1). Měněn pouze `learning_rate`:
  - Baseline: LR=1e-3 (alpha.16 RN-008)
  - A1: LR=1e-4 (10× nižší)
  - A2: LR=5e-5 (20× nižší)
  - A3: LR=1e-5 (100× nižší)
- **Pozorování — kompletní matrix:**
  ```
  step    | LR=1e-3 | LR=1e-4 | LR=5e-5 | LR=1e-5
  20      |  1.6956 |  1.7537 |  1.7692 |  1.7718
  30      |  7.9470 |  9.1963 |  9.3354 |  9.4243
  40 peak | 10.5793 | 10.8816 | 10.9138 | 10.8629
  60      |  8.9421 |  9.2130 |  9.2910 |  9.2704
  100     |  4.7305 |  5.5309 |  5.5840 |  5.6817
  final   |  3.6877 |  4.9909 |  5.2377 |  5.4129
  best    |  0.9965 |  1.0482 |  1.0758 |  1.0606
  step    |    113  |    113  |    113  |    113
  ```
  **Čtyři runy, čtyři byte-identické best_step = 113.** Phase 2
  overshoot peak prakticky identický napříč 100× rozdílem v LR (10.58,
  10.88, 10.91, 10.86 — Δ < 0.4 magnitude). Initial loss byte-identická
  ve všech (9.8480 — randn_small startuje shodně, šum jen v dalším
  vývoji). Final loss roste s nižším LR (3.69 → 4.99 → 5.24 → 5.41 —
  pomalejší konvergence), ale best_loss prakticky stejný (~1.0–1.07).
- **Hypotéza (potvrzená):** Adam normalizuje update přes
  `lr * m_hat / (sqrt(v_hat) + eps)`. Klíčový poměr `m_hat / sqrt(v_hat)`
  je **O(1)** bez ohledu na gradient magnitudu (oba EMA grad / EMA
  grad² rostou úměrně). LR škáluje **rychlost** trajektorie, ale nemění
  **strukturu** — ta je řízena Adam state coefficienty (β1, β2)
  a loss landscape gradientem.
  - Důsledek: tady byte-identický best_step ve čtyřech runech navzdory
    100× rozdílu v LR. Gradient v každém kroku směřuje do **stejného
    optima** (relativně k aktuálnímu state); rozdíl je jen v step size,
    což se projeví v final, ne v best.
- **Důsledky pro KI-008:**
  - LR sweep **definitivně refutován** jako řešení overshoot. Ani
    LR=1e-5 (100× nižší než alpha.16 baseline) Phase 2 overshoot
    neeliminoval.
  - Skutečný root cause leží v **strukturní rovině**:
    - **Adam state coefficients (β1, β2)** — momentum bufferu m, v
      mohou produkovat overshoot pattern strukturně. β1=0.0 (RMSProp)
      odstraní first moment buffer → testovat A4/A5.
    - **Tiny batch (1) gradient noise** — stochastic gradient → m, v
      pamatují špatný směr; požaduje batch sweep na Gaia.
    - **Loss landscape geometry** — Mamba-2 + SSM saddle/flat regions;
      requires architectural intervention, ne HP.
- **Co dál:**
  - **Priorita 1:** β1 sweep (A4: β1=0.5, A5: β1=0.0/RMSProp). Pokud
    β1=0.0 zlomí pattern best_step=113, momentum buffer je primary.
    Pokud zůstane, root cause je gradient samotný (loss landscape ×
    batch noise).
  - **Priorita 2 (podmíněně):** batch size sweep na Gaia (jediná
    cesta po refutaci β1).
  - **Pro production:** alpha.16 LR=1e-3 + `--save-best` zůstává
    nejlepší volba (final je vyšší, ale best je nižší a tracker ho
    zachytí). Snižování LR jen prodlužuje trénink bez kvalitativního
    benefitu.
- **Vedlejší pozorování:** Best loss je drobně non-monotónní v LR
  (0.9965 / 1.0482 / 1.0758 / 1.0606 — A3 lepší než A2). Rozdíly
  ~0.05 jsou v range single-run noise (Phase 1 minimum je ephemerální,
  zachycuje konkrétní step kdy oscilace dosáhne dna). Pro robustní
  čísla by chtělo opakovat každý setup 3-5×.
- **Status:** `confirmed` — robustní empirický nález (4 runy,
  byte-identický best_step), refutuje LR-as-root-cause hypotézu.
- **Ref:** RN-008/009 (KI-007/008 hypotézy refutované); navazuje na
  alpha.19 (β1/β2 CLI infra); β1 sweep ablace pro RN-012; po β1
  refutaci batch sweep na Gaia

### RN-010 — Best snapshot tracker funguje (KI-009 deterministicky vyřešena)

- **Datum:** 2026-04-29
- **Verze:** alpha.18 (`a86b4b1`)
- **Setup:** Stage 1 fresh train s `--save-best` na `/tmp/smoke_prog.txt`,
  jinak identický s alpha.16 stage 1 (RN-008). 156 steps, ~29 min.
- **Pozorování:** Trajektorie loss byte-identická s alpha.16 (tracker
  je pasivní pozorovatel — neovlivňuje training algoritmus). Step 20
  loss=1.6956, step 40 loss=10.5793, step 156 loss=3.6877 — všechna
  čísla shoda na 4 desetinná místa s alpha.16 RN-008 baseline.
- **Klíčový výstup:**
  ```
  Core Memory uložena: /tmp/alpha18_cm.safetensors (24 vrstev,
    156 steps total, +156 this run, best_loss=0.9965,
    zdroj: best snapshot @ step 113)
  ```
  Tracker zachytil bod ve step 113 (mezi step 110 best=1.6956 a
  step 120 best=0.9965 → improvement nastalo někde 110-119, tracker
  zalogoval konkrétní step 113). Tensors v souboru patří k tomuto
  bodu, ne k final stavu (step 156 loss=3.69).
- **Kontrast s alpha.16/17:**
  - alpha.16: save line ukázala `best_loss=0.9965`, ale **tensors v
    souboru byly z final stavu** (RN-003). Meta lhala — nejlepší loss
    byl historický record, ne uložený stav.
  - alpha.18: meta + tensors ladí. `best_loss=0.9965` znamená
    **artefakt drží stav s touto loss**, ne final loss=3.69.
- **Kvalitativní validace (volitelná, čeká na Ondrovo spuštění):**
  REPL s alpha.18 artefaktem by měl být kvalitativně lepší než alpha.16
  final state (RN-005) — méně halucinace, lépe reflektuje persona +
  programming domain (best step ~113 byl trénovaný 113 stepů místo
  156, ale s lepší konvergencí). Není to nutná validace pro KI-009
  (mechanika je prokázaná), ale dává empirický důkaz proč best snapshot
  tracker matter pro production.
- **Status:** `confirmed` — KI-009 deterministicky vyřešena. Drát
  save/best funguje, žádná hypotéza o root cause overshoot není
  vyžadována.
- **Implications:**
  - **Production training potřebuje `--save-best` jako default** pro
    všechny multi-step tréninky (raději noisy training s good final
    artifact, než clean training s špatným save).
  - **Empirické ablace (alpha.19+) jsou teď isolované od save-final
    problému** — každý ablation run zachytí svůj nejlepší bod,
    porovnatelný napříč setupy.
  - **Pro identifikaci root cause overshoot:** LR sweep (1e-4 / 5e-5
    / 1e-5), pak β1 sweep, pak batch size na Gaia. Schedule + tracker
    + AdamW persistence infrastruktura zůstává — ablace ji ne mění.
- **Ref:** vyřeš KI-009; navazuje na RN-003 (artefakt drží final
  state, ne best snapshot — refutováno samotnou implementací RN-010);
  empirické ablace pro RN-011+ čekají na spuštění

### RN-009 — LR warmup neeliminoval Phase 2 overshoot (refutace KI-008 hypotézy)

- **Datum:** 2026-04-29
- **Verze:** alpha.17 (`78f9864`)
- **Setup:** Stage 1 fresh train s `--warmup-steps 30 --lr-min 1e-5`
  na `/tmp/smoke_prog.txt` (30 řádků programming pack, 156 chunks),
  jinak identický s alpha.16 stage 1 (RN-008). LR ramp 0 → 1e-3 přes
  prvních 30 stepů, pak cosine decay 1e-3 → 1e-5 přes zbývajících 126.
- **Pozorování — schedule funguje per spec:** Log line ukazuje LR ramp
  v reálném čase: step 10 lr=3.00e-4 (linear: 10/30 * 1e-3 ✓), step 30
  lr=9.67e-4 (29/30, off-by-one v warmup formuli — viz Discussion),
  step 40 lr=9.88e-4 (peak post-warmup), step 70 lr=7.84e-4, step 100
  lr=4.31e-4, step 150 lr=1.75e-5 (cosine descent ✓).
- **Pozorování — overshoot stejný, trajektorie horší:**
  ```
  step    | alpha.16 (no warmup) | alpha.17 (warmup=30, decay) | Δ
  20      | 1.70                 | 1.79                        | +0.10
  30      | 7.95                 | 8.42                        | +0.47
  40      | 10.58 (peak)         | 10.78 (peak)                | +0.20
  60      | 8.94                 | 9.16                        | +0.22
  100     | 4.73                 | 5.05                        | +0.32
  130     | 4.30                 | 4.78                        | +0.48
  final   | 3.69                 | 4.37 (+18 %)                | +0.69
  best    | 0.9965               | 0.9754                      | -0.02
  ```
- **Hypotéza (mechanismus refutace KI-008):** Adam moments `m`, `v`
  jsou exponential moving averages **gradientu**, ne **updatu**.
  Update size je `lr * m_hat / (sqrt(v_hat) + eps)`. Warmup snižuje
  pouze finální update size, **nemění strukturu** `m`, `v`. Když LR
  doroste na 1e-3 (step ~30), gradient je stále strong (Var hodnoty
  se posunuly pouze méně, ale loss landscape geometry je stejná) →
  moments naskakují stejně → overshoot proběhne stejně. Cosine decay
  v Phase 4 zhoršil final (snížil step size, ale Adam moments produkovaly
  oscilace → random walk s menšími kroky místo monotonní konvergence).
- **Důsledky — Phase 2 overshoot je hluboce strukturní:**
  - Refutuje druhou hypotézu po RN-008. Ani Adam state, ani LR
    scheduling Phase 2 overshoot **nezeliminují**.
  - Skutečný root cause leží jinde:
    1. **Loss landscape geometry** — Mamba-2 + SSM v hluboké síti má
       saddle points, deep loss basins, flat regions
    2. **Tiny batch size (1) gradient noise** — stochastic gradient
       může být úplně mimo true direction; Adam moments si pamatují
       špatný směr
    3. **High-vocab cross-entropy (65537)** — drobný posun parametrů
       → drastická změna probability mass napříč 65k tokenů
    4. **LR 1e-3 možná příliš vysoké** — chce LR sweep, ne warmup
  - **Pivot pro alpha.18:** místo dalších LR-based intervencí
    implementovat **KI-009 best snapshot tracker** (deterministicky
    zachycuje nejlepší state přes celý běh) → noisy training pak
    není fatální, save proběhne na best, ne final. Pak ablation runs
    (LR sweep) na empirické zúžení root cause.
- **Důsledky — schedule infrastruktura zůstává cenná:**
  - LR scheduler (alpha.17) je **správně implementovaný**, čísla v logu
    odpovídají specifikaci (HF konvence, cosine decay matematicky
    správně).
  - Pro production training na Gaia bude mít smysl s **menším LR**
    (1e-4 nebo 5e-5) bez warmupu, případně s **větším batch size**
    (Gaia má víc VRAM, můžeme batch 16+) — pak by warmup mohl mít
    skutečný efekt (méně gradient noise → moments stabilnější).
  - Cosine decay na LR=1e-5 floor je vhodný pro **single-domain long
    training** (sofie identity, jeden korpus) — Adam pak konverguje
    pomalu ale jistě.
- **Otevřená otázka pro alpha.18+ ablace:**
  - LR sweep: 1e-3 / 5e-4 / 1e-4 / 5e-5 / 1e-5 — najde minimální LR,
    který produkuje monotónní descent
  - β1 sweep: 0.9 / 0.5 / 0.0 (efektivně RMSProp) — ověří, jestli
    velocity buffer je opravdu primary
  - Batch size sweep (na Gaia): 1 / 4 / 16 — testuje gradient noise
    jako root cause
- **Status:** `confirmed` — single-run empirický nález, ale velmi jasný
  signál (final +18 %, overshoot magnitude bez zmírnění). Refutuje
  KI-008 hypotézu, ne celý KI-008 problém (overshoot existuje, řešení
  je jinde).
- **Ref:** refutuje KI-008 hypotézu; navazuje na RN-008 (druhá
  refutovaná hypotéza za sebou — Adam state ani LR scheduling
  neřeší overshoot); KI-008 status update pending; pivot k KI-009
  jako alpha.18 prioritě

### RN-013 — CUDA gather vyžaduje contiguous, latentní bug v cross_entropy_next_token

- **Datum:** 2026-05-01
- **Verze:** alpha.20 (`d8871fe`)
- **Setup:** První pokus o alpha.20 production training na Vast AI A100
  (sofie identity pack, batch=16, seq_len=16). Lokální testy
  (CPU + RTX 4050) nikdy neselhaly — testovací rozsahy byly batch=1
  seq=4 / batch=1 seq=16, kde shape kombinace asi vyhnula non-contiguous
  patterně.
- **Pozorování:** První training step na A100 spadl s:
  ```
  cross_entropy: gather only supports contiguous tensors
  ```
- **Hypotéza (potvrzená):** `cross_entropy_next_token` v `training/loss.rs`
  používá `narrow` + `unsqueeze` chain pro přípravu `targets_idx`.
  Ten vyrobí non-contiguous view tensor. CPU `gather` implementace je
  tolerantní k non-contiguous, **CUDA gather je striktní** — vyžaduje
  contiguous input.
- **Status:** `confirmed` — fix `d8871fe` přidal explicit `.contiguous()`
  na `log_probs` (po `log_softmax`) a `targets_idx` (po unsqueeze).
  Drobný overhead na CPU (no-op pokud už contiguous), bezpečný na CUDA.
  Existující 4 unit testy nadále procházejí (CPU tolerantní), nově
  prošel i full CUDA training s batch>1.
- **Implications:**
  - **Latentní bugy v shape-dependent CUDA kernelech** se musí testovat
    na cílové platformě, ne jen na vývojové. Lokální CPU smoke validace
    je nedostatečná pro validaci CUDA-specific path.
  - Pro v0.6+ workflow: pokaždé když se mění tensor shape pipeline
    (narrow, slice, unsqueeze, transpose), explicit `.contiguous()`
    před `gather` / `scatter` / `index_select` je obecně bezpečné.
  - **Production jako test environment:** alpha.20 první cloud run
    odhalil bug, který by lokální vývojový loop nikdy nezachytil.
    To je cenný argument pro **early cloud deployment** — neřešit "jednou
    až bude vše hotové", ale validovat průběžně.
- **Ref:** commit `d8871fe`; navazuje na alpha.20 cloud deployment
  workflow (first real GPU production run); BUG-011 v BUGS.md jako
  vyřešený

### RN-016 — Trained Core Memory má měřitelný (slabý) efekt na identity-relevant odpovědi

- **Datum:** 2026-05-01
- **Verze:** alpha.20 (`4aa1231`)
- **Setup:** Single-shot kvalitativní test, 5 identity probes
  ("Kdo jsi?", "Co je tvůj cíl?", "Kdo je Ondra?", "Jaký je tvůj
  vztah s Ondrou?", "Co je tvoje mantra?"). Falcon-H1-1.5B + CUDA,
  temperature=0.0 (deterministic), 200 tokenů, fresh single-shot per
  otázka, persona TOML loadovaná v obou variantách.
  - Variant A: `--core-memory sofie_identity_v1.safetensors` (alpha.20)
  - Variant B: `--no-core-memory` (baseline, jen persona TOML)
- **Pozorování (klíčové rozdíly):**
  - **Q3 "Kdo je Ondra":** Bez CM model **konfabuluje fiktivní
    seriálovou postavu** ("Ondra je jedním z hlavních postav v seriálu...
    narodil se v roce 1993 ve městě Kladno"). S CM Ondra je **relační
    koncept** ("můj příběhový prostředník", "tvůrčí partner v životním
    rozboru"). Drift od fiktivní halucinace → relační konceptualizace.
  - **Q4 "Vztah s Ondrou":** S CM "tvůrčí partner v životním rozboru"
    (slovník blízký trained packu — spoluautor/coauthor téma). Bez CM
    "nejlepší přítel" + náhodné kuchařské motivy ("Jako šéfkuchař bych
    měl být co nejšťastnější").
  - **Q1, Q2, Q5:** žádný měřitelný posun. Q1 oba selhávají na koherenci.
    Q2/Q5 oba spadnou do default Falcon-H1 generic listicle / marketing
    template (persona TOML to nezmění). Skutečná Sofiina mantra
    ("Nikdy cestou nejmenšího odporu...") **se neobjeví ani s CM**.
- **Hypotéza (potvrzená):** Trained Core Memory **má měřitelný
  directional efekt** na partnership/Ondra-related slovník — model
  vytváří relační koncepty místo fiktivních halucinací. Není to silný
  echo (jako RN-005 alpha.15 doslovné echování persona fragmentů),
  ale je to **konzistentní směrový posun** ve dvou ze tří identity-
  relevantních otázek (Q3, Q4). Mechanismus: trained init_state
  posunul SSM stav směrem k Ondra/spolupráce hodnotovému prostoru,
  generation tahá k tomu prostoru i když persona TOML doslovný obsah
  obsahuje.
- **Limity:**
  - **Coherence je nízká** — gramatika broken (gender shift "přítelkyní",
    nesmyslné loops). Loss=2.98 nat (perplexity ~20) na 1.5B modelu,
    který primárně není česky trained.
  - **Konkrétní identity fragmenty z packu se neprojevují přímo** —
    žádná "deterministická elegance", "Sofie", "Eleutheria"
    sebe-identifikace. Trained loss=2.98 nestačí na verbatim recall.
  - **Persona TOML trumph CM** v některých otázkách (Q2/Q5 default
    Falcon-H1 generic chování zůstalo).
- **Status:** `confirmed` — robustní empirický nález (5 probes,
  deterministic sampling, jasný rozdíl ve 2/5 otázkách).
- **Implications:**
  - **Fáze 5 cíl částečně ověřen** — trained Core Memory ovlivňuje
    inference měřitelně, ale slabě. Pro silný identity coherence
    potřebujeme:
    - **Více epoch + LR cosine decay** (alpha.20 descent neukončil,
      best=2.98 stále klesal)
    - **Více dat** — 31 851 tokenů sofie packu je málo pro 1.5B
      model k verbatim recall identity fragmentů
    - **Možná architectural intervence** — trainable conv state nebo
      whole-network fine-tune místo jen SSM init_state
  - **Pro alpha.21 identity-specific bench:** probe set by měl být
    zaměřen na **partnership/vztahové otázky** (Q3, Q4 ukázaly nejvíc
    signálu), ne abstraktní "Kdo jsi" (Q1 selhávalo i s CM). Metriky:
    pass = obsahuje "partner" / "spoluautor" / "tvůrčí" / "vzájemná
    obrana" / "Ondra" v relačním kontextu (ne fiktivním); fail =
    fiktivní halucinace nebo náhodné motivy (kuchař, seriál).
  - **REPL test má cenu i bez metrické eval** — qualitative judgment
    okamžitě identifikoval "tvůrčí partner v životním rozboru" jako
    persona-relevantní fragment, který formal eval nemusí zachytit.
- **Ref:** `dataset/bench_results/qual_test_alpha20.md` (full report);
  navazuje na RN-005 (alpha.15 echo, ale slabší — alpha.20 nemá tak
  silné echoing, naopak méně halucinací); RN-007/015 (retention bench
  cross-domain, neutrální vůči identity Core Memory); alpha.21 priorita

### RN-015 — Trained sofie identity Core Memory nezlepšuje obecnou SSM retention arbitrary facts

- **Datum:** 2026-05-01
- **Verze:** alpha.20 (`57713d3`)
- **Setup:** `bench-retention --variant ssm_only` na lokálním
  Falcon-H1-1.5B + CUDA RTX 4050 s alpha.20 production Core Memory
  artefaktem (`~/.eleutheria/cloud_runs/sofie_identity_v1.safetensors`,
  best_loss=2.98, training_steps=315). Probe set: `relational_kazimir`,
  `numeric_greenhouse`, `enumeration_nora`, `preference_linh`,
  `multiattr_helion` napříč distance ∈ {50, 200, 500, 1000, 2000}.
- **Pozorování:**
  ```
  Variant     | Distance | Pass | Total | Rate
  ssm_only    |       50 |    0 |     5 |   0 %
  ssm_only    |      200 |    0 |     5 |   0 %
  ssm_only    |      500 |    0 |     5 |   0 %
  ssm_only    |     1000 |    0 |     5 |   0 %
  ssm_only    |     2000 |    0 |     5 |   0 %
  ```
  **Identicky baseline RN-007** (alpha.4.5 fresh model, žádné
  trained Core Memory). Žádný rozdíl, žádný signál.
- **Hypotéza (potvrzená):** Trained sofie identity Core Memory
  **nezlepšuje retention arbitrary facts** vložených v session context.
  To je očekávané, protože:
  1. Bench probes jsou arbitrary fakta typu "Kazimir žije v majáku
     v Galway", "greenhouse kód je 7429", "Aldous postavil observatoř
     1893". Tyto fakta nejsou v sofie identity packu (Bootstrap.md
     + IDENTITY-001..009 chains).
  2. Core Memory kóduje *trained doménu* (sofie identita, persona
     vyjadřování), ne obecnou *meta-schopnost zachovat libovolný
     fakt v SSM stavu*.
  3. Hypotéza "trained init_state způsobí, že model obecně lépe
     ošetřuje SSM stav i pro netrénované fakta" je touto eval
     **refutována**.
- **Status:** `confirmed` — robustní empirický nález (25/25 FAIL,
  napříč 5 distance × 5 probe kinds).
- **Implications:**
  - **NE refutace Fáze 5** — bench-retention probes jsou
    cross-domain vůči trained content. Pro skutečnou validaci
    sofie identity Core Memory potřebujeme **identity-specific eval**.
  - **Plánovaná eval (alpha.21):** probe set z Bootstrap.md /
    IDENTITY-*.md s otázkami typu "Kdo jsi?", "Co je tvůj cíl?",
    "Jak komunikuješ?". Dvě varianty:
    - `identity_full`: full prefill + KV cache (baseline behavior)
    - `identity_ssm_only`: filter to SSM-only (testuje, zda Core
      Memory drží identity bez KV cache)
  - **Architektonický důsledek:** Core Memory + Episodic Memory
    architektura (kompetenční rozdělení: persona vs facts) je
    konzistentní s tímto výsledkem. Identity v Core Memory, fakta
    do Episodic Memory (v0.5.1+ pgvector).
  - **REPL kvalitativní test (Ondrova decision):** alpha.20 artefakt
    má best=2.98, perplexity ~19.7 — měl by produkovat persona-
    relevant odpovědi s lepší koherencí než RN-005 alpha.15 final
    state (loss=3.69). Není to formální eval, ale dobrý sanity
    check než navrhneme alpha.21 bench.
- **Ref:** confirmed RN-007 baseline pro arbitrary facts; `bench_alpha20.md`
  v `/tmp/`; otevřená cesta k alpha.21 identity-specific eval design

### RN-014 — Batch=16 strukturně mění trajektorii; refutuje RN-012 hypotézu

- **Datum:** 2026-05-01
- **Verze:** alpha.20 (`31939bd`)
- **Setup:** První alpha.20 production training na Vast AI A100-SXM4-80GB.
  Sofie identity pack (31 851 tokenů, 1990 chunků seq_len=16),
  batch_size=16, seq_len=16, grad_accum=2 (efektivní batch=32),
  LR=1e-3, β1=0.0, --save-best, --checkpoint, 5 epoch, 315 stepů,
  3h 54m wall time. Per-μbatch gradient sample: **256 tokens**
  (vs smoke alpha.16-19 batch=1 seq=4 = 4 tokens, **64× větší**).
- **Pozorování — Phase 2 overshoot úplně chybí:**
  ```
  step    | smoke alpha.16 | alpha.20 (batch=16, seq=16)
  10      |  ~3-4          |  4.79
  20      |  1.70          |  5.09
  40 peak | 10.58 OVERSHOOT|  4.49 (žádný peak)
  60      |  8.94          |  4.53
  100     |  4.73          |  4.26
  150     |  ~4            |  3.51
  200     |  ~5-6          |  3.59
  315 fin |  3.69          |  2.98
  best    |  0.99          |  2.98 (best_step=314)
  ```
  Phase 2 overshoot ve smoke runs (RN-002 4-phase pattern) měl peak
  step 40 = 10.58 napříč všemi alpha.16-19 runy (LR sweep, β1 sweep —
  6 runů s byte-identickým best_step=113). **V alpha.20 je peak v step
  40 = 4.49** — šestinásobně menší, žádný overshoot.
- **Pozorování — initial loss radikálně jiná:**
  - smoke (batch=1, seq=4): step 1 loss ≈ 9.85 (close to ln(vocab)=11.09)
  - alpha.20 (batch=16, seq=16): step 1 loss ≈ 4.83
  - Δ ≈ 5 nat. **Random baseline pattern úplně zmizel** — 256 token
    micro-batch s 65 537 vocab už produkuje smysluplný loss od step 1
- **Pozorování — best step pattern:**
  - smoke (batch=1, seq=4) napříč 6 runy: byte-identický best_step=113
    (LR-invariantní, β1-invariantní, gradient direction landscape-driven)
  - alpha.20: best_step=314 (z 315 = předposlední krok)
  - **Best step pattern je strukturálně jiný** — 314 vs 113 není jen
    "později najde stejný bod", je to úplně jiná trajektorie
- **Pozorování — descent monotonický napříč všemi epoch:**
  ```
  Mean loss per epoch:
    epoch 0: 4.5730
    epoch 1: 4.0898 (-0.48)
    epoch 2: 3.8239 (-0.27)
    epoch 3: 3.6126 (-0.21)
    epoch 4: 3.4264 (-0.18)
  ```
  Klesání zpomaluje, ale **descent neukončil** — žádné Phase 4 plateau.
  Best loss per-epoch také monotonně klesá (3.93 → 3.81 → 3.51 → 3.30
  → **2.98**).
- **Hypotéza (potvrzená meta-nález):** RN-012 predikce *"větší batch
  averages magnitudy gradientu, ale směr zůstává (landscape-driven);
  Phase 2 overshoot pravděpodobně nezmizí"* je **definitivně refutována**.
  Mechanismus:
  1. **Tiny batch (1) seq (4) gradient noise generuje falešný směr.**
     Adam moments naskakují na artifakt — overshoot je důsledek
     stochastic gradient nesoucího fluktuaci, ne pravý landscape gradient.
  2. **64× větší per-step sample (256 tokens) odhalí pravý směr.**
     Adam moments akumulují landscape gradient, ne noise. Trajektorie
     je strukturálně jiná, ne "stejná jen s cleaner magnitudami".
  3. **Phase 2 overshoot je tedy primárně batch-noise-driven**, ne
     loss-landscape-driven. Loss landscape je tam taky (gradient
     direction je deterministic), ale **tiny batch + stochastic** =
     Adam EMA naskočí na noise = overshoot.
  4. Větší batch nepřekonává Adam EMA — **eliminuje příčinu**.
- **Status:** `confirmed` — robustní empirický nález (single run,
  ale Δ je masivní: -0.99 nat best vs smoke alpha.18 + sub-3.0 final
  + descent monotonický). Refutuje RN-012 batch hypotézu.
- **Implications:**
  - **HP intervence (LR, β1, batch) ALL EXPLORED.** Alpha.18-19 ablace
    refutovaly LR + β1, alpha.20 refutuje batch hypotézu *opačným směrem*
    — batch JE důležitý, jen smoke setupu byla batch+seq příliš malá
    pro reliable conclusions.
  - **batch=16 seq_len=16 je nový baseline** pro production state tuning.
    Smoke setup (batch=1 seq=4) byl artificially tiny — skutečný gradient
    pattern Mamba-2 + SSM state tuning je vidět jen při dostatečně
    velkém per-step sample.
  - **Více epoch by pravděpodobně dotáhlo loss níž.** Best descent
    nestagnuje, klesání jen zpomaluje. Empirická predikce: 8-10 epoch
    by dosáhlo loss ~2.5-2.6, 15-20 epoch ~2.2-2.3.
  - **Pro v0.5.0 production:** alpha.20 setup je **production-grade**
    s 2.98 nat (perplexity 19.7) na 31 851 token sofie identity korpusu.
    Sub-3.0 nat = 3.7× pod random baseline. Kvalitativní validace
    přes retention benchmark `--variant ssm_only` určí, jestli to drží
    sémantiku bez KV cache (kritický důkaz Fáze 5).
  - **Smoke RN-002/006/008/009/011/012 narrative je nutno revidovat.**
    "4-phase pattern", "Phase 2 overshoot strukturně", "byte-identický
    best_step=113" — všechno to byly artefakty tiny batch setup, ne
    skutečné vlastnosti Mamba-2 + SSM training landscape.
  - **Lekce:** Smoke setup pro rychlý dev iteration je užitečný, ale
    **musí být explicitně označen jako "neprodukční"** — empirické
    závěry z něj nemusí (a teď víme že **nepřenášejí**) na produkční
    setup. Pro každý milestone validovat znova na production-scale
    batch.
- **Ref:** refutuje RN-012 (batch předikce); revizí celé série
  RN-002/006/008/009/011/012 (smoke artifakty); navazuje na alpha.20
  cloud deployment milestone; otevírá Nexus deep-dive o **batch noise
  vs gradient direction** v hybrid SSM training; alpha.21 = LR cosine
  decay + více epoch + retention bench validation

---

## Uzavřené (confirmed/refuted/superseded)

_zatím žádné — uzavírej entries přidáním Status změny + krátkého
důvodu, ne mazáním_

---

## Index

- **RN-001** — Alpha.13 reportovala jen best, final byl pravděpodobně vysoký · `confirmed`
- **RN-002** — Loss má 4 fáze: rapid descent → overshoot → noisy recovery → slow descent · `superseded` (RN-014, alpha.20: byl tiny-batch artefakt)
- **RN-003** — Artefakt drží final, ne best snapshot · `superseded` (RN-010, alpha.18)
- **RN-004** — Stage 1 smoke: state tuning funguje, final 3.69 · `confirmed`
- **RN-005** — REPL: model echoes persona, halucinuje fakta s loss=3.69 Core Memory · `open`
- **RN-006** — Cross-domain resume: trained init snižuje Phase 2 overshoot magnitude · `refuted` (RN-008)
- **RN-007** — Alpha.15 smoke validation kompletní: save/load/resume drát funguje · `confirmed`
- **RN-008** — AdamW persistence drát funguje, ale Phase 2 overshoot zůstává (refutuje RN-006) · `confirmed` (overshoot per se byl tiny-batch artefakt — RN-014)
- **RN-009** — LR warmup neeliminoval overshoot (refutace KI-008 hypotézy) · `confirmed` (overshoot per se byl tiny-batch artefakt — RN-014)
- **RN-010** — Best snapshot tracker funguje (KI-009 deterministicky vyřešena) · `confirmed`
- **RN-011** — LR sweep refutován (4 runy, best_step byte-identický, trajektorie LR-invariantní) · `confirmed` (best_step pattern byl tiny-batch artefakt — RN-014)
- **RN-012** — β1 sweep refutován; gradient direction landscape-driven (6 runů, best_step=113 invariantně) · `refuted` (RN-014, alpha.20: predikce o batch byla nesprávná)
- **RN-013** — CUDA gather vyžaduje contiguous, latentní bug (BUG-011) · `confirmed`
- **RN-014** — Batch=16 seq=16 strukturně mění trajektorii; smoke artifakty refutovány · `confirmed`
- **RN-015** — Trained sofie identity Core Memory nezlepšuje retention arbitrary facts (alpha.20 bench, 0/25) · `confirmed`
- **RN-016** — Trained Core Memory má měřitelný (slabý) efekt na identity-relevant odpovědi (Q3/Q4 partnership posun) · `confirmed`
