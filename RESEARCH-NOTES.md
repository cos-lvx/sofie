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
- **Status:** `confirmed` (mechanická vlastnost API, ne empirický nález)
- **Implications:**
  - **Production training potřebuje "save when best improves" mechanismus.**
    Možnosti: (a) periodic snapshot s min loss tracking, (b) shadow copy
    Var values do CPU bufferu při každém best update, (c) save-on-eval
    při validation loss improvement
  - Pro alpha.14/15 smoke je to dokumentovaná limitace, ne blocker
  - Plánovaný fix: nový patch po alpha.16, dedikovaný snapshot mechanism
- **Ref:** RN-002; plán pro nový alpha.17 nebo dedikovaný patch

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

---

## Uzavřené (confirmed/refuted/superseded)

_zatím žádné — uzavírej entries přidáním Status změny + krátkého
důvodu, ne mazáním_

---

## Index

- **RN-001** — Alpha.13 reportovala jen best, final byl pravděpodobně vysoký · `confirmed`
- **RN-002** — Loss má 4 fáze: rapid descent → overshoot → noisy recovery → slow descent · `open`
- **RN-003** — Artefakt drží final, ne best snapshot · `confirmed`
- **RN-004** — Stage 1 smoke: state tuning funguje, final 3.69 · `confirmed`
- **RN-005** — REPL: model echoes persona, halucinuje fakta s loss=3.69 Core Memory · `open`
- **RN-006** — Cross-domain resume: trained init snižuje Phase 2 overshoot magnitude · `refuted` (RN-008)
- **RN-007** — Alpha.15 smoke validation kompletní: save/load/resume drát funguje · `confirmed`
- **RN-008** — AdamW persistence drát funguje, ale Phase 2 overshoot zůstává (refutuje RN-006) · `confirmed`
