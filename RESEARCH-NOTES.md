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
- **Status:** `open` — Phase 1-4 model je popisný, ne mechanistický.
  Otázka: byla by trajektorie monotonní s LR warmup? S menším LR? Bez
  Adamu (SGD)? Toto chce ablaci.
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
