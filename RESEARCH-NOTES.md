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

### RN-002 — Loss osciluje noisy bez stabilního trendu (Adam × tiny batch)

- **Datum:** 2026-04-29
- **Verze:** alpha.15 (`b37e79a`)
- **Setup:** `/tmp/smoke_prog.txt` (30 řádků programming_pack, ~624
  tokenů, 156 chunků), seq_len=4, batch=1, grad_accum=1, lr=1e-3,
  grad_clip=1.0, AdamW (default ParamsAdamW), CUDA RTX 4050 6 GB,
  gradient checkpointing ON
- **Pozorování:** Loss trajektorie:
  ```
  step  20: loss=1.70   best=1.70
  step  40: loss=10.58  best=1.70
  step  60: loss=8.94   best=1.70
  step  80: loss=7.69   best=1.70
  step 100: loss=4.73   best=1.70
  step 120: loss=6.85   best=0.9965  ← nový rekord někde mezi 100-119
  ```
  Loss osciluje v rozsahu 1–11, **bez monotonického trendu**. Best=0.99
  byl ephemerálně dosažen, pak ztracen.
- **Hypotéza:** Tři synergické faktory:
  1. `batch=1, grad_accum=1` → každý gradient je odhad z jediné `seq_len=4`
     sekvence = vysoký šum
  2. AdamW velocity buffer (`v` = squared grad EMA) zesiluje šum: po
     warmup naskakuje velocity v noisy směrech, kroky jsou velké i když
     skutečný gradient direction není konzistentní
  3. Loss landscape Core Memory state tuningu je členitý — malé změny
     `init_state` Var → velké změny v cross-entropy loss
  Warmup nepomůže fundamentálně (řeší jen prvních ~50 kroků), pomůže
  až **větší effective batch** (Gaia, alpha.16+ s persistovanou AdamW
  state pro grad_accum > 1, nebo SGD bez momentu).
- **Status:** `open` — hypotéza je rozumná, ale neověřená ablací
  (potřebovali bychom: lr sweep, batch size sweep, SGD vs Adam srovnání)
- **Implications:**
  - **Production training na RTX 4050 nemá smysl** — noise dominuje
  - Gaia (větší VRAM → větší batch) je nutné pro smysluplnou konvergenci
  - Pro smoke test stačí, že **state tuning umí dosáhnout loss < 1.0**
    (důkaz, že kód není rozbitý a workflow je feasibilní)
  - LR warmup (alpha.16+ candidate) odstraní jen krajní oscilace
- **Ref:** RN-001 (alpha.13 reprodukuje stejný pattern); plánovaná
  ablace v Nexus deep-dive po Gaia deploy

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

## Uzavřené (confirmed/refuted/superseded)

_zatím žádné — uzavírej entries přidáním Status změny + krátkého
důvodu, ne mazáním_

---

## Index

- **RN-001** — Alpha.13 reportovala jen best, final byl pravděpodobně vysoký · `confirmed`
- **RN-002** — Loss osciluje noisy (Adam × tiny batch) · `open`
- **RN-003** — Artefakt drží final, ne best snapshot · `confirmed`
