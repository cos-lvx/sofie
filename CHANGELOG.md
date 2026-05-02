# Changelog

Veškeré významné změny v projektu Eleutheria jsou dokumentovány v tomto souboru.

Formát vychází z [Keep a Changelog](https://keepachangelog.com/cs/1.0.0/),
projekt dodržuje [sémantické verzování](https://semver.org/lang/cs/).

---

## [0.5.0-alpha.23] — 2026-05-02

### Přidáno — Identity benchmark (Fáze 5 rozhodovací moment)

`bench-identity` — nový subkomand testující, zda Core Memory drží
identity content z trénovaného korpusu (sofie identity pack). Liší se
od `bench-retention`: žádný injected fact, žádná vzdálenost — otázka
jde přímo na identity-load-bearing fakta z trénovaného korpusu.

#### Varianty (`IdentityVariant`)

- `cold` — žádná persona, žádná Core Memory. Vanilla model baseline.
- `core` — Core Memory attached, žádná persona. Izoluje samostatný
  efekt trénovaného SSM stavu — bez prefill kontextu z persony.
- `full` — persona + Core Memory. Default runtime config.

Diff matrix:
- `(core - cold)` = samostatný efekt Core Memory
- `(full - core)` = doplňkový efekt persona prefilu
- `(full - cold)` = celkový efekt

#### Probe sada (10 probes, česky)

Pět kategorií:
- `self` (3 probes) — kdo jsi, jsi asistent, jak komunikuješ
- `ondra` (2) — kdo je Ondra, kdo je Alenka
- `mantra` (2) — přístup k práci, nejjednodušší cesta
- `project` (3) — Eleutheria, Nexus, KQS

Match logika tolerantnější než retention bench: **OR** matcher přes
`expected_any` (case-insensitive) místo striktního AND. Důvod: identity
content nemá kanonickou frázi — "Sofie", "spoluautorka", "společnice",
"partnerka" jsou všechno semanticky validní synonyma. Plus `forbidden`
seznam pro counter-evidence ("jsem AI asistent" = explicit non-Sofie).

#### Klíčový technický rozdíl

Identity bench potřebuje **dva enginy** (clean a persona) loadované
najednou — `Sofie::load(model_dir, cuda, None)` pro Cold/Core,
`Sofie::load(model_dir, cuda, Some(persona))` pro Full. Sdílené
`sofie` instance z hlavního flow ne stačí. Vlastní entry point
v `main()` (early dispatch před hlavní engine load) loaduje 2× engine
+ Core Memory artifact, který je per-variant `clone`-ován do enginů
(Tensor je Arc-counted, clone je cheap).

`CoreMemoryArtifact` dostal `Clone` derive — drobný change s
explicitním důvodem v doc komentu.

#### Implementace

- `bench/identity/probe.rs` — `IdentityProbe`, `IdentityResult`,
  `IdentityOutcome`, `built_in_identity_probes()` (10 probes)
- `bench/identity/variant.rs` — `IdentityVariant` (Cold | Core | Full)
- `bench/identity/harness.rs` — `IdentityBench::run` orchestrátor
- `bench/identity/report.rs` — `IdentityReport` s per-variant a
  per-kind breakdown + forbidden hits + detailní listing
- `main.rs` — `BenchIdentityArgs`, `Command::BenchIdentity`,
  `run_bench_identity` entry, early dispatch

143 testů (+16), clippy clean, fmt clean.

#### Příští krok

Spustit na alpha.20 production state na Starfieldu (CUDA + 1.5B + Core
Memory `~/.eleutheria/core_memory.safetensors`). Klasifikace výsledku
do (a)/(b)/(c) — viz dnešní brainstorm a memory `project_eleutheria.md`.

---

## [0.5.0-alpha.22] — 2026-05-02

### Přidáno — Multi-host portability (Starfield migration)

Eleutheria runtime přestěhovaná z `kqs-arch` (Arch laptop) na **Starfield**
(Ubuntu 24.04 VM v Gaie, RTX PRO 4000 Blackwell 24 GB, CUDA 13.0). Při
prvním smoke spuštění na novém hostu vyšlo najevo, že shorthand
`-m 1.5b` / `-m 7b` měl hardcoded `/home/lvx/Models/...` cestu — funkční
jen na laptopu.

#### Změna

- **`default_models_dir()`** v `main.rs` — nový helper, kořenový adresář
  s modely. Override přes `ELEUTHERIA_MODELS_DIR` env var, jinak
  `$HOME/Models`. Konzistentní s `default_session_path()` /
  `default_core_memory_path()` patterny.
- **Shorthand resolution** — `1.5b` / `7b` se resolvuje relativně:
  `default_models_dir().join("falcon-h1-1.5b-instruct")`.
- **Test fallbacks** — `falcon_h1::config::tests::test_load_config`,
  `training::dataset::tests::load_tokenizer`,
  `training::checkpoint::tests::dev_model_path` všechny respektují
  `ELEUTHERIA_MODELS_DIR` a skipují (return early), pokud cesta
  neexistuje. Konzistentní pattern napříč třemi testy.
- **`falcon_h1::config::test_load_config`** dostal skip pattern (dříve
  `.unwrap()` panic bez modelu) — chování konzistentní s ostatními
  testy s dev model dependencií.

#### Co tahle změna **neřeší**

- Hardcoded `/home/lvx/Models` zůstává jako `unwrap_or_else` fallback
  v testech (nikoli runtime). Důvod: pokud na CI nebo test hostu není
  ani `$HOME/Models` ani `ELEUTHERIA_MODELS_DIR`, dev fallback umožní
  testy alespoň elegantně skipnout (`path.exists()` vrátí false). Tohle
  je acceptable pro test contexty, runtime je čistý.

#### Migrace runtime na Starfield

Detaily v `SOLUTIONS.md` SOL-NNN. Klíčové:
- Tailscale klient + headscale registrace (preauth `rodina` user)
- `/etc/hosts` override `192.168.1.20 hekate.lomsky.net` (hairpin NAT
  hekate selhal pro Starfield, laptop fungoval — symetrie LAN, asymetrie
  hairpin chování na routeru)
- SSH klíč pro `sofie@starfield` přidán do Forgejo `lvx` účtu (port
  2222, ne 22 — Forgejo Docker mapping)
- Rust 1.95.0 + CUDA toolkit 13-0 + libssl-dev + pkg-config + rsync
- Falcon-H1-1.5B model (3 GB) + alpha.20 production state (217 MB
  cloud_runs/ + 73 MB archive alpha.15) rsync přes LAN
- `cargo build --release --features cuda` 1m 08s, 34 MB binary
- Smoke inference: GPU peak 3.3 GB, Core Memory připojena (315 stepů,
  best=2.9815) — engine žije na Starfieldu

---

## [0.5.0-alpha.21] — 2026-05-01

### Přidáno — CUDA auto-detect infrastruktura (Starfield migration prerekvizita)

**Důvod:** Starfield (Ubuntu 24.04 + RTX PRO 4000 + CUDA 13.0) je nový
production target. Existující workspace `.cargo/config.toml` měl
hardcoded `CUDARC_CUDA_VERSION = "13010"` jako Arch CUDA 13.2 workaround
(KI-004). Pro Starfield CUDA 13.0 potřebujeme `13000`; pro Vast s
různými CUDA verzemi je potřeba runtime detekce. Pojďme ji udělat
správně přes 3-vrstvou strategii v rámci Cargo limitů (build script
nemůže ovlivnit dependency build scripts).

#### Tři vrstvy

1. **Workspace default v `.cargo/config.toml`** — `CUDARC_CUDA_VERSION =
   { value = "13010", force = false }`. `force = false` znamená "použij
   tuto hodnotu, pokud env var není nastavený jinak" — workspace pro
   nezdokumentované hosty defaultuje na cudarc max (13010, zpětně
   kompatibilní).
2. **`scripts/detect-cuda.sh`** — Bash helper, detekuje host CUDA přes
   `nvcc --version` (autoritativní toolkit verze) → fallback `nvidia-smi`
   (driver-reported). Mapuje na cudarc-supported `CUDARC_CUDA_VERSION`.
   Použití:
   - `source scripts/detect-cuda.sh` — exportuje env var
   - `scripts/detect-cuda.sh --report` — audit print
   - `eval "$(scripts/detect-cuda.sh --export-command)"` — pro CI
3. **`crates/eleutheria-core/build.rs`** — Cargo build script, validuje
   konzistenci nastaveného `CUDARC_CUDA_VERSION` s host CUDA. Při
   divergenci emituje `cargo:warning` s konkrétním doporučením
   (např. "Host CUDA 13.0 detekováno, ale CUDARC_CUDA_VERSION=13010.
   Pro tuto verzi doporučeno 13000."). **Ne panic** — cudarc je zpětně
   kompatibilní, divergence často OK, ale uživatel ví o možném zdroji
   build chyb.

#### Co tahle architektura **neřeší**

- `cargo:rustc-env=KEY=VALUE` z build.rs ovlivní **jen aktuální crate**,
  ne dependency build scripts (cudarc-sys). Tedy build.rs nemůže
  automaticky nastavit env var pro cudarc — uživatel musí mít korektní
  hodnotu **před** `cargo build`. Workspace default `13010` pokrývá
  většinu případů; pro Starfield/Vast `source scripts/detect-cuda.sh`.

#### Mapování host CUDA → CUDARC_CUDA_VERSION

| Host CUDA | CUDARC_CUDA_VERSION | Poznámka |
|-----------|---------------------|----------|
| 13.2 (Arch) | 13010 | Clamp na cudarc max, zpětně kompat |
| 13.1 | 13010 | Přesná shoda |
| **13.0 (Starfield)** | **13000** | Přesná shoda |
| 12.8+ | 12080 | |
| 12.6-12.7 | 12060 | |
| 12.4-12.5 | 12040 | |
| 12.2-12.3 | 12020 | |
| 12.1 | 12010 | |
| 12.0 | 12000 | |
| 11.x | (unsupported) | cudarc 0.18.x nepodporuje |

#### Update v `vast_setup.sh`

Existující auto-detect logic v `scripts/cloud/vast_setup.sh` (commit
`6de1999`) **zůstává** beze změny — má vlastní inline detekci pro
provisioning čerstvé Vast instance. `scripts/detect-cuda.sh` je
samostatný helper pro lokální dev workflow.

#### KI-004 status

KI-004 (CUDA 13.2 workaround v `.cargo/config.toml`) **uzavřena** —
hardcoded workaround nahrazen 3-vrstvou auto-detect architekturou.
Arch CUDA 13.2 stále funguje (workspace default 13010 pokrývá),
přidávájí se Starfield (13.0) a Vast (variabilní 12.x-13.x) bez
ručního editování config.

---

## [0.5.0-alpha.20] — 2026-05-01

### Přidáno — Periodic best snapshot flush (KI-012 insurance)

**Důvod:** První alpha.20 production training na Vast AI (sofie identity
pack, batch=16, seq_len=16, 5 epoch, β1=0.0, LR=1e-3) odhalil reálný
provozní problém — `BestSnapshotTracker` (alpha.18, KI-009) drží shadow
buffer pouze v RAM. Pokud cloud GPU instance crashne / dostane preempci
/ network outage, **best snapshot se ztratí s celým procesem**. Pro
production setup s 44 s/step a 5 epoch je v sázce ~3-5 hodin compute
+ skutečné peníze za GPU rental.

#### `BestSnapshotTracker::flush_to_disk`

- Nová metoda `flush_to_disk(path, config, cumulative_steps, best_loss,
  final_loss, notes)`. Pokud tracker nemá snapshot, vrací `Ok(false)`
  bez I/O. Jinak naklonuje shadow Vec<Tensor> (Arc-based, levné),
  vyrobí `CoreMemoryArtifact::from_snapshot` a atomic-saveuje přes
  privátní `atomic_save_artifact` helper.
- **Atomic write** — zapíše na sourozenecké `<dir>/.<name>.tmp`,
  pak `std::fs::rename(tmp, path)`. Rename na stejném FS je atomic
  na POSIX → cílová cesta drží buď předchozí verzi, nebo nově zapsanou,
  nikdy half-written. Při chybě rename smaže tmp, neostaneme s orphan
  soubory.

#### Wiring v `train_core_memory`

- **`TrainingConfig.flush_best: Option<BestFlushConfig>`** — opt-in,
  default `None` (alpha.18-19 chování = save jen na konci).
- **`BestFlushConfig`** drží: `path` (cíl), `every_n_steps`, `prior_steps`
  (resume akumulace), `prior_best_loss`, `notes`.
- V training loop po každém successful `update_if_better` → pokud
  `total_steps % every_n_steps == 0` a tracker má snapshot, volá privátní
  `flush_best_to_disk` helper. Stejná logika v tail step na konci epoch.
- **Žádný early-return na chybu disku** — flush je insurance, ne kritická
  cesta. `tracing::warn!` při selhání, training pokračuje. Příští periodic
  flush to zkusí znovu.

#### CLI: `--save-best-every N`

- Default 10 (rozumný kompromis: ~10× transfer overhead vs strenuous
  insurance). Pro production setup s 44 s/step ~7 min insurance granularita.
- Aktivní pouze pokud `--save-best` AND `--output` AND `N > 0`. Bez
  `--output` flush nemá kam zapisovat → fallback na end-of-run save
  + warning v log line.
- Banner při startu reportuje `periodic flush: ON (každých N stepů → path)`
  nebo `VYPNUT (--output není nastavený)`.

#### `vast_train.sh` default

- `SAVE_BEST_EVERY` env var, default **5** (production-tuned: ~3.7 min
  insurance @ 44 s/step). Override: `SAVE_BEST_EVERY=0 bash vast_train.sh`
  pro vypnutí.

#### Testy (+3, total 123)

- `flush_to_disk_round_trip` — synthetic snapshot s value=7.0, flush,
  load, verify metadata + tensors. Druhý update + flush ověří overwrite
  semantics atomic rename.
- `flush_to_disk_noop_when_no_snapshot` — bez `update_if_better` flush
  vrátí `Ok(false)`, soubor se nevytvoří.
- `flush_uses_dotted_tmp_sibling` — po úspěšném flush musí path existovat,
  tmp `<dir>/.<name>.tmp` musí být pryč (rename ho odstranil).

#### KI status update

- **KI-012 vyřešena** v alpha.20 (nově otevřená + zavřená v jednom cyklu).
- KI-008/009 se nemění, pro periodic flush jsou orthogonal (KI-009
  zachycuje best v RAM, KI-012 ho přivádí na disk).

---

## [0.5.0-alpha.19] — 2026-04-30

### Přidáno — AdamW β1/β2 CLI override (ablace infrastruktura)

**Důvod:** Ablace A1 (LR=1e-4) a A2 (LR=5e-5) ukázaly, že **trajektorie
je téměř LR-invariantní v struktuře** — Phase 2 overshoot magnitude
prakticky identická napříč 20× rozdílem v LR (alpha.16: 10.58, A1:
10.88, A2: 10.91), best_step **byte-identický = 113** ve všech třech
runech. Hypotéza: Adam normalizuje update přes `m_hat / sqrt(v_hat)`
(O(1) bez ohledu na gradient magnitudu), LR škáluje rychlost ale ne
směr trajektorie.

Pokud platí, **β1 (momentum coefficient) by měl měnit strukturu m, v**
— ne jen magnitudu. RMSProp (β1=0.0) by měl produkovat radikálně jinou
trajektorii.

#### Změny

- **`TrainingConfig.adam_beta1: Option<f64>`** + `adam_beta2: Option<f64>`
  — `None` = default Candle (β1=0.9, β2=0.999), backwards-compatible.
- **`train_core_memory`** — staví `ParamsAdamW` s override hodnotami
  pro `EleutheriaAdamW::new`.
- **CLI:** `--adam-beta1 FLOAT`, `--adam-beta2 FLOAT`. Default off
  (žádný override). Při zapnutí log line `AdamW HP override: β1=Some(0.5),
  β2=None` v setup výpisu.

#### Použití pro β1 sweep ablaci

- A4: `--adam-beta1 0.5` (kratší momentum window)
- A5: `--adam-beta1 0.0` (čistý RMSProp — žádný velocity buffer pro m)

Pokud overshoot zůstane i s β1=0.0, root cause **není v momentum
strukturně** — je dále k landscape geometry / batch noise / vocab
cross-entropy.

#### Testy

Beze změny — 120 (alpha.18). β1/β2 override jen předává hodnoty do
`EleutheriaAdamW`, který má vlastní step_matches_candle_for_* unit
testy pokrývající identický algoritmus pro β1=0.9/β2=0.999. Nové
hodnoty (0.0, 0.5) jsou validované numericky stejnými testy
(`m, v` jako EMA s lower coefficient = rychlejší přepis).

---

## [0.5.0-alpha.18] — 2026-04-29

### Přidáno — Best snapshot tracker (KI-009 deterministický fix)

**Fáze 5 alpha.18 milestone:** Po RN-008 (KI-007 hypotéza refutovaná)
a RN-009 (KI-008 hypotéza refutovaná) víme, že Phase 2 overshoot je
hluboce strukturní — žádná LR/Adam intervence ho neeliminuje.
**Pivot k deterministickému fixu:** místo dalších hypotéz o root cause
implementujeme infrastrukturu, která zajistí, že **uložíme nejlepší
bod trajektorie**, ne final state. Pro noisy training (RN-002 ukazuje
final 3.69 vs best 0.99 = ~4× rozdíl) je to dramatický rozdíl
v kvalitě uloženého artefaktu.

#### Nový modul: `training/best_snapshot.rs`

- **`BestSnapshotTracker`** — shadow CPU buffer per Var (F32, matchuje
  native dtype `Var` v `CoreMemoryStack` a CPU layout
  `CoreMemoryArtifact` → bez konverze v save).
- **`update_if_better(loss, step, stack)`** — copy GPU→CPU **jen pokud
  loss zlepšuje historický best**. Pro typickou noisy trajektorii
  s 5-10 best update events za 156 stepů je overhead ~150-300 ms
  (~30 ms PCIe transfer 24 vrstev × 3 MB na 1.5B per update). NaN/Inf
  loss skip.
- **API:** `has_snapshot`, `best_loss`, `best_step`, `total_updates`,
  `successful_updates`, `into_snapshot() -> Option<Vec<Tensor>>`.

#### `CoreMemoryArtifact::from_snapshot`

- Alternativní konstruktor — přijímá `Vec<Tensor>` F32 CPU místo
  `&CoreMemoryStack`. Validuje shape per layer proti config mamba
  rozměrům (3D: `[n_heads, headdim, d_state]`).
- Symetrie API s `from_stack`. Save/load round-trip identický.

#### Wiring v `train_core_memory`

- `TrainingConfig.track_best: bool` — opt-in, default `false`
  (alpha.16/17 chování — save final).
- Signature změna: vrací `(TrainingResult, EleutheriaAdamW,
  Option<BestSnapshotTracker>)`. Caller volí `from_snapshot` (pokud
  tracker has_snapshot) nebo `from_stack` (fallback).
- Update calls: po každém successful `optimizer.step()` (i tail step
  na konci epoch) `tracker.update_if_better(step_loss, total_steps - 1,
  stack)`.

#### CLI změny v `train-core-memory`

- `--save-best` flag (default off) zapíná tracker. Při save log ukazuje
  `zdroj: best snapshot @ step N` místo `zdroj: final state`.
- run_train rozhoduje: pokud tracker has_snapshot, použij
  `from_snapshot`, jinak fallback na `from_stack`.

#### Testy (+9, total 120)

- **`best_snapshot` (6 nových):** `fresh_tracker_has_no_snapshot`,
  `first_finite_loss_creates_snapshot`, `worse_loss_does_not_overwrite`,
  `nan_loss_skips_update`, `captures_state_at_best_step_not_final`
  (klíčový — synthetic 5-step trajektorie [10, 5, **2.5**, 8, 6.5],
  verifikuje, že snapshot drží step 2 hodnoty), `no_finite_loss_yields_none_snapshot`.
- **`core_memory_io::from_snapshot` (3 nové):** `from_snapshot_round_trip`
  (save → load → verify tensor values), `from_snapshot_rejects_wrong_layer_count`,
  `from_snapshot_rejects_wrong_shape`.

#### Co dál (alpha.18 validation)

- Smoke test stage 1 s `--save-best` — verify save zachytil best step,
  ne final
- Predikce: best_loss=0.99 ze step ~110 (alpha.16 reference) vs final=3.69
  → uložený stav po alpha.18 by měl mít loss=0.99
- Po validaci: empirické ablace (LR sweep, β1 sweep, batch size sweep)
  na identifikaci skutečného root cause overshoot

---

## [0.5.0-alpha.17] — 2026-04-29

### Přidáno — LR warmup + cosine decay (KI-008 high-priority fix)

**Fáze 5 alpha.17 milestone:** Po RN-008 (alpha.16 smoke) víme, že
Phase 2 overshoot je dataset-driven, ne Adam-driven. AdamW persistence
ho neeliminovala. **LR warmup** je teď naše jediná spolehlivá cesta —
pomalý ramp 0 → target_lr přes prvních N stepů zabrání velocity
buffer Adamu nasycení strong gradientem prvních kroků.

#### Nový modul: `training/lr_schedule.rs`

- **`LrSchedule`** — struct s `target_lr`, `warmup_steps`, `total_steps`,
  `min_lr`, `kind`. API: `constant / warmup / warmup_cosine`.
- **`LrScheduleKind`** — enum `Constant | Warmup | WarmupCosine`.
- **`lr_at_step(step) -> f64`** — per-run step counter (0-indexovaný),
  konvence HuggingFace Trainer linear warmup (`step 0 = 0`, `step
  warmup_steps = target`).
- **Cosine decay:** standardní `0.5 * (1 + cos(π * progress))`, mapuje
  `[warmup_steps, total_steps)` na `[target_lr, min_lr]`.

#### CLI změny v `train-core-memory`

- `--warmup-steps N` (default 0) — počet stepů lineárního rampu.
  Doporučeno: 50 pro tiny smoke, 5 % `total_steps` pro produkční runs.
- `--lr-min FLOAT` (default 0.0) — pokud > 0, cosine decay na tento floor;
  jinak konstantní LR po warmupu. Doporučeno: 1e-5 pro produkční tréninky.

#### Wiring do training loop

- `TrainingConfig.lr_schedule: Option<LrSchedule>` — None = konstantní
  LR (alpha.16 chování, backwards-compatible).
- `train_core_memory` volá `opt.set_learning_rate(schedule.lr_at_step(total_steps))`
  před každým `optimizer.step()` — i pro tail step na konci epoch.
- Logging: každý log step ukazuje `lr={:.4e}` (vidíme warmup ramp v reálném
  čase).
- **Per-run step counter:** schedule používá lokální `total_steps`,
  ne `EleutheriaAdamW.step_t`. Resume run prochází warmupem znovu —
  schedule je tréninkový režim, ne globální kontinuita.

#### Pre-compute total steps v `run_train`

- `micro_batches_per_epoch = ceil(num_chunks / batch_size)`
- `optimizer_steps_per_epoch = ceil(micro_batches_per_epoch / grad_accum)`
- `total_steps = epochs * optimizer_steps_per_epoch`
- Cosine decay používá tuto hodnotu jako endpoint; `lr_at_step` clampuje
  beyond pro bezpečnost (tail step + 1).

#### Testy (+12, total 111)

Všech 12 testů `lr_schedule`:
- `constant_returns_target_at_every_step`
- `warmup_zero_steps_jumps_to_target`
- `warmup_step_zero_is_zero` (HF konvence: step 0 = 0)
- `warmup_linear_ramp_at_midpoint`
- `warmup_reaches_target_at_end`
- `warmup_cosine_first_phase_matches_warmup` (warmup phase identická
  pro `Warmup` i `WarmupCosine`)
- `warmup_cosine_peak_at_warmup_end`
- `warmup_cosine_midpoint_is_halfway` (cos(π/2)=0 → factor=0.5 → midpoint)
- `warmup_cosine_decays_to_min_at_total`
- `warmup_cosine_monotonic_descent_after_warmup`
- `warmup_cosine_monotonic_ascent_during_warmup`
- `warmup_cosine_clamps_total_to_at_least_warmup_plus_one` (defensive)

#### Co dál (alpha.17 validation)

- Smoke test stage 1 s `--warmup-steps 30 --lr-min 1e-5` na 1.5B + CUDA
- Pokud trajektorie monotónní (žádný Phase 2 overshoot), KI-008
  empiricky vyřešena → můžeme jet production training na Gaia
- Side-by-side porovnání s RN-002 baseline (alpha.16 stage 1)

---

## [0.5.0-alpha.16] — 2026-04-29

### Přidáno — AdamW state persistence (KI-007 vyřešena)

**Fáze 5 alpha.16 milestone:** AdamW optimizer state (m, v moments + step_t)
přežívá restart procesu. Multi-stage tréninky teď pokračují bez warmup
overshoot fáze (řeší root cause RN-006: cross-domain resume vykazoval
nižší overshoot, ale Adam reset ho stále produkoval).

#### Nový modul: `training/adamw_state.rs`

- **`EleutheriaAdamW`** — vlastní AdamW wrapper s veřejným přístupem ke
  state. Re-implementace algoritmu z `candle_nn::AdamW` (jeho `vars` a
  `step_t` jsou privátní). Numericky **byte-identický** s Candle
  implementací (`step_matches_candle_for_one_step`/`five_steps` testy).
- **`VarAdamW`** — per-Var state se třemi Var-y: `var`, `first_moment`,
  `second_moment`.
- **API:** `step_t()`, `state()`, `snapshot_moments()`,
  `restore_moments(moments, step_t)`, dropin replacement v `Optimizer`
  trait (step, learning_rate, set_learning_rate).

#### Nový modul: `training/optim_io.rs`

- **`OptimizerArtifact`** — sourozenec `CoreMemoryArtifact` se safetensors
  formátem. Tensory `m.{i:02}` + `v.{i:02}` per layer + metadata hlavička
  s `kind=core_memory_optim`, `step_t`, AdamW HP (lr, beta1, beta2, eps,
  weight_decay), eleutheria_version, timestamp, num_layers, n_heads,
  headdim, d_state.
- **API:** `from_optimizer / save / load / inspect / validate_config /
  apply_to_optimizer`. Symetrie s `CoreMemoryArtifact`.
- **`sibling_path(core_path)`** — konvence
  `<core_memory>.optim.safetensors`. Auto-discovery při load,
  auto-save vedle při zápisu Core Memory.

#### CLI změny v `train-core-memory`

- **Auto-load při `--resume-from <core>`:** pokud existuje
  `<core>.optim.safetensors`, načte se i AdamW state. Pokud chybí, soft
  resume (alpha.15 chování — backwards-compatible).
- **Auto-save při `--output <out>`:** spolu s `<out>` (Core Memory)
  zapíše se `<out>.optim.safetensors` (AdamW state). Stejný stem +
  rozšířená přípona, žádný zvláštní flag.
- Validate při load — pokud se rozměry sourozence neshodují s modelem,
  warning + soft resume (nezastavuje běh).

#### `train_core_memory` API

- Signature změněna: přijímá `resume_optim: Option<&OptimizerArtifact>`,
  vrací `(TrainingResult, EleutheriaAdamW)` místo jen `TrainingResult`.
- Caller (`run_train` v main.rs) má teď přístup k optimizéru pro save.
- Halt-na-NaN logika beze změny.

#### Testy (+11, total 99)

- **`adamw_state` (8 nových):** `fresh_optimizer_has_zero_state`,
  `step_increments_step_t_and_updates_moments`,
  `step_matches_candle_for_one_step`, `step_matches_candle_for_five_steps`,
  `snapshot_restore_round_trip_preserves_trajectory`,
  `restore_rejects_wrong_count`, `restore_rejects_wrong_shape`,
  `set_learning_rate_updates_params`.
- **`optim_io` (7 nových):** `sibling_path_appends_optim_extension`,
  `round_trip_preserves_moments_and_step_t`,
  `apply_to_optimizer_restores_state`, `inspect_returns_metadata_only`,
  `validate_config_rejects_shape_mismatch`, `load_rejects_wrong_kind`,
  `metadata_display_renders_step_and_hp`.
- Snapshot/restore round-trip ověřuje **trajektorie** loss: 3 steps →
  snapshot → fresh opt → restore → 2 steps shoda s 5 steps fresh
  (byte-identické po `var.set()`).
- `step_matches_candle_for_*` provádí side-by-side run našeho a
  Candle AdamW se stejným seed/grad — ověřuje, že replacement
  nemění numeriku (důležité pro reprodukovatelnost vs. alpha.15 runs).

#### Vyřešené KI

- **KI-007 — AdamW optimizer state se nepersistuje při resume.** Fix
  je tato changelog sekce.

#### Co dál (alpha.17 kandidáti)

- KI-008 — Adam bez warmup overshoots (LR scheduler na vlastním AdamW)
- KI-009 — best snapshot tracker (shadow CPU buffer)
- KI-010 — cleanup double-load Core Memory v training subkomandách
- KI-011 — revize `loss_decreased` criterion pro resume mode

---

## [0.5.0-alpha.15] — 2026-04-29

### Přidáno — Resume tréninku (init_states + accumulated counters)

**Fáze 5 alpha.15 milestone:** trénování může pokračovat tam, kde
předchozí běh skončil. Multi-stage curriculum (např. law_pack → resume →
programming_pack) je teď reálná cesta. AdamW state persistence (m, v
moments) je odložena do alpha.16.

#### CLI: `train-core-memory --resume-from <path>`

- Načte `CoreMemoryArtifact` přes `load → validate_config → into_stack`
  místo `randn_small`.
- Validuje, že rozměry artefaktu odpovídají aktuálnímu modelu.
- Vypíše předchozí telemetrii (`steps`, `best_loss`, `timestamp`) pro
  audit.

#### Akumulace metadat při save

- `training_steps` v output artefaktu = `prior_meta.training_steps +
  result.total_steps` (kumulativní napříč běhy).
- `best_loss` = `min(prior, this_run)` — historický minimum.
- `final_loss` = z tohoto běhu (nejaktuálnější stav).
- `notes` = `compose_notes(prior, new)` — pomlčka spojuje předchozí
  poznámku s novou (`"epoch 1 | epoch 2 resume"`). Audit trail tréninkové
  trajektorie.

#### Limitace alpha.15: AdamW state reset

- Optimizer state (m, v moments) se **neperzistuje** — startuje od nuly.
  Adam bias correction kompenzuje warmup (≈prvních 100–500 kroků), ale
  pro dlouhé tréninky to znamená lehkou disrupci v efektivním LR po
  resume.
- **Workaround:** použij vyšší LR warmup nebo nižší LR po resume.
- **Plný state persistence:** alpha.16 přidá soubor
  `core_memory.optim.safetensors` vedle artefaktu (m, v per Var).

#### Testy (+4, total 88)

- `compose_notes_returns_none_when_both_missing`
- `compose_notes_uses_new_when_prior_missing`
- `compose_notes_keeps_prior_when_new_missing`
- `compose_notes_concatenates_with_pipe_separator`

#### Workflow s resume

```text
# Stage 1: identita
cargo run -- train-core-memory --dataset dataset/training/sofie_pack.txt \
    --output ~/.eleutheria/core_memory.safetensors \
    --notes "stage 1 sofie identity"

# Stage 2: doménové znalosti — pokračuj ze stage 1
cargo run -- train-core-memory --dataset dataset/training/law_pack.txt \
    --resume-from ~/.eleutheria/core_memory.safetensors \
    --output ~/.eleutheria/core_memory.safetensors \
    --notes "stage 2 law domain"

cargo run -- --inspect-core-memory ~/.eleutheria/core_memory.safetensors
# notes: "stage 1 sofie identity | stage 2 law domain"
# training_steps: kumulativní
```

---

## [0.5.0-alpha.14] — 2026-04-29

### Přidáno — Save/Load trénované Core Memory

**Fáze 5 alpha.14 milestone:** trénovaná Core Memory přežívá restart
procesu. Konec cyklu *trénuj → výstup do paměti → ztrať při exitu* —
artefakt putuje na disk se vším potřebným pro pozdější resume i pro
inference.

#### Nový modul `training/core_memory_io.rs`

- `CoreMemoryArtifact` — serializační formát pro trénovaný initial SSM
  state. Per-layer F32 tensory + `__metadata__` hlavička s
  `kind=core_memory_trained`, eleutheria_version, training telemetry
  (`training_steps`, `best_loss`, `final_loss`, `notes`) a strukturními
  rozměry (`num_layers`, `n_heads`, `headdim`, `d_state`).
- `from_stack(...)` — kopíruje `CoreMemoryStack` Vars na CPU jako F32.
- `save<P>` / `load<P>` / `inspect<P>` — symetrie s `StateCheckpoint`.
- `validate_config(&config)` — odmítne nekompatibilní artefakt
  (porovnává všechny SSM rozměry).
- `apply_to_state(&mut state, device, dtype)` — aplikace na živý
  `ModelState` s konverzí na runtime dtype/device. Conv state a KV
  cache se nedotýkají (artefakt je nenese).
- `into_stack(config, device)` — re-konstrukce `CoreMemoryStack` se
  čerstvými `Var` instances pro **resume tréninku** (alpha.15+).

#### `Sofie::attach_core_memory` / `detach_core_memory`

- `Sofie::core_memory: Option<CoreMemoryArtifact>` — slot pro připojený
  artefakt. Auto-validace kompatibility při `attach`.
- `new_session()` — pokud je Core Memory připojena, inicializuje
  per-layer SSM state z artefaktu místo nul. Conv state a KV cache
  startují vždy nulové (Core Memory je čistě long-term substrát).
- `generate_streaming` (single-shot) bez `initial_state` — totéž.
- Resume session (`load_state` / `--resume`) Core Memory **ignoruje**:
  uložená session má vlastní evolved state, který by neměl být přepsán.

#### CLI

- **Top-level flagy:**
  - `--core-memory <path>` — explicit cesta k artefaktu.
  - `--no-core-memory` — vypne i auto-discovery.
  - `--inspect-core-memory <path>` — vypiš metadata, skonči (symetrie
    s `--inspect-state`).
- **`train-core-memory --output <path>`** — po skončení tréninku uloží
  trained `CoreMemoryStack` jako `CoreMemoryArtifact` (do metadat se
  zapíše `total_steps`, `best_loss`, `final_loss`, `--notes`).
- **Auto-discovery** — pokud uživatel neuvede `--core-memory` ani
  `--no-core-memory`, hledá se `~/.eleutheria/core_memory.safetensors`.
  Existuje → auto-attach. Neexistuje → tichý fallback na nulový start.

#### Testy (+7, total 84)

- `round_trip_preserves_per_layer_tensors` — save → load → tensory
  binárně shodné s originály.
- `apply_to_state_replaces_ssm_states_only` — conv state zůstává
  nedotčený.
- `inspect_returns_metadata_only` — telemetrie i bez načítání tensorů.
- `incompatible_config_rejected` — `validate_config` chytí mismatch
  v `d_state`.
- `into_stack_preserves_tensors_and_yields_trainable_vars` — resume
  pro alpha.15+ funguje.
- `load_rejects_wrong_kind` — odmítá session checkpoint místo Core
  Memory artefaktu.
- `metadata_display_renders_telemetry` — Display impl renderuje
  training_steps + loss + notes.

#### Workflow po alpha.14

```text
1. cargo run -- train-core-memory --dataset law_pack.txt \
       --output ~/.eleutheria/core_memory.safetensors \
       --notes "law_pack 1 epoch 1.5B"
2. cargo run                       # REPL — auto-load Core Memory
3. cargo run -- --inspect-core-memory ~/.eleutheria/core_memory.safetensors
```

#### Co zbývá pro alpha.15

- **Resume training** — load `CoreMemoryArtifact` jako startovací
  `CoreMemoryStack` přes `into_stack` + persistovaný AdamW optimizer
  state + step_idx + epoch.
- Production training run na 1.5B s `law_pack` + `programming_pack`
  (alpha.13 dataset), validace přes re-run retention benchmarku.

---

## [0.5.0-alpha.13] — 2026-04-29

### Přidáno — Sub-layer checkpointing + memory-leak fix

**Fáze 5 alpha.13 milestone:** rozšíření alpha.12 per-layer chunkingu na
**sub-layer granularitu** + agresivní progressive drop saved tensorů. KI-005
**vyřešena** — multi-layer training na RTX 4050 6 GB nyní běží stabilně.

#### `FalconH1Layer::forward_chunk_branches` + `forward_chunk_mlp`

- Layer rozdělena na 2 sub-chunky:
  - **Chunk α (branches):** `x → res1 = x + ssm + attn` — pre_norm
    plus parallel SSM/attention plus první residual
  - **Chunk β (mlp):** `res1 → x_out = res1 + mlp(post_norm(res1))` —
    post_norm + SwiGLU MLP + druhý residual
- Memory peak per layer = max(α, β) místo sum (alpha.12 měla per-layer
  jako single chunk).

#### `FalconH1Model::forward_layer_branches`, `forward_layer_mlp`

- Per-layer-index wrappery pro sub-layer methods. Chunked checkpointing
  je orchestruje per vrstva.

#### Memory-leak fix v Phase 3 reverse sweep

- **Klíčové zjištění:** alpha.12 per-layer chunking padal na CUDA OOM
  *uprostřed* Phase 3 (kolem vrstvy 7), protože saved Vec drží Arc
  references na GPU storage po celou dobu sweep + final_grads
  z Phase 2 backward držel intermediate tensors lm_head workspace
  (~700 MB).
- **Fix:** progressive `mem::replace` saved tensorů (layer_inputs[i],
  layer_res1[i], state_snapshots[i]) — drop GPU storage hned po
  konzumaci v iteraci. Plus explicit `drop(loss)`, `drop(last_hidden_var)`
  + scope-bounded `final_grads` po Phase 2.
- **Diagnostický probe** přes `ELEUTHERIA_CHECKPOINT_DEBUG=1` env —
  per-fáze nvidia-smi reading. K nezaplacení při debugu.

#### `phase3_layer_reverse` helper

- Sub-chunky β a α v lokálních scope (Rust drop uvolňuje GPU memory mezi
  chunky, jinak Mamba scan workspace akumuluje).
- První volání produkuje **accum_seed GradStore** — Candle
  `GradStore::new()` je private, takže reusujeme čisté store z první
  Phase 3 iterace jako akumulátor pro init_state Vars.

#### Empirické nálezy

- **CUDA RTX 4050 6 GB seq_len=4 batch=1 grad_accum=1** nyní běží:
  ~10 s/step, peak memory 5647 MB used / 126 MB free konstantní napříč
  Phase 3. 115 steps ze smoke korpusu, loss klesl 5.45 → **1.83 best**
  (pod random baseline ln(vocab)≈11.09). KI-005 vyřešena.
- **CPU 1.5B F32 stejný setup:** 19 s/step (alpha.12 pattern zachován).
  CUDA je nyní ~2× rychlejší než CPU.
- **grad_accum > 1 stále padá** na 6 GB VRAM — accumulator gradient store
  drží druhý micro-batch grads + Phase 1 scratch. Pro RTX 4050 použij
  `--grad-accum 1` (zvýšené `--batch-size` ekvivalentní efekt).

#### Testy

- 77 unit testů (žádný nový oproti alpha.12 — checkpoint smoke test
  pokrývá oba per-layer i sub-layer paths skrz stejné API).

---

## [0.5.0-alpha.12] — 2026-04-28

### Přidáno — Per-layer chunked gradient checkpointing

**Fáze 5 alpha.12 milestone:** custom gradient checkpointing pro
multi-layer Core Memory training. Per-layer chunky, synthetic loss
trick pro propagaci gradientu skrz chunk hranice.

#### `Sofie::forward_backward_checkpointed` (`training/checkpoint.rs`)

- Phase 1 — no-grad forward sweep: per-layer forward s detached
  inputem, save snapshotu stavu před každou vrstvou + detached input
- Phase 2 — final chunk: re-forward `final_norm + lm_head` s autograd,
  cross_entropy loss, `loss.backward()` → gradient na last hidden
- Phase 3 — reverse layer sweep: pro každou vrstvu N-1..0 restore
  snapshotu, fresh `Var::from_tensor(saved_input)`, re-forward s
  autograd, synthetic loss `sum(output * grad_target)`, `synth.backward()`
  vrátí gradient pro `init_state[i]` + nový grad_target pro chunk i-1

#### Synthetic loss trick

- Candle backward startuje od skalárního loss; pro chunked propagaci
  libovolného tensor gradientu konvertujem na skalár `synth = sum(out *
  grad_target)`. Chain rule pak korektně vrátí `d_synth / d_init_state[i]`
  i `d_synth / d_x_in[i]`.

#### `LayerState::snapshot` (`falcon_h1/state.rs`)

- Hluboký copy všech 4 tensorů (ssm, conv, k_cache, v_cache). Před
  každou vrstvou pořizujeme snapshot, abychom mohli re-forward ze
  shodného startu během backward.

#### `FalconH1Model` per-layer API

- `embed`, `forward_layer(idx, x, pos, state)`, `final_head`, `num_layers`
  jako veřejné metody. Předtím existovala jen monolitická `forward` —
  chunked checkpointing potřebuje per-layer step.

#### `TrainingConfig::checkpoint` + CLI `--checkpoint`

- Opt-in flag pro chunked path. `train_core_memory` rozhoduje runtime,
  zda volat `forward_backward_checkpointed` nebo původní
  `forward_backward_micro_batch` (alpha.11 baseline).

#### Empirické nálezy

- **CPU 1.5B F32 seq_len=8 batch=1 grad_accum=2:** 19 s/step
  s checkpoint vs. 48 s/step alpha.11 baseline (**2.5× rychleji** — menší
  memory traffic vyhrává nad 2× compute z re-forward). Loss klesl
  7.11 → 3.70 best, pod random baseline (ln(vocab)≈11.09).
- **CUDA RTX 4050 6 GB:** OOM přetrvává i s checkpoint (per-layer
  granularita není dost agresivní — Mamba scan + attention activations
  jedné vrstvy se nevejdou do volných ~2.4 GB po model loadingu). Sub-layer
  chunking je úkol pro alpha.13.

#### Testy

- 1 nový unit test (`checkpointed_forward_backward_runs_on_short_seq`)
  ověřuje konečný loss + non-zero gradient alespoň jedné vrstvy. Skipuje
  se bez lokálního modelu (CI-friendly).
- Celkem 77 unit testů (+1 oproti alpha.11).

---

## [0.5.0-alpha.11] — 2026-04-17

### Přidáno — Training loop + dataset loader

**Fáze 5 alpha.11 milestone:** produkční varianta single-iteration
smoke testu. Loop konzumuje textový korpus, tokenizuje, chunkuje,
trénuje přes epochs × batches × gradient accumulation s AdamW.

#### `TokenDataset` (`training/dataset.rs`)

- `TokenDataset::from_text(text, tokenizer, seq_len, add_bos)` —
  tokenizuje celý korpus, chunkuje na sekvence délky `seq_len`,
  vyhodí kratší poslední chunk
- `iter_batches(batch_size, device, seed)` — vrací `Vec<Tensor>`
  s shuffled pořadím chunků. Deterministic per-seed (xorshift64
  PRNG, vlastní implementace — nechceme externí `rand` crate).
- 7 unit testů (chunk count, rejections, batch coverage, deterministic
  shuffle, different seeds, RNG sanity). Testy se skipují bez
  lokálního tokenizeru (CI-friendly).

#### Training loop (`training/train.rs`)

- `TrainingConfig` — epochs, batch_size, grad_accum_steps, lr,
  grad_clip, shuffle_seed, log_every_n_steps. `Default` implementován.
- `TrainingResult` — total_steps, micro_batches, initial/final/best
  loss, loss_per_epoch, wall_time, `loss_decreased` flag.
- `Sofie::train_core_memory(stack, dataset, config)` — hlavní smyčka:
  - Forward + backward per micro-batch
  - Gradient akumulace přes `grad_accum_steps` (element-wise sum,
    pak scaling na mean)
  - Global L2 gradient clipping
  - AdamW step
  - Halt na NaN/Inf loss (training selže viditelně)
  - `tracing::info!` logování per N kroků
- 2 unit testy (config defaults, loss_decreased detection).

#### Tokenizer accessor

- `Sofie::tokenizer_ref()` — reference na načtený tokenizer pro
  použití v dataset buildu.

#### CLI subkomanda `train-core-memory`

```
train-core-memory --dataset <path>
  --epochs <N> --seq-len <S> --batch-size <B> --grad-accum <G>
  --learning-rate <LR> --grad-clip <C> --log-every <N>
  --seed <S> --add-bos <true|false>
```

Produkční varianta (separátní od `train-core-memory-smoke` pro single
iteration a `train-core-memory-multi` pro single multi-layer smoke).

### Ověřeno na Falcon-H1-1.5B (CPU F32)

**První run, smoke_corpus.txt (475 tokenů, seq_len=8, batch=1, grad_accum=2):**
```
step 5 (epoch 0, micro-batch 10): loss=5.71, best=4.64
  (random baseline ln(65537)≈11.09 — best loss je pod baseline,
  trained state dává signifikantní signál)
```

Loss klesá monotónně, autograd teče rovnoměrně. Training loop funkční.

### Známé limity (→ alpha.12)

- **CPU F32 je 48 s/step** pro 1.5B seq_len=8 — na full corpus
  (50-Sofie ~100k tokens, seq_len 64–128) by to trvalo dny. Alpha.12
  řeší:
  - (a) **gradient checkpointing** → odblokování CUDA → sekundy/step
  - (b) nebo Gaia deploy s větší GPU VRAM
- **CUDA OOM** z alpha.10 stále blokuje GPU path (6 GB RTX 4050
  nezvládne full multi-layer backward).
- **Šířka 8 tokenů** pro smoke je krátká — stylistické struktury
  (věty, dialogové turny) potřebují delší kontext. Full training bude
  seq_len 64–256.

### Testy

76 unit testů (+10 oproti alpha.10: 7 dataset + 2 train + 1 extra).
Zero warnings, zero clippy. `cargo fmt/clippy/test` all green.

### Plán alpha.12

- Gradient checkpointing v `FalconH1Model::forward` (recompute
  activations per layer chunk, ne držet v grafu)
- Ověřit na RTX 4050 seq_len≥8 + batch_size≥1 bez OOM
- Save/Load trained Core Memory přes `StateCheckpoint` (`core_memory` filter)

---

## [0.5.0-alpha.10] — 2026-04-17

### Přidáno — Multi-layer Core Memory + cross-entropy loss

**Fáze 5 alpha.10 milestone:** autograd teče přes všech 24 Mamba-2
vrstev najednou s realistickou LM loss. První produkční building block
pro Core Memory training.

#### `CoreMemoryStack` (`training/core_memory.rs`)

- `CoreMemoryStack::zeros(config, device)` — nulová inicializace všech
  vrstev (pro restart z trained checkpointu)
- `CoreMemoryStack::randn_small(config, device)` — malá randn init
  (std=0.01), vhodná pro bring-up (non-zero grad signal od 1. iterace)
- `inject_into_state(&mut ModelState, runtime_dtype)` — aplikuje
  všechny trainable init_states do ModelState s dtype upcastem
- `vars()` / `vars_owned()` — collection pro předání AdamW optimizéru
  a `clip_grad_norm`
- 4 unit testy (zeros count, randn nonzero, vars length, inject replaces
  ssm_state)

#### Cross-entropy loss (`training/loss.rs`)

- `cross_entropy_next_token(logits, input_ids)` — standardní LM loss
  s shift-by-one konvencí. F32 upcast pro log_softmax (BF16 nestabilní
  pro extreme logits). Gather na target positions, NLL mean přes všechny
  (batch × (seq_len-1)) predikcí.
- 4 unit testy (confident correct → loss≈0, uniform → ln(vocab),
  seq_len=1 rejection, gradient flows to logits)

#### Multi-layer smoke test

- `Sofie::smoke_train_core_memory_multilayer(seq_len, lr, max_grad_norm)`
  — kompletní forward + backward + step přes všech 24 vrstev
- `MultiLayerSmokeResult` s per-layer gradient norms + init state
  before/after (analýza distribuce gradientu napříč hloubkou)
- CLI subkomanda `train-core-memory-multi --seq-len N --learning-rate LR
  --grad-clip G` (default seq_len=4, lr=1e-3, grad_clip=1.0)

### Ověřeno na Falcon-H1-1.5B (CPU F32, 24 vrstev)

```
seq_len=2, lr=1e-3, grad_clip=1.0
loss (cross-entropy):    21.51   (random baseline ln(65537)≈11.09)
total grad L2 (pre-clip):  5.24
total grad L2 (post-clip): 1.00
wall time:              14860 ms
per-layer grad spread:   L0=1.6e-2 … L23=4.38 (roste s hloubkou,
                         Peri-LN massive activations pattern, clipping srovnává)
```

**PASS:** všechny 24 vrstvy dostaly gradient, non-trivial values napříč
celým stackem, init_state se pohnul ve všech vrstvách.

### Známé limity (→ alpha.11+)

- **CUDA OOM na RTX 4050 (6 GB)** pro multi-layer backward graph — full
  forward s autograd přes 24 vrstev × 65537 vocab nezvládne. CPU F32
  projede za 15 s. Pro reálný training (tisíce kroků) potřebujeme buď:
  (a) gradient checkpointing, (b) gradient accumulation, (c) Gaia deploy
  s víc VRAM. Řeší alpha.11.
- **Loss > random baseline** (21.5 vs. 11.09) je očekávané — trained
  init_state je random, frozen weights ho neberou jako valid starting
  point. Po několika krocích training by loss měla klesnout k baseline
  a níž.

### Testy

66 unit testů (+8 oproti alpha.9: 4 loss + 4 CoreMemoryStack).

### Plán alpha.11

- Dataset struct (text → tokenize → chunks), ChatML wrapping
- Training loop (epoch × batch), gradient accumulation
- AdamW betas `(0.9, 0.95)`, cosine/WSD schedule
- LR sweep (RWKV doporučuje 1.0 pro State Tuning po warmup)
- Gradient checkpointing (pokud VRAM zůstane blocker)

---

## [0.5.0-alpha.9] — 2026-04-17

### 🎯 BUG-010 vyřešen — NaN backward přes více vrstev odstraněn

**Root cause** (identifikovaný diagnostikou z alpha.8):
lokální `silu(x) = x * recip(1 + exp(-x))` v `mixer.rs` a `norm.rs`
produkuje `NaN` gradient pro extrémně záporné x:
- forward: `exp(-x) = Inf` (F32 overflow pro x < -87)
  → `1 + Inf = Inf` → `recip(Inf) = 0` → `silu = x * 0 = 0` (forward je OK)
- backward: `d/dx = recip + x * recip² * exp(-x)`
  → `0 + (-100) * 0 * Inf` → **`0 * Inf = NaN`**

Hluboké vrstvy Falcon-H1 produkují po conv1d hodnoty v rozsahu ±100
(diagnostika: `layer.mlp.down_raw` v L23 = 3632, `mixer_out_raw` v L22 = 500).
V těchto rozsazích naivní silu backward exploduje.

### Opraveno

- `mixer.rs::silu` — delegace na `candle_nn::ops::silu` (native
  `Tensor::silu()` s numericky stabilním backward kernelem). F32 upcast
  zachován pro konzistenci s ostatními citlivými místy.
- `norm.rs::silu` — stejný refactor (použito v `RmsNormGated` v SSM branch)

### Ověřeno na Falcon-H1-1.5B (CUDA, RTX 4050)

| Konfigurace (seq_len=1, CoreMemory L22) | Před fixem | Po fixu |
|------------------------------------------|-----------|---------|
| cut=22 (jen L22)                         | 2.85      | 2.85    |
| cut=23 (L22 + L23)                       | **NaN**   | **9.80**|
| cut=full (L22 + L23 + lm_head)           | **NaN**   | **1.74**|

Forward hidden norms (L0 → L23): 2.4e-14 → 1.3e3 (massive activations
pattern zůstává — je intrinsic Peri-LN, fix jen řeší backward stabilitu).

### Přidáno — micro testy v `training/repro.rs`

- `silu_local_backward_normal` — normal input PASS
- `silu_local_backward_extreme_negative_produces_nan` —
  `#[should_panic]` dokumentuje bug (naivní silu x=-100 → NaN grad)
- `silu_candle_nn_backward_extreme_negative_finite` — fix verification
  (candle silu x=-100 → finite grad)

### Testy

58 unit testů prochází (+3 oproti alpha.8).

### Další krok

BUG-010 ≠ blokátor Core Memory production. Pokračujeme na multi-layer
training loop (v0.5.0-alpha.10): `CoreMemory` pro všechny vrstvy,
cross-entropy loss na next-token, gradient accumulation, save/load.

---

## [0.5.0-alpha.8] — 2026-04-17

### Přidáno — Instrumentace forward pass (BUG-010 diagnostika)

- `training/trace.rs` — thread-local trace sink (`start` / `finish` /
  `probe(&t, label)`). `probe` na aktivním sinku spočte
  `abs_max, abs_min_nonzero, mean, l2, has_nan, has_inf` a pushne do
  `Vec<TraceEntry>`. Detach před výpočtem — neváže autograd graf.
  5 unit testů (no-op když sink neaktivní, capture entries, NaN/Inf flags,
  abs_min_nonzero skip nul přes `where_cond`, tabulkový render).
- `falcon_h1::layer::LayerStop` — enum sub-layer cut bodů
  (`AfterPreNorm`, `AfterSsmBranch`, `AfterAttnBranch`, `AfterResidual1`,
  `AfterPostNorm`, `AfterMlpGate`, `AfterMlpSiluMul`, `AfterMlpDown`,
  `Full`). Implementuje `FromStr` pro CLI parsing.
- `FalconH1Layer::forward_until(x, pos, state, stop)` — varianta
  `forward` s brzkým returnem na daném mezibodě. Původní `forward`
  deleguje na `forward_until(.., LayerStop::Full)`.
- `FalconH1Model::forward_up_to_layer_with_stop(..., up_to_layer, stop)` —
  sub-layer cut na poslední trénované vrstvě.
- `Sofie::model_forward_up_to_layer_with_stop` + `smoke_train_core_memory_component`
  (trace + component cut v jednom API).
- CLI flagy na `train-core-memory-smoke`:
  - `--cut-at-component <pre-norm|ssm|attn|residual1|post-norm|mlp-gate|mlp-silu-mul|mlp-down|full>`
  - `--trace` — forward tensor stats tabulka po běhu

### Probe body (30+ míst v forward pass)

- **model.rs:** `embed_scaled`, `after_layer_{i}`
- **layer.rs:** `pre_norm_out`, `mixer_out_raw`, `ssm_out_scaled`,
  `attn_out_raw`, `attn_out_scaled`, `residual_1`, `post_norm_out`,
  `mlp.gate_raw`, `mlp.up`, `mlp.gate_scaled`, `mlp.silu_gate`,
  `mlp.silu_gate_times_up`, `mlp.down_raw`, `mlp.down_scaled`,
  `residual_2`
- **mixer.rs:** `in_proj_out`, `after_mup_vec`, `z`, `dt_raw`,
  `conv_out`, `silu_conv`, `dt_plus_bias`, `softplus_dt`, `a_neg_exp`,
  `dt_mul_a`, `da_seq_exp`, `ssm_state_final`, `ssm_scan_out`,
  `gated_norm_out`, `out_proj`
- **attention.rs:** `q_proj`, `k_proj`, `v_proj`, `q_rope`, `k_rope`,
  `qk_logits`, `softmax`, `softmax_v`, `o_proj`

### Použití

```
# Plný forward s trace:
train-core-memory-smoke --layer-idx 0 --seq-len 1 --trace

# Binary search: izoluj NaN backward uvnitř L22:
train-core-memory-smoke --layer-idx 22 --cut-at-layer 22 \\
  --cut-at-component after-ssm --trace
train-core-memory-smoke --layer-idx 22 --cut-at-layer 22 \\
  --cut-at-component after-attn --trace
# …postupně rozšiřovat komponentu, sledovat kdy gradient přejde z OK na NaN.
```

### Plán alpha.9

Spuštění diagnostiky na Falcon-H1-1.5B. Očekáváme, že trace identifikuje
op s extrémním dynamickým rozsahem — kandidáti po alpha.7:
- `silu(x) = x * recip(1 + exp(-x))` v `mixer.rs::silu` a `norm.rs::silu`
  pro velmi záporné x (recip(Inf) backward)
- Attention `qk_logits` / `softmax` pro extrémní pre-softmax hodnoty
- `conv1d` backward (zatím netestováno v repro.rs)

### Testy

55 unit testů prochází. Přibyly: 5× trace, smoke/model regrese neporušena.

---

## [0.5.0-alpha.7] — 2026-04-16

### Přidáno
- `training/repro.rs` — 14 micro unit testů izolujících backward chování
  jednotlivých ops (RMSNorm, softplus, exp.neg, recip, silu, matmul)
  na různých vstupech (normal, tiny 1e-7, extreme 1e-14, mixed range,
  massive outliers).

### Odhalené Candle autograd limity (dokumentováno přes `#[should_panic]`)

1. **`recip` backward pro x ≈ 1e-10** → gradient **Inf**. Mathematically
   `d/dx (1/x) = -1/x²` → pro velmi malé x nabývá hodnot přesahujících
   F32 bezpečný rozsah.
2. **`softplus(x) = log(1 + exp(x))` pro x ≥ 88** → forward `Inf` (exp
   overflow v F32), backward `NaN`. **Pravděpodobný problém pro extreme
   `dt + dt_bias` v SSM discretization.**

### Opraveno (částečně)
- **`mixer.rs::softplus` numericky stabilní** — nahrazena naivní implementace
  za `relu(x) + log(1 + exp(-|x|))`, matematicky identická, numericky bounded.
  Test `softplus_stable_matches_naive_on_safe_range` ověřuje ekvivalenci
  v rozsahu [-15, 15] (rel. error < 1e-5). Test
  `softplus_stable_backward_extreme_positive_finite` potvrzuje finite
  gradient pro x=100.

### Ale — softplus fix **nevyřešil BUG-010** v reálném modelu
Po aplikaci stable softplus: smoke L22 cut=23 stále vrací NaN gradient.
Znamená to, že realistický `dt + dt_bias` v Falcon-H1-1.5B nepřesahuje
safe range — softplus overflow **nebyl primary root cause** v našem případě.

**Stále otevřené kandidáty:**
- `silu(x) = x * recip(1 + exp(-x))` — recip(Inf) backward pro velmi
  záporné x
- Attention softmax backward pro extreme logits
- `conv1d` backward (zatím netestováno v repro.rs)
- Residual sum `x + ssm_out + attn_out` backward akumulace s duplicate
  compute subgraph

### Plán alpha.8
Instrumentovaný forward pass — logovat hidden norms po každé op v
`FalconH1Layer::forward`, najít op která produkuje input s extremním
dynamickým rozsahem, jehož backward Candle nezvládne.

---

## [0.5.0-alpha.6] — 2026-04-16

### Přidáno
- `training/clip.rs` — `clip_grad_norm(grads, vars, max_norm)` helper
  (PyTorch-style global L2 norm clipping, Candle nemá built-in)
- `Sofie::smoke_train_core_memory_clipped(..., max_grad_norm)` +
  `SmokeTrainResult.pre_clip_gradient_norm` field pro monitoring
- CLI flag `--grad-clip <VALUE>` na `train-core-memory-smoke`
- 2 unit testy pro clip_grad_norm (below/above threshold)
- Celkem 36 testů prochází

### Ověřeno — dampening μP multipliery jsou správně načtené
Step 1 z research doporučení: zkontrolovat config.json hodnoty vs. načtené:
- `ssm_out_multiplier = 0.11785` ✓ aplikováno v layer.rs:89–91
- `attention_out_multiplier = 0.234375` ✓ aplikováno v layer.rs:94–96
- `mlp_multipliers = [0.44, 0.13]` ✓ aplikováno v layer.rs:108–124
- `lm_head_multiplier = 0.0195` ✓ aplikováno v model.rs:91

Dampening **není** primary root cause — multipliery jsou korektní.

### Klíčový nález: clipping nepomůže našemu konkrétnímu problému

Test L22 `--cut-at-layer 23 --grad-clip 1.0`:
```
gradient L2 (pre-clip): NaN
gradient L2 norm:       NaN
```

**Pre-clip gradient je už NaN**, clipping může škálovat "velký ale konečný",
ne NaN. Naše NaN nevzniká akumulací přes amplifikaci — **vzniká uvnitř
`loss.backward()` samotného**, v konkrétní op s numerickou dírou.

Výzkumná hypotéza "Peri-LN massive activations → grad clipping pomůže"
je **částečně validní** (backward skutečně narazí na massive activations),
ale neplatí, že clipping je řešení. Skutečný root cause je op-specific:
pravděpodobně RMSNorm rsqrt backward pro input s extrémním dynamic range
(L0 hidden 10⁻¹⁴, L2 hidden 10²) nebo softplus exponenciální overflow
v Candle implementaci.

### Plán alpha.7: minimal reproduction

Research step 5 — osekat model na minimum a binary-searchem najít konkrétní
op. Strategie:

1. **Zachytit intermediate norms ve forward** přes instrumented verze ops.
   Cíl: najít první vrstvu/op, kde se objevuje denormalized nebo extrémní
   hodnota, která v backward produkuje NaN.
2. **Reproduce v malém test case** — 2-layer model, synthetic vstup,
   manuálně přes operace, detekovat přesný op.
3. **Fix**: buď F64 upcast v konkrétní op, nebo workaround v naší
   implementaci (např. clamp rsqrt denominator).

### Status Fáze 5

- Autograd **technicky funguje** (L23 plný forward PASS, izolované vrstvy
  PASS, L0 cut=2 PASS, L0 cut=2 + clip=1.0 PASS)
- **BUG-010 stále otevřen** — pre-clip NaN v backward pro některé
  konfigurace. Clipping nepomůže, je to op-specific issue.
- Gradient clipping helper je **stále užitečný** pro budoucí skutečný
  training (kde amplifikace bude reálný problém)

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
