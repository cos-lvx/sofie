# Solutions — Eleutheria

Znalostní báze vyřešených problémů pro budoucí referenci.

Formát: `SOL-NNN` s problémem, kořenovou příčinou, řešením a ponaučením.

---

## SOL-001 — F32 upcast pro numericky citlivé operace

- **Problém:** BF16 má jen 7 mantissa bitů → akumulace chyb v normalizaci a aktivacích
- **Řešení:** F32 dočasný výpočet pro: RmsNorm, SiLU, softplus, RoPE, temperature sampling
- **Ponaučení:** BF16 je skvělé pro storage a matmul, ale citlivé operace vždy F32

## SOL-002 — muP multiplikátory jako f64 konstanty

- **Problém:** Falcon-H1 vyžaduje Maximal Update Parametrization pro správný výstup
- **Řešení:** Konstanty aplikované přes `affine()`: embedding 5.66, lm_head 0.0195, atd.
- **Ponaučení:** muP je kritický — bez něj model generuje nesmysly

## SOL-003 — conv1d_step roll-left pattern

- **Problém:** Token duplikace při naivním přidávání do conv state
- **Řešení:** Roll state vlevo, zapiš nový token, konvoluce přes celý state (HF reference)
- **Ponaučení:** Vždy ověř state management proti referenční implementaci

## SOL-004 — Weight key audit před implementací

- **Problém:** Safetensors key names se liší mezi modely (např. `model.final_layernorm` vs `model.norm`)
- **Řešení:** Inspekce safetensors souboru před psaním weight loading kódu
- **Ponaučení:** 5 minut inspekce ušetří hodiny debugování

## SOL-005 — Segmentace in_proj výstupu pro Mamba-2

- **Problém:** in_proj output se dělí na [z/x/B/C/dt] segmenty s per-segment muP
- **Řešení:** Explicitní split + mup_vector [0.354, 0.25, 0.177, 0.5, 0.354]
- **Ponaučení:** Hybrid modely mají jemnou granularitu škálování — čti paper pečlivě

## SOL-006 — Debugging garbage output layer-by-layer

- **Problém:** Model generoval nesmysly po parallel prefill
- **Řešení:** Systematické porovnání výstupů vrstva po vrstvě s HF referencí
- **Ponaučení:** Chyba v normalizaci je multiplikativní — akumuluje se přes vrstvy.
  Vždy kontroluj normy jako první

## SOL-007 — Safetensors metadata přes přímou závislost

- **Problém:** `candle_core::safetensors::save()` hardcoduje `None` pro metadata
  hlavičku — nelze uložit `__metadata__` do safetensors souboru
- **Řešení:** Přidat `safetensors = "0.6"` jako explicitní závislost (už je transitivní
  přes candle-core, nula nových bytů) a volat `safetensors::tensor::serialize_to_file()`
  přímo s `Some(metadata_map)`. Candle `Tensor` implementuje `safetensors::View`,
  takže `&HashMap<String, Tensor>` funguje jako `data` argument
- **Ponaučení:** Candle wrappery jsou pohodlné, ale občas zakrývají features
  underlying crate. Vždy zkontroluj zdroj wrapperu, ne jen jeho API

## SOL-008 — CUDA 13.2 na Arch Linux (cudarc workaround)

- **Problém:** `cudarc` 0.18.2 (závislost Candle) podporuje max CUDA 13.1.
  Arch Linux rolling release má CUDA 13.2 — build script panikuje:
  "Unsupported cuda toolkit version: `13.2`"
- **Řešení:** Env variable `CUDARC_CUDA_VERSION=13010` přeskočí detekci přes
  `nvcc --version` a použije 13.1 bindings. CUDA 13.2 je zpětně kompatibilní.
  Nastaveno permanentně v `.cargo/config.toml`
- **Ponaučení:** Na Arch Linux s bleeding edge balíčky vždy kontroluj, zda
  Rust build skripty podporují aktuální verze systémových knihoven

## SOL-009 — Stabilní aktivační fn přes native Candle Tensor methods

- **Problém:** Naivní `silu(x) = x * recip(1 + exp(-x))` v Rust produkuje
  `NaN` gradient pro extrémní |x| (BUG-010). Pro `x < -87`: forward
  `exp(-x) = Inf → recip(Inf) = 0 → silu = 0` (OK), ale backward
  obsahuje chain `x * recip² * exp(-x)` = `0 * Inf = NaN`. Hluboké
  vrstvy Falcon-H1 produkují po conv1d hodnoty ±100, kde tahle
  implementace exploduje.
- **Řešení:** Delegovat na `candle_nn::ops::silu` (volá native
  `Tensor::silu()` s vlastním backward kernelem napsaným explicitně
  pro numerickou stabilitu). F32 upcast zachován pro konzistenci
  s ostatními citlivými místy (RmsNorm, softplus).
- **Ponaučení:** Candle má native implementace aktivačních funkcí
  s ručně napsanými backward kernely. Vlastní rekonstrukce přes
  naivní math (`exp`, `recip`, `affine`) skládá ops, jejichž samostatně
  validní backward může v kombinaci produkovat `0 * Inf = NaN`. **Před
  psaním vlastní aktivace grep `candle_nn::ops::` a `Tensor::`** —
  pravděpodobně už existuje.

## SOL-010 — Thread-local trace sink pro forward pass diagnostiku

- **Problém:** Backward v Candle je black-box — nemáme hooky "před/po
  každé op". Když `loss.backward()` produkuje NaN, nevíme, která
  konkrétní op selhala. BUG-010 diagnostika vyžadovala lokalizovat
  problém mezi stovkami tensorů v forward passu.
- **Řešení:** `training/trace.rs` — thread-local `RefCell<Option<Vec<TraceEntry>>>`.
  `trace::start()` aktivuje sběr, `trace::probe(&t, label)` zapíše stats
  (abs_max, abs_min_nonzero, mean, l2, NaN/Inf flags), `trace::finish()`
  vrátí záznamy. Rozptýleno ~30 probe bodů v `layer`, `mixer`,
  `attention`, `norm`. Probe dělá `t.detach()` před výpočtem — neváže
  autograd graph, nezasahuje do tréninku.
- **Ponaučení:** Forward instrumentace nevyřeší backward bug přímo,
  ale ukáže tensory s extrémním dynamickým rozsahem (denormal, Inf).
  To jsou prvními kandidáty pro numerickou explozi v backward.
  Thread-local + detach pattern je čistý, reusable — podobný přístup
  platí pro jakékoli ML framework, kde chceme non-invasive forward
  instrumentaci.

## SOL-011 — Sub-layer cut binary search pro lokalizaci backward bugs

- **Problém:** Když v modelu s N vrstvami backward produkuje NaN,
  hrubá diagnostika `--cut-at-layer K` řekne "problém je mezi 0 a K".
  Pro přesnou lokalizaci uvnitř jedné vrstvy (SSM branch? attention?
  MLP? residual?) potřebujeme jemnější granularitu.
- **Řešení:** `LayerStop` enum v `falcon_h1::layer` s 9 cut body
  (pre_norm, ssm, attn, residual_1, post_norm, mlp_gate, mlp_silu_mul,
  mlp_down, full). `FalconH1Layer::forward_until(x, pos, state, stop)`
  s early return na daném bodu. `FalconH1Model::forward_up_to_layer_with_stop`
  aplikuje cut na poslední vrstvě stacku. CLI `--cut-at-component` +
  unified API `smoke_train_core_memory_component`.
- **Ponaučení:** Binary search v prostoru "kde backward padá" je
  výkonný debug nástroj, pokud máme dostatečně jemnou granularitu.
  Pro BUG-010: `after-ssm=NaN` + `after-attn=PASS` + `after-pre-norm=PASS`
  jednoznačně identifikovalo SSM branch jako viníka. Pattern aplikovatelný
  na jakýkoli hierarchický forward (N vrstev × M komponent per vrstva):
  **přidej early-return mezibodě, měř backward per mezibod**.

## SOL-012 — Per-layer gradient checkpointing s synthetic loss trickem

- **Problém:** Multi-layer Core Memory training drží plný 24-layer
  autograd graf v paměti. Na CPU 1.5B F32 to je 48 s/step (KI-006 — memory
  traffic dominuje). Na CUDA 6 GB OOM ihned při prvním backward kroku
  (KI-005). Candle nemá `torch.utils.checkpoint` ekvivalent.
- **Řešení:** Custom 3-fázový checkpointing v `training/checkpoint.rs`.
  - **Phase 1 (forward sweep):** per-layer forward s `Tensor::detach()`
    na input/output, save state snapshotu před každou vrstvou. Žádný
    autograd graf nevzniká, jen arr `Vec<Tensor>` saved inputs.
  - **Phase 2 (final chunk):** re-forward `final_norm + lm_head` s
    autograd, cross_entropy loss, `loss.backward()`. Z `GradStore`
    vyjmem gradient na last hidden — to je vstupní target pro Phase 3.
  - **Phase 3 (reverse layer sweep):** pro vrstvu N-1 dolů na 0:
    restore stavu, fresh `Var::from_tensor(saved_input)` (leaf v re-comp
    grafu), re-forward s autograd ON, **synthetic loss = `sum(output *
    grad_target)`**, `synth.backward()` chain rule vrátí gradient pro
    `init_state[i]` + nový `grad_target` pro chunk i-1.
- **Synthetic loss trick:** Candle backward jde od skalárního loss.
  Pro propagaci libovolného tensor gradientu skrz chunk hranici
  konvertujem na skalár `synth = sum(out * grad_target)` — chain rule
  pak korektně propaguje. Pattern přenositelný na jakýkoli Candle-based
  framework.
- **Ponaučení:**
  1. **Detach + Var::from_tensor je standardní gradient checkpointing
     primitive v Candle** (chybí explicit API jako PyTorch). Pochopení
     `track_op` a `BackpropOp::none()` je klíč.
  2. **Per-layer chunking není vždy dost agresivní pro malé GPU.**
     Mamba scan + attention v jedné vrstvě 1.5B Falcon-H1 nesedí do
     ~2.4 GB volné VRAM po loadingu modelu. Pro RTX 4050 6 GB potřebujem
     sub-layer chunking nebo selective component-aware drop.
  3. **CPU benefits memory traffic víc než compute overhead bolí.**
     Re-forward během backward (2× compute) na CPU vyhrává nad held graf
     (memory bandwidth bottleneck). Naše alpha.12: 19 s/step vs. 48 s/step
     baseline — 2.5× **zrychlení** na CPU.
  4. **State snapshots jsou nutné** — Mamba scan modifikuje SSM state
     in-place, KV cache roste. `Tensor::copy()` (deep copy storage) per
     vrstva před forward, restore před re-forward.

## SOL-013 — Progressive drop saved tensorů v Rust gradient checkpointingu

- **Problém:** Sub-layer checkpointing (alpha.13) měl korektní algoritmus
  — Phase 1 (no-grad forward, save state snapshots + sub-chunk inputs),
  Phase 2 (loss.backward na lm_head), Phase 3 (per-vrstva re-forward
  sub-chunků α/β + synthetic loss backward) — ale na CUDA RTX 4050
  (6 GB VRAM, ~2.4 GB free po model loadingu) padal s
  `CUDA_ERROR_OUT_OF_MEMORY` *uprostřed* Phase 3 reverse sweep, kolem
  vrstvy 7 (z 24).
- **Diagnostika:** Probe přes `nvidia-smi` před každou fází (nový
  `ELEUTHERIA_CHECKPOINT_DEBUG=1` env flag). Memory roste lineárně
  ~64-96 MB per Phase 3 iterace. Phase 2 alone přidá +736 MB skok
  (lm_head + cross_entropy backward graph). Zjevné: něco nedropuje.
- **Kořenová příčina:**
  1. **`final_grads` z Phase 2 `loss.backward()` drží Arc references**
     na intermediate tensors lm_head workspace. Pokud po extract
     `grad_target` neuvolním explicitně, tyto references přetrvávají
     po celou Phase 3 (~700 MB neaktivní paměti).
  2. **Saved tensorů `Vec` (layer_inputs, layer_res1, state_snapshots)**
     drží Arc references na GPU storage po celou Phase 3 sweep, i když
     iterace i konzumuje jen jeden index. Pro 24 vrstev × ~70 MB =
     1.7 GB pinned memory.
  3. **Mamba scan workspace** v Candle alokuje per call ~50-100 MB,
     CUDA caching allocator drží v poolu. Pro chunk α v Phase 3 každá
     iterace alokuje, akumulátor roste.
- **Řešení:**
  1. **Scope-bounded loss.backward** — extrakce `grad_target` v
     samostatném scope, `drop(final_grads)` ihned po extract:
     ```rust
     let mut grad_target = {
         let final_grads = loss.backward()?;
         let g = final_grads.get(...)?.clone();
         drop(final_grads);
         g
     };
     drop(loss);
     drop(last_hidden_var);
     ```
  2. **Progressive `mem::replace`** v Phase 3 loop — po konzumaci
     `layer_inputs[i]`, `layer_res1[i]`, `state_snapshots[i]` v iteraci,
     replace dummy zero-tensor. Storage Arc se uvolní hned.
  3. **`phase3_layer_reverse` helper** drží sub-chunky β a α v
     lokálních scope. Synth losses, GradStores, intermediate activations
     dropnou na konci helper-callu.
  4. **`accum_seed` GradStore** — Candle `GradStore::new()` je private,
     takže reusujeme čisté store z první Phase 3 iterace (po cleanup
     `x_in_var` entry) jako akumulátor pro init_state Vars.
- **Empirie:** GPU memory během Phase 3 nyní **konstantní 5647/126 MB
  free** napříč všemi 24 vrstvami (před fixem rostlo +64-96 MB per
  iterace). Multi-layer training 1.5B na RTX 4050 6 GB nyní stabilní
  10 s/step.
- **Ponaučení:**
  1. **Rust drop semantika neni dost agresivní pro GPU memory.** I když
     proměnná zmizí ze scope, Arc reference v jiné struktuře (Vec, GradStore,
     parent autograd graph) drží storage. Explicit `drop()` + `mem::replace`
     jsou nutné v paths, kde memory pressure je realnost.
  2. **Phase 2 backward leak je univerzální Candle pattern.**
     `loss.backward()` vrací `GradStore`, který drží intermediate refs
     pro debug. V production loopu **vždy scope-boundovat** a extrahovat
     jen potřebný gradient před `drop`.
  3. **Diagnostický nvidia-smi probe je nepostradatelný.** Bez něj OOM
     vypadá jako "hardware limit", s ním vidíš lineární růst per iterace
     a víš, že je to leak. Probe stojí 50 ms volání, debugging time
     ušetří hodiny.
  4. **GradStore::new() je private — reuse z existing backward call.**
     Pattern: vyrobíš první GradStore kdekoli (loss.backward, synth
     backward), pak `insert` další gradients. Public API neumožňuje
     vytvoření prázdného store od nuly.

## SOL-014 — Best snapshot tracker pro noisy training trajektorie

- **Problém:** `from_stack(&CoreMemoryStack)` ukládá aktuální Var hodnoty
  v okamžiku save volání = **final state**. Pro noisy training (Phase 1
  rapid descent → Phase 2 overshoot → Phase 3-4 oscilace, RN-002) je
  rozdíl mezi best (~1.0) a final (~3.7) **dramatický 4×**. Save final
  pak zahodí best bod trajektorie.
- **Příčina:** Eleutheria do alpha.17 měla single save call po training
  loop. Adam moments mají paměť (β1=0.9, β2=0.999), velocity buffer
  naskakuje na strong gradient v Phase 1 → overshoot v Phase 2 → noisy
  oscilace. Trained state v `Var.as_tensor()` v okamžiku save je v
  noisy fázi, ne na nejlepším bodě.
- **Řešení (alpha.18, KI-009):** `BestSnapshotTracker` v
  `training/best_snapshot.rs` — shadow CPU F32 buffer per Var,
  `update_if_better(loss, step, stack)` lazy copy GPU→CPU **jen pokud
  loss zlepší historický best**. Pro typickou noisy trajektorii s 5-10
  best update events za 156 stepů je overhead ~150-300 ms (PCIe transfer
  24 vrstev × ~3 MB).
- **API:** `CoreMemoryArtifact::from_snapshot(Vec<Tensor>, ...)` jako
  alternativa k `from_stack`. Caller volí: `tracker.has_snapshot()` →
  `from_snapshot`, jinak fallback `from_stack`. CLI `--save-best` flag
  default off (backwards-compatible s alpha.16/17).
- **Ponaučení:**
  1. **Save vs final state je orthogonal koncept.** Pokud trajektorie
     je noisy, save final je strukturně špatně — uložené tenzory mohou
     být mnohem horší než nejlepší dosažený bod. Tracker tuhle disonanci
     řeší deterministicky.
  2. **Lazy update is critical.** Naivní implementace by kopírovala
     state na CPU každý step (drahé PCIe transfer). Trigger jen na best
     loss improvement = ~5-10 transferů celkem.
  3. **F32 CPU storage matchuje native dtype Var** v `CoreMemoryStack`
     i v `CoreMemoryArtifact`. Tedy round-trip save → load → restore je
     bez konverze, byte-identický s training state.

## SOL-015 — Periodic best snapshot flush (cloud GPU insurance)

- **Problém:** `BestSnapshotTracker` (SOL-014) drží shadow buffer
  **pouze v RAM**. Save na disk proběhne až jednou na konci
  `train_core_memory`. Pokud cloud GPU instance crashne / dostane
  preempci / network outage před koncem, **best snapshot se ztratí
  s celým procesem**. Pro alpha.20 production training (Vast AI A100,
  3.5+ hodin compute, reálné peníze) je to cost-meaningful risk.
- **Příčina:** Alpha.18 KI-009 fix vyřešil "save the right state, not
  the final state" — ale neřešil "what if we don't reach the save call".
  To je defekt jiné vrstvy. Pro lokální dev (RTX 4050, žádné
  preempce, killable interaktivně) nebyl reálný; pro cloud production
  je kritický.
- **Řešení (alpha.20, KI-012):**
  1. `BestSnapshotTracker::flush_to_disk(path, config, ...)` — clone
     shadow Vec (Arc-cheap), `CoreMemoryArtifact::from_snapshot`,
     atomic save přes privátní `atomic_save_artifact`.
  2. **Atomic write** — zapíše do sourozeneckého `<dir>/.<name>.tmp`,
     pak `std::fs::rename(tmp, path)`. Rename na stejném FS je atomic
     na POSIX → cílová cesta drží buď předchozí verzi, nebo nově
     zapsanou, nikdy half-written. Při chybě rename smaže tmp.
  3. `TrainingConfig.flush_best: Option<BestFlushConfig>` opt-in,
     CLI `--save-best-every N` (default 10). V training loop po každém
     successful `update_if_better` → step % every_n_steps == 0 → flush.
  4. **Non-fatal** — chyba disku se loguje (`tracing::warn!`), training
     pokračuje. Příští periodic flush to zkusí znovu.
- **Ponaučení:**
  1. **Insurance je orthogonal feature** od správnosti algoritmu. Algo
     může být perfectly correct, ale pokud nedostane šanci save, výsledek
     je ztracený. Insurance vrstva je samostatný design problem.
  2. **Cargo development (lokální) ≠ production (cloud).** Risks které
     jsou nulové lokálně mohou být cost-meaningful v cloud. Designovat
     features s ohledem na production environment, ne jen "works on my
     machine".
  3. **Atomic file write je triviální Rust pattern** (write tmp + rename),
     ale **musí být explicit**. Naivní `safetensors::serialize_to_file`
     na cílovou cestu by při crash zanechal half-written soubor a
     overwritl prior verzi.
  4. **Periodic flush overhead je < 1 % budget.** 75 MB safetensors
     write per flush = ~200-500 ms. Pro 44 s/step training s flush
     every 5 stepů = 0.2-0.5 s každých 220 s = pod 0.5 % overhead.

## SOL-016 — CUDA gather vyžaduje contiguous tensor

- **Problém:** První alpha.20 cloud GPU run na A100 spadl při prvním
  training stepu s:
  ```
  cross_entropy: gather only supports contiguous tensors
  ```
  Lokální testy (RTX 4050 + CPU) **nikdy neselhaly**, i přes 100+
  smoke runs napříč alpha.10-19.
- **Příčina (BUG-011):** `cross_entropy_next_token` v
  `training/loss.rs` používá `narrow + unsqueeze` chain pro přípravu
  `targets_idx`, který produkuje **non-contiguous view tensor**.
  CPU `gather` implementace v Candle je tolerantní k non-contiguous
  inputs (vnitřní convert-on-demand), CUDA `gather` kernel je striktní
  — vyžaduje contiguous input. Lokální smoke runs s batch=1 seq_len=4
  asi měly shape kombinaci, kde view byla incidentally contiguous,
  takže bug nikdy netriggovaný.
- **Řešení:** Explicit `.contiguous()` na `log_probs` (po `log_softmax`)
  a `targets_idx` (po unsqueeze). Drobný overhead na CPU (no-op pokud
  už contiguous), bezpečný na CUDA. Existující 4 unit testy nadále
  procházejí (CPU tolerantní), nově prošel i full CUDA training s
  batch>1 seq_len>4.
- **Ponaučení:**
  1. **Lokální smoke testy nejsou náhrada cloud GPU validace.** Latentní
     bugy v shape-dependent CUDA kernelech mohou ležet nezachycené
     dlouho, pokud test rozsahy vyhnou trigger pattern. Production-scale
     test (batch>1, seq_len>4, větší korpus) musí být součástí dev
     cycle, ne post-hoc verifikace.
  2. **Pro shape-dependent tensor pipeline obecně:** explicit
     `.contiguous()` před `gather` / `scatter` / `index_select` je
     bezpečný default. Drobný overhead na CPU, žádný overhead pokud už
     contiguous, eliminuje třídu bugů.
  3. **Cloud-first dev workflow** by zachycen byl tento bug ve fázi
     alpha.16-17 (kde batch>1 mělo smysl). Lokální RTX 4050 6 GB byl
     constraintující na batch=1 seq=4 (KI-005), ale od alpha.20 je
     cloud GPU validace (Vast → Starfield) první-class testovací
     prostředí.

## SOL-017 — CUDA auto-detect: 3-vrstvá architektura v rámci Cargo limitů

- **Problém:** Hardcoded `CUDARC_CUDA_VERSION = "13010"` v
  `.cargo/config.toml` byl Arch CUDA 13.2 workaround (KI-004) — ale
  pro multi-environment workflow (Arch lokál, Vast variabilní CUDA
  12.x-13.x, Starfield CUDA 13.0) je hardcoded přístup blokující.
  Nový host = manual config edit.
- **Příčina (Cargo design limit):** `cargo:rustc-env=KEY=VALUE` z build
  scriptu ovlivní **jen aktuální crate**, ne dependency build scripts
  (cudarc-sys). Tedy build.rs v eleutheria-core nemůže přímo nastavit
  env var pro cudarc-sys, který ji potřebuje při svém vlastním buildu.
  "Auto-detect z build.rs" naivním způsobem **nelze**.
- **Řešení (alpha.20 KI-004 → vyřešena alpha.21):** 3-vrstvá strategie:
  1. **Workspace default** v `.cargo/config.toml`:
     ```toml
     [env]
     CUDARC_CUDA_VERSION = { value = "13010", force = false }
     ```
     `force = false` = "použij default pokud env var není nastavený
     jinak". Pokrývá Arch CUDA 13.2 (clamp na cudarc max, zpětně kompat).
  2. **`scripts/detect-cuda.sh`** — Bash helper. Detekuje host přes
     `nvcc --version` (autoritativní, preferred) → fallback `nvidia-smi`
     (driver-reported, méně přesný pro toolkit). Mapuje na cudarc-supported
     hodnotu (13.0 → 13000, 13.x≥1 → 13010 clamp, 12.x → 12000-12080).
     Módy: `--source` (export inline), `--report` (audit), `--export-command`
     (eval-friendly).
  3. **`crates/eleutheria-core/build.rs`** — validace, ne auto-set.
     Detekuje host CUDA, porovná s nastaveným `CUDARC_CUDA_VERSION`,
     emituje `cargo:warning` s konkrétním doporučením při divergenci.
     **Ne panic** — cudarc je zpětně kompat, warning informativní.
- **Ponaučení:**
  1. **Cargo build script API má specific limity.** Rustc env vars jsou
     scoped per-crate, nelze leak do dependencies. Pro env vars které
     dependency build scripts vyžadují, řešení musí být **before
     cargo invocation** (workspace config default + setup scripts), ne
     v build.rs.
  2. **`force = false` v `.cargo/config.toml`** je elegantní pattern
     pro "workspace default s opt-in override". User nemusí nic dělat
     pro standard environments (Arch); pro non-default (Starfield, Vast)
     `source scripts/detect-cuda.sh` před `cargo build`.
  3. **Validation > silent fix.** Build.rs validace s warning je lepší
     než silent auto-fix. User vidí, že je něco jinak, dostane konkrétní
     instrukci, může rozhodnout. Silent fix by maskoval konfigurační
     problém.
  4. **`vast_setup.sh` inline detekce zůstala beze změny.** Cloud
     setup scripty mají vlastní jednoúčelový kód pro provisioning
     čerstvé instance — ne refactor pro DRY, ale clear separation
     mezi "ad-hoc setup" a "dev workflow helper".
