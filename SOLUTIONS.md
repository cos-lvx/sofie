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
