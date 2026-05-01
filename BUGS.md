# Bugs — Eleutheria

Aktivní bugy k opravě.

Formát: `BUG-NNN` s verzí nálezu, závažností (P1–P4), reprodukcí a stavem.

---

## Aktivní

*(Žádné aktivní bugy.)*

---

## Vyřešené

### BUG-011 — CUDA gather vyžaduje contiguous v cross_entropy_next_token ✓
- **Nalezeno:** v0.5.0-alpha.20 (první cloud GPU production run)
- **Závažnost:** P1 (training crash na CUDA s batch>1)
- **Reprodukce:** `train-core-memory --batch-size 16 --seq-len 16 --cuda`
  na A100. Lokální RTX 4050 + CPU smoke testy (batch=1, seq=4 / batch=1
  seq=16) nikdy neselhaly — shape pattern asi vyhnul.
- **Chyba:** `cross_entropy: gather only supports contiguous tensors`
- **Příčina:** `cross_entropy_next_token` v `training/loss.rs` používá
  `narrow` + `unsqueeze` chain pro přípravu `targets_idx`, který
  produkuje non-contiguous view tensor. CPU `gather` je tolerantní
  k non-contiguous, **CUDA gather je striktní**.
- **Řešení:** explicit `.contiguous()` na `log_probs` (po `log_softmax`)
  a `targets_idx` (po unsqueeze). Drobný overhead na CPU (no-op pokud
  už contiguous), bezpečný na CUDA.
- **Opraveno:** v0.5.0-alpha.20 (commit `d8871fe`)
- **Ref:** RN-013 v RESEARCH-NOTES.md

### BUG-010 — Inter-layer backward amplifikuje gradient do NaN ✓
- **Nalezeno:** v0.5.0-alpha.4 | **Závažnost:** P2 (blokoval Core Memory
  training přes více vrstev)
- **Reprodukce:** `train-core-memory-smoke --seq-len 1 --layer-idx 22
  --cut-at-layer 23` → NaN. Izolovaně cut=22 → PASS gradient 2.84.
- **Root cause (identifikováno alpha.8 diagnostikou):** lokální
  `silu(x) = x * recip(1 + exp(-x))` v `mixer.rs` a `norm.rs`.
  Pro extrémně záporné x: forward `exp(-x) = Inf` → `1+Inf = Inf`
  → `recip(Inf) = 0` → `silu = 0` (forward OK); backward
  `recip + x * recip² * exp(-x)` = `0 + x * 0 * Inf` → **`0 * Inf = NaN`**.
  Hluboké Falcon-H1 vrstvy produkují po conv1d hodnoty ±100, kde
  tato naivní implementace exploduje.
- **Řešení (v0.5.0-alpha.9):** delegace `silu` v `mixer.rs` a `norm.rs`
  na `candle_nn::ops::silu` (native `Tensor::silu()` s numericky
  stabilním backward kernelem). F32 upcast zachován.
- **Ověřeno:** L22 cut=22 grad 2.85 (beze změny), cut=23 NaN→**9.80**,
  cut=full NaN→**1.74**. Sweep všech cut bodů PASS.
- **Opraveno:** v0.5.0-alpha.9

### Historie diagnostiky (alpha.4–alpha.8)
- **alpha.4:** `forward_up_to_layer` + binary search cut-at-layer
  — lokalizace na poslední vrstvu ve stacku.
- **alpha.5:** diagnostic sweep, forward hidden norms per-layer.
- **alpha.6:** gradient clipping ověřeno jako neefektivní (pre-clip grad
  už NaN). Dampening μP multipliery ověřeny jako správně načtené.
- **alpha.7:** minimal reproduction (`training/repro.rs`), stable softplus.
  Fix softplus **nevyřešil** BUG-010 — realistický `dt + dt_bias`
  safe range nepřekračuje. Dokumentovány Candle autograd limity
  přes `#[should_panic]` (recip near-zero → Inf, naivní softplus
  x=100 → NaN).
- **alpha.8:** instrumentace — thread-local forward trace sink
  (`training/trace.rs`, 30+ probe bodů) + sub-layer cut-at-component
  (`LayerStop` enum, 9 pozic uvnitř vrstvy). Diagnostika jednoznačně
  potvrdila: **SSM branch backward selhává, attention OK**
  (`after-ssm`=NaN, `after-attn`=0.60, `after-pre-norm`=2.08).

### BUG-009 — Panic v streaming diff při BPE retokenizaci ✓
- **Nalezeno:** v0.4.2 | **Závažnost:** P1
- **Reprodukce:** `bench-retention --variant all` na SsmOnly — model bez KV
  cache halucinuje česky (UTF-8 multi-byte), diff streaming panikuje
  `byte index N is out of bounds`
- **Příčina:** `&full_text[emitted_len..]` v `generate_from_logits` je
  byte-level slice. BPE tokenizer při novém tokenu může re-dekódovat celý
  generated vektor jinak — `full_text` z iterace N+1 nemusí být
  byte-prefix extension z iterace N. `emitted_len` pak ukazuje mimo
  řetězec nebo doprostřed UTF-8 sekvence.
- **Řešení:** Resync na nejbližší nižší UTF-8 char boundary pomocí
  `str::is_char_boundary`. Final decode je z úplného `generated` vektoru,
  takže žádný data loss; streaming může v ojedinělých případech krátce
  opakovat znaky nebo přeskočit 1–2 znaky.
- **Opraveno:** v0.4.3

### BUG-008 — BF16 temperature sampling crash ✓
- **Nalezeno:** v0.2.0 | **Závažnost:** P1
- **Reprodukce:** Jakýkoliv generate s temperature > 0 na CUDA
- **Příčina:** BF16 tensor ÷ f64 scalar není podporováno v Candle
- **Řešení:** Cast logits na F32 před dělením teplotou
- **Opraveno:** v0.2.0

### BUG-007 — RmsNormGated invertovaná logika ✓
- **Nalezeno:** v0.1.0 | **Závažnost:** P1
- **Reprodukce:** Model generuje garbage po ~5 tokenech
- **Příčina:** Gate/norm pořadí obrácené pro `norm_before_gate=false` — chyba se akumuluje přes 24/44 vrstev
- **Řešení:** Swap větví dle HF referenční implementace
- **Opraveno:** v0.1.0

### BUG-006 — Parallel prefill garbage output ✓
- **Nalezeno:** v0.1.0 | **Závažnost:** P1
- **Příčina:** Důsledek BUG-007
- **Opraveno:** v0.1.0

### BUG-005 — Špatný final norm key ✓
- **Nalezeno:** v0.1.0 | **Závažnost:** P1
- **Příčina:** `model.norm` místo `model.final_layernorm`
- **Opraveno:** v0.1.0

### BUG-004 — mlp_multipliers aplikovány na gate i up ✓
- **Nalezeno:** v0.1.0 | **Závažnost:** P2
- **Příčina:** `mlp_multipliers[0]` použit pro obě projekce místo jen gate
- **Opraveno:** v0.1.0

### BUG-003 — Chybějící mup_vector na in_proj segmentech ✓
- **Nalezeno:** v0.1.0 | **Závažnost:** P1
- **Příčina:** muP vektor nebyl aplikován na [z/x/B/C/dt] segmenty
- **Opraveno:** v0.1.0

### BUG-002 — ssm_in_multiplier na špatném místě ✓
- **Nalezeno:** v0.1.0 | **Závažnost:** P2
- **Příčina:** Aplikováno PŘED in_proj místo PO
- **Opraveno:** v0.1.0

### BUG-001 — Token duplikace v conv1d_step ✓
- **Nalezeno:** v0.1.0 | **Závažnost:** P2
- **Příčina:** State cat zahrnul token dvakrát
- **Řešení:** Roll-left + append pattern
- **Opraveno:** v0.1.0
