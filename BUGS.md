# Bugs — Eleutheria

Aktivní bugy k opravě.

Formát: `BUG-NNN` s verzí nálezu, závažností (P1–P4), reprodukcí a stavem.

---

## Aktivní

### BUG-010 — Inter-layer backward amplifikuje gradient do NaN
- **Nalezeno:** v0.5.0-alpha.4 | **Závažnost:** P2 (blokuje Core Memory training
  přes více vrstev, ale alpha.1 cíl "autograd teče" splněn pro L23)
- **Reprodukce:** `train-core-memory-smoke --seq-len 1 --layer-idx 22`
  (CPU, 1.5B) → NaN. Izolovaně `--cut-at-layer 22` → PASS gradient 2.84.
  Přidání 1 vrstvy `--cut-at-layer 23` → NaN.
- **Pattern:** intra-layer SSM backward stabilní; inter-layer forward
  Jacobian sporadicky exploduje. L20–L22 exploze po 1 přidané vrstvě,
  L0 prochází přes 2 ale selže na 3. Není rovnoměrné.
- **Hypotézy:** RMSNorm rsqrt backward pro bohatší aktivace pozdějších
  vrstev / softplus numerická díra / konstruktivní interference gradientu
  v paralelním hybridu
- **Workaround pro alpha/experimenty:** trénovat jen L23 (`--layer-idx 23`
  bez cut) nebo používat `--cut-at-layer` s úzkým rozsahem
- **Update v0.5.0-alpha.6:** Gradient clipping (`--grad-clip 1.0`)
  **nepomohl** — pre-clip gradient je už NaN. Research hypotéza
  "amplifikace přes vrstvy" není úplná; skutečný root cause je
  **op-specific NaN uvnitř `loss.backward()` samotného**. Dampening μP
  multipliery ověřeny jako správně načtené (ne primary root cause).
- **Update v0.5.0-alpha.7:** Minimal reproduction `training/repro.rs`
  (14 micro testů). Stable softplus implementován (`relu(x) + log1p(exp(-|x|))`).
  Fix softplus **nevyřešil BUG-010** — realistický `dt + dt_bias` nepřesahuje
  safe range. Dokumentovány další Candle autograd limity přes `#[should_panic]`
  (recip near-zero → Inf, softplus naivní x=100 → NaN).
- **Update v0.5.0-alpha.8:** Přidána infrastruktura pro diagnostiku:
  (1) thread-local forward trace sink s 30+ probe body
  (`training/trace.rs`), (2) sub-layer cut-at-component (`LayerStop` enum
  v `falcon_h1::layer`, `forward_until`, `forward_up_to_layer_with_stop`),
  (3) CLI `--trace` a `--cut-at-component`. Instrumentace sama
  problém neřeší, ale umožní alpha.9 identifikaci konkrétní op.
- **Plánované řešení:** alpha.9 — spuštění diagnostiky, fix konkrétního op
  (primární kandidáti po alpha.7/.8: local `silu` v mixer.rs/norm.rs pro
  velmi záporné x (recip(Inf) backward), attention softmax pre-max-subtract,
  conv1d backward)

---

## Vyřešené

---

## Vyřešené

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
