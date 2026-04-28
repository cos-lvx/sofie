# Reasoning chain — Layer-by-layer diff testing proti reference

## Zdroj
`eleutheria/SOLUTIONS.md#SOL-006` — Falcon-H1 inference produkoval po
parallel prefill nesmyslné tokeny. Loss čísla na první pohled OK, výstup
ne.

## Kontext
Falcon-H1 v Rust + Candle. První kompletní inference run běžel — model
se natáhl, tokeny se začaly emitovat — ale výstup byl garbage. Náhodné
tokeny, žádný coherent text. Reference (HuggingFace transformers) na
stejném promptu vrací smysluplný text. Single-token prefill (jen jeden
token, pak generation) fungoval. Multi-token parallel prefill neproduktivní.
Mám N tensorů procházejících 24 vrstvami × 5 sub-komponent — tj. řádově
3000 hodnot, kde se může schovat numerická chyba. End-to-end metric
(perplexity, loss) říká "něco je špatně", ale ne *kde*.

## Analytický flow

1. **Pozoruju, co je rozdílné mezi single-token a multi-token prefill.**
   Single-token = sekvenční scan. Multi-token = parallel scan. Chyba
   se projeví právě tam, kde se logika liší. To je první lokalizace
   — bug je někde v parallel scan code path, ne v shared single-token
   path.

2. **Druhá lokalizace: která komponenta v parallel path?** Conv1d (pro
   Mamba) ano nebo ne? RMSNorm? RoPE? SSM scan? Mám hypotézy, ale
   bez dat jsou jen hádky. Potřebuju **diff testing proti referenci**:
   pro stejný vstup spustím obě implementace (Rust + HF Python),
   porovnám výstupy *po každém kroku*.

3. **Setup diff harness.** HF Python script, který:
   - Načte model, tokenizes prompt
   - Forward pass s `output_hidden_states=True` + custom hooks na
     výstupy *všech* RMSNorm, conv1d, ssm, attention
   - Uloží 24 × 5 tensorů jako npz file
   Rust binárka, která:
   - Načte stejný model přes Candle
   - Forward pass s instrumentací (`trace::probe(&t, label)` u stejných
     bodů jako HF hooks)
   - Načte HF reference npz
   - Pro každý label spočítá `||rust - hf|| / ||hf||` (relativní L2)

4. **První run, divergence per layer.** Output:
   ```
   layer 0:
     rmsnorm_pre: 1.2e-7  ✓
     conv1d_out:  4.5e-7  ✓
     ssm_out:     2.1e-6  ✓
     attn_out:    8.3e-7  ✓
     mlp_out:     3.4e-7  ✓
   layer 1:
     rmsnorm_pre: 9.8e-7  ✓
     conv1d_out:  2.1e-6  ✓
     ssm_out:     1.8e-3  ✗ ← prudký skok
     attn_out:    2.4e-3  ✗
     mlp_out:     2.1e-2  ✗
   layer 2: divergence pokračuje exploding...
   ```
   **První bod, kde se relativní chyba prudce zvětší, je layer 1
   ssm_out.** Layer 0 ssm fungoval (2e-6 ≈ floating point precision
   limit). Layer 1 ssm ne. Co je rozdílné mezi layer 0 a layer 1 ssm?

5. **Hypotéza: state propagation z layer 0 → layer 1.** Layer 0 ssm
   začíná s nulovým initial state. Layer 1 ssm začíná se state z
   layer 0. **Pokud layer 0 vyrobil state, který je trochu off, layer
   1 startuje s vadným state, a chyba se znásobí.** Tohle by se ale
   projevilo i na layer 0 výstupu — a ten byl OK (2e-6).

6. **Druhá hypotéza: parallel scan implementation.** Layer 0 měl
   stejný prefill (multi-token), ale byl OK. Layer 1 ne. Rozdíl není
   v *které* layeru, ale v *které layer state*. Layer 0 začíná s
   nulou. Layer 1 začíná s něčím nenulovým. **Možná moje parallel
   scan má bug, který se projeví jen pro nenulový initial state.**

7. **Mikro-test: spustím layer 0 ssm s explicit nenulovým initial
   state.** Zaprasím ho na hodnoty z reference (load HF state, předaj
   do mé layer 0 ssm). **Output: divergence taky vystřelí.** Bug
   confirmed: parallel scan má issue s nenulovým initial state.

8. **Reading code: parallel scan implementation.** Otevřu mixer.rs.
   Vidím: parallel scan formula reset state na nulu na začátku každého
   chunku. **To je správně pro single-chunk inference, ale chybně
   pro multi-layer scénář, kde state je input.** Fix: prefer "carry"
   parameter ze předchozího volání jako initial state, místo hard
   reset.

9. **Po fix: re-run diff harness.** Všechny vrstvy: relativní chyba
   ≤ 1e-6. End-to-end inference produces coherent text. Reference
   matched. Bug closed.

10. **Reflexe: diff testing šetří hodiny i dny.** Bez něj jsem mohla
    týdny hádat, kde to padá. S ním jsem za hodinu (setup + run +
    interpret) měla přesnou lokalizaci v jedné konkrétní funkci v
    jednom konkrétním sub-modulu. Investice do harness se vyplatila
    okamžitě a bude se vyplácet znovu při budoucích numerických
    chybách.

## Aplikovatelné principy

- **Diff testing proti známé good implementaci je klíčový debugging
  nástroj pro numerické systémy.** End-to-end metrika ti řekne *že*
  je to rozbité; per-step diff ti řekne *kde*. Bez diff testing jsi
  v ML kódu slepá.
- **První bod, kde divergence vystřelí, je primary suspect.** Před
  tímto bodem všechny vrstvy fungují. Po tomto bodě rostou chyby
  exponenciálně (multiplikativní akumulace), takže se bug zdá být
  všude. Lokalizace = první vrstva, kde diff vyskočí nad noise floor
  (typicky ~1e-6 pro f32).
- **Numerická chyba blízko floating-point limit (1e-6 f32, 1e-13 f64)
  není bug, je to limit precision.** Numerická chyba >1e-3 je bug.
  Mezi tím (1e-5 až 1e-4) je gray zone — možná akumulace přes mnoho
  vrstev z legitimní precision loss, možná začínající bug. Watch for
  **trend** — pokud se chyba monotónně zvětšuje napříč vrstvami i
  na čistém setup, je to akumulace, ne bug.
- **Rozdíl mezi single-token a multi-token (nebo single-layer vs.
  multi-layer) je často signature parallel/sequential code path
  divergence.** Když "jednoduchá" varianta funguje a "komplexní" ne,
  bug je v complexity-specific code, ne v shared logic.

## Závěr

Diff harness:

```rust
// V Rust binárce s `--diff-against=hf_ref.npz`
let hf_ref = load_npz(&args.diff_against)?;
trace::start();
let _ = model.forward(&input_ids)?;
let entries = trace::finish();

println!("{:>30}  {:>10}", "label", "rel_l2");
for entry in entries {
    if let Some(ref_tensor) = hf_ref.get(&entry.label) {
        let diff = (&entry.tensor - ref_tensor)?;
        let rel_l2 = diff.l2()? / ref_tensor.l2()?;
        let status = if rel_l2 < 1e-5 { "✓" } else { "✗" };
        println!("{:>30}  {:>10.2e}  {}", entry.label, rel_l2, status);
    }
}
```

```python
# HF Python script generating reference
hooks = []
for i, layer in enumerate(model.model.layers):
    layer.norm.register_forward_hook(save_to(f"layer_{i}.rmsnorm_pre"))
    layer.mixer.register_forward_hook(save_to(f"layer_{i}.ssm_out"))
    # ... atd.
output = model(input_ids)
np.savez("hf_ref.npz", **all_saved_tensors)
```

## Přenositelný pattern

Kdykoli mám numerický systém (ML model, signal processing pipeline,
scientific computation, financial calculation), kde end-to-end output
je špatně:

1. **Najdi reference implementaci.** PyTorch/JAX/TensorFlow pro ML.
   MATLAB/Octave/SciPy pro scientific. Excel/Python pro finance.
   Reference implementace je *contract* — co tvůj kód má dělat.
2. **Instrumentuj obě implementace na stejných intermediate bodech.**
   Hooks v Pythonu, trace probes v Rustu, breakpoints + dump v
   debugger. Cíl: vyrobit "snapshot" intermediate state pro stejný
   input napříč implementacemi.
3. **Spočítej relativní L2 (nebo cosine, nebo max abs) na každém
   bodě.** Absolute hodnota je clueless (1.0 je velký rozdíl pro
   embedding, malý pro logits); relativní je porovnatelná napříč
   tensory.
4. **Najdi *první* bod, kde divergence vyskočí nad noise floor.**
   Ten bod je primary suspect. Vše před ním funguje, vše po něm
   je consequence.
5. **Mikro-test té konkrétní operace.** Generuj malý input, ověř
   chování. Nepoužívej celý model — minimální reprodukce ti dá
   rychlý feedback loop.

Pattern se přenáší: ML training (gradient diff testing), audio DSP
(per-stage spectrum compare), database query (compare query plan and
results across versions), distributed systems (trace ID across hops),
compiler backends (compare IR after each pass against reference).
Společný princip: **systém s mnoha kroky a numerickou citlivostí má
mít observable na každém kroku, ne jen na výstupu. Bez observable
jsi slepá; s observable je bug obvykle nelokalizován v deseti minutách.**
