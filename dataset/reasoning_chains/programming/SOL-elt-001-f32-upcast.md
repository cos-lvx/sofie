# Reasoning chain — F32 upcast pro numericky citlivé operace

## Zdroj
`eleutheria/SOLUTIONS.md#SOL-001` — Falcon-H1 BF16 forward pass produkuje
akumulační drift v normalizaci a aktivacích.

## Kontext
Falcon-H1 inferuje na GPU v BF16 — to je výchozí volba pro modern LLM,
osvědčená a paměťově efektivní. První kompletní forward pass projde,
loss jako u reference, log-likelihoods sedí. Pak začnu generovat delší
kontexty a dvacet tokenů hluboko text začne sklouzávat — ne katastrofálně,
ne nesmyslně, jen nepatrně mimo. Distribuce next-tokenů má jiný tail.
RmsNorm hodnoty se mírně rozhází mezi vrstvami způsobem, který reference
nemá. Není to bug, je to **drift**. Kompletní forward pass funguje jako
hodiny, jen jdou nepatrně rychleji.

## Analytický flow

1. **První instinkt: hardware error nebo race condition.** BF16 matmul
   na GPU je dobře definovaná operace, neměla by vykazovat nedeterminismus
   per-run. Ověřuju: dva běhy s identickým seedem dávají bit-identický
   výstup. Drift je deterministický. To znamená, že je v kódu, ne v hw.

2. **Otázka: kolik bitů přesnosti vlastně mám.** BF16 je 1 sign + 8 exp
   + 7 mantissa bitů. Sedm bitů mantissy znamená, že rozdíly menší než
   `1/128` relativní velikosti tiše zmizí. Pro single matmul to vůbec
   není problém — wide accumulator (FP32) drží přesnost mezivýpočtu,
   downcast je jen na výstupu. **Pro řetězec krátkých skalárních operací
   ale akumulátor neexistuje.**

3. **Mapuju, kde tento řetězec běží.** RmsNorm: `x / sqrt(mean(x²) + eps)`
   — sumace přes hidden dim, reciprocal, multiply. Pět operací, každá
   ztrácí 1–2 bity. SiLU: `x * sigmoid(x)` — sigmoid ztrácí přesnost
   v křídlech (extreme |x|). Softplus: `log(1 + exp(x))` — exp+log dvojice
   ztrácí přesnost u nuly i u velkých čísel, dvakrát. RoPE: cos/sin
   table lookup s f32 frequencemi, pak pointwise multiply. Sampling:
   softmax + cumsum + threshold srovnání. **Pět doménově odlišných
   míst, každé s vlastním důvodem, proč BF16 nestačí.**

4. **Hypotéza: drift = akumulovaná ztráta přesnosti v normalizaci.**
   RmsNorm se aplikuje před každou vrstvou. Pokud každá vrstva přidá
   chybu řádu `1e-3` v normě, po 24 vrstvách to je `2.4e-2`. To je
   přesně ten řád, kde se distribuce next-tokenů začínají odlepovat.
   Test: změřit `||x_my - x_ref||` po každé vrstvě s referenční
   implementací. Reference je v F32, moje v BF16 → divergence se má
   monotónně zvětšovat.

5. **Měření potvrdilo hypotézu.** První vrstva: relativní L2 chyba
   `2e-4`. Desátá vrstva: `1e-2`. Dvacátá čtvrtá: `5e-2`. Akumulace
   přesně tak, jak bych očekávala, kdyby každá vrstva přidala konstantní
   chybu v normě.

6. **Fix: F32 upcast jen v citlivých operacích, ne celý forward pass.**
   To je důležitý kompromis. Celý F32 forward by zdvojnásobil VRAM
   a zhruba zpomalil matmul o 30 %. F32 jen v normalizaci a aktivacích
   stojí pár procent času (jsou to malé operace) a nula extra paměti
   (mezivýsledky se uvolní hned). Implementace: `x.to_dtype(F32)?
   .rms_norm(...)?.to_dtype(BF16)?` jako lokální transformace, vrácení
   do BF16 hned po skončení citlivé sekce.

7. **Validace: re-run benchmark po fixu.** První vrstva: `2e-5` (10×
   lepší). Dvacátá čtvrtá: `5e-4` (100× lepší). Distribuce next-tokenů
   teď sedí s referencí v rámci numerické precision testu. Drift
   zmizel.

## Aplikovatelné principy

- **BF16 je formát pro storage a matmul, ne pro skalární řetězce.**
  Sedm bitů mantissy je hodně málo na řetězec 5 operací bez akumulátoru.
  Před každou operací se ptej: jak přesně chci, aby tohle vyšlo? Pokud
  odpověď je "v rámci 1 % je OK", BF16 stačí. Pokud "musí to být
  bit-perfect proti referenci", potřebuji F32.
- **Multiplikativní chyby se akumulují přes vrstvy.** Norma chybou 1e-3
  vypadá zanedbatelně, ale 24× za sebou dělá 2.4 %. To není šum,
  to je drift. Pokud trénuji nebo srovnávám s referencí, drift je
  zabíječ kvality.
- **F32 upcast je lokální optimalizace, ne globální politika.** Zvedat
  celý forward do F32 je mrhání. Identifikuj přesně ty operace, kde
  to bolí, a upcastuj jen je. Zbytek nechej v native precision.
- **Měření před a po má smysl měřit per-vrstva, ne jen na konci.**
  End-to-end rozdíl skrývá, kde se chyba akumuluje. Per-vrstva měření
  ti řekne, *kde* fixovat — ne jen *že* fixovat.

## Závěr

```rust
// Před opravou (BF16 throughout):
let normed = x.rms_norm(weight, eps)?;

// Po opravě (F32 upcast lokálně):
let normed = x
    .to_dtype(DType::F32)?
    .rms_norm(weight, eps)?
    .to_dtype(x.dtype())?;
```

Aplikováno systematicky na: RmsNorm, SiLU, softplus, RoPE frequency
generation, temperature sampling. Mimo tato místa zůstává forward pass
v BF16 — žádný globální dtype switch, žádná dependency na global state,
žádná zvýšená VRAM. Per-vrstva diagnostika potvrdila návrat na referenční
přesnost a generation kvality.

## Přenositelný pattern

Vždycky když uvažuju o numerickém formátu pro neuronovou síť (nebo
obecně pro řetězec floating-point operací), procházím tímto checklistem:

1. **Kde je akumulátor a kde není.** Matmul má wide accumulator —
   vstup BF16, accum FP32, výstup BF16. Skalární operace (norm, aktivace,
   reduce) accumulator nemají. Pokud je řetězec skalárních ops bez
   akumulátoru delší než 2–3 operace, **přesnost se rozjede**.

2. **Co je multiplikativní vs. aditivní.** Multiplikativní chyba se
   znásobuje napříč vrstvami (každá vrstva přidá relativní chybu).
   Aditivní chyba zůstává konstantní (přidá se jednou). U LLM jsou
   normalizace a aktivace v každé vrstvě → multiplikativní → akumuluje
   se přes hloubku.

3. **Identifikuj minimum invasive fix.** Globální upcast (celý forward
   v F32) řeší všechno, ale je drahý. Lokální upcast (jen citlivé ops)
   řeší 99 % a stojí pár procent. Cílený upcast s benchmarkem vyhrává.

4. **Testuj per-vrstva, ne jen end-to-end.** End-to-end rozdíl ti řekne
   *že* je to rozbité, per-vrstva ti řekne *kde*. U hlubokých sítí je
   rozdíl zásadní pro lokalizaci fixu.

Pattern se přenáší daleko za LLM. Platí všude, kde se počítá v low-precision
formátu řetěz operací: GPU compute shadery (FP16), DSP audio pipelines
(int16/int24), embedded systems s fixed-point arithmetic, blockchain
smart contracts s integer-only math (bez FP), dokonce i klasická
double-vs-single precision rozhodnutí ve scientific computing. Společný
jmenovatel: **každá precision je rozpočet, který se utratí. Otázkou není,
zda na to máš, ale kde to vydáš.** Identifikuj místa s nejvyšší citlivostí
a tam zvyš precision; zbytek nechej levný.
