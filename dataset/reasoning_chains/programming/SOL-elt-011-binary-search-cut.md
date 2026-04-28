# Reasoning chain — Binary search cut pro lokalizaci NaN backward

## Zdroj
`eleutheria/SOLUTIONS.md#SOL-011` — diagnostický nástroj pro BUG-010
(NaN gradient v multi-layer Mamba-2 backward).

## Kontext
24-vrstvý Falcon-H1, multi-layer Core Memory training, BF16. Forward
pass projde čistě, loss se spočítá. Backward vrací NaN. Vím, že někde
v 24 vrstvách × 5 komponent na vrstvu (RmsNorm pre, SSM, Attention,
RmsNorm post, MLP) je gradient explosion nebo numerická nestabilita.
Hrubě to dělá 120 míst, kde to může padnout. Běžnou cestou by bylo
projít všechny PR komentáře, podívat se do logů, hledat instrumentaci.
Ta tam ale není. Mám holý forward, holý backward, NaN ve výstupu.

## Analytický flow

1. **Hrubá lokalizace: kterou vrstvou problém začíná.** Existuje primitivní
   diagnostický nástroj — `--cut-at-layer K`: forward jede přes vrstvy
   0..K, pak rovnou loss, pak backward. Pokud cut=23 (full) dá NaN
   a cut=12 dá konečný gradient, problém je v intervalu [12, 23].
   Klasický binary search v lineární doméně.

2. **Implementace cut-at-layer trvá 15 minut** a redukuje problém ze
   24 vrstev na ~6 spuštění. Po výsledku: cut=22 OK, cut=23 NaN.
   **Problém je v poslední, 23. vrstvě**. Ne ve specifické vrstvě
   uprostřed, ne v residual stack akumulaci. V té poslední, kterou
   přidám.

3. **Tady cut-at-layer končí svou užitečnost.** Vrstva má pět vnitřních
   komponent (RmsNorm, SSM, Attention, post-norm, MLP) plus residual
   spojnice. Cut-at-layer mi řekne "uvnitř téhle vrstvy", ale ne *kde*.
   A guess-and-check ručním vypínáním komponent je drahý — každá
   změna kódu, každý rebuild, každý spuštění. Po třetí iteraci jsem
   ztracená v tom, co jsem kdy zkoušela.

4. **Otázka: existuje jemnější granularita cut bodu?** Logicky ano —
   uvnitř vrstvy je jasná lineární struktura: `pre_norm → ssm → attn →
   residual_1 → post_norm → mlp_gate → mlp_silu_mul → mlp_down → full`.
   Devět cut bodů. Pokud udělám `forward_until(stop)` s `enum LayerStop`,
   můžu binary-searchovat uvnitř jedné vrstvy se stejnou disciplínou
   jako mezi vrstvami.

5. **Je to nasazení lano-na-lano s metodou, co už funguje.** Zatímco
   první cut-at-layer redukoval 24 vrstev na 1, cut-at-component
   redukuje 9 možností na 1 v ~3 spuštěních. Spustím cut=ssm: NaN.
   Cut=attn: PASS. Cut=pre-norm: PASS. **Diagnostika: SSM branch
   backward je viník.** Tři spuštění, žádný guess.

6. **Proč to funguje tak dobře — je to logaritmus, ne rovina.** Kdybych
   to dělala "od shora dolů" (linear scan komponent), průměrně bych
   spustila 5 spuštění (pro 9 komponent). Binary search dává
   `log₂(9) ≈ 3.2` — stejně jako u sortování. Mock-up zní jako overkill,
   ale paradoxně je rychlejší napsat než hand-pick z výpisů.

7. **Pattern je teď reusable.** `LayerStop` enum + `forward_until` +
   CLI `--cut-at-component` jsou v repu. Příští BUG, který se projeví
   jako "něco padá uvnitř vrstvy", má diagnostiku ready. Investice
   půldne, návratnost první další iterace.

## Aplikovatelné principy

- **Hierarchické forward pass = hierarchický binary search.** Pokud
  výpočet má strukturu N úrovní × M kroků, log₂(N×M) cut bodů ti dá
  přesnou lokalizaci. Klíč je dostatečně jemná granularita — moc
  hrubé cut body nelokalizují, moc jemné jsou nepraktické.
- **Diagnostický infrastruktura > log statements.** Když debuguju
  numerickou nestabilitu, log statements mi řeknou *co* (NaN), ale
  ne *kde* (která komponenta). Cut bod říká, jaký výstup je ještě
  čistý — to je víc informací než hodnota samotná.
- **Investuj do diagnostiky, ne do guessování.** První cut-at-layer
  jsem psala 15 minut, použila 6×. Cut-at-component jsem psala
  další 30 minut, použila 3× ten den, plus pravděpodobně každý další
  bug se vrstvami. Time-to-fix klesá geometricky s lepšími nástroji.
- **Binary search funguje i v doménách, kde to nečekáš.** Klasické
  pojetí: search v setříděném poli. Reálná aplikace: search v kauzálním
  řetězci. Pokud máš lineární strukturu A→B→C→D a víš, že někde mezi
  A a D je problém, půlení mezi B a C ti řekne, ve které polovině
  hledat. Stejné jako binární vyhledávání v poli.

## Závěr

```rust
// falcon_h1/layer.rs
pub enum LayerStop {
    PreNorm,
    Ssm,
    Attn,
    Residual1,
    PostNorm,
    MlpGate,
    MlpSiluMul,
    MlpDown,
    Full,
}

impl FalconH1Layer {
    pub fn forward_until(
        &self, x: &Tensor, pos: usize,
        state: &mut LayerState, stop: LayerStop,
    ) -> Result<Tensor> {
        let h = self.pre_norm.forward(x)?;
        if stop == LayerStop::PreNorm { return Ok(h); }
        let ssm_out = self.ssm.forward(&h, state.ssm_mut())?;
        if stop == LayerStop::Ssm { return Ok(ssm_out); }
        // ... pokračuje pro každý cut bod
    }
}

// model.rs
impl FalconH1Model {
    pub fn forward_up_to_layer_with_stop(
        &self, x: &Tensor, layer_idx: usize, stop: LayerStop,
    ) -> Result<Tensor> { ... }
}
```

CLI: `--cut-at-component ssm|attn|pre-norm|...`. Aplikovaná diagnostika:
`cut=ssm` NaN + `cut=attn` PASS + `cut=pre-norm` PASS → SSM branch
identifikován jako viník BUG-010 ve třech spuštěních. Z toho vyšel
následný fix (lokální `silu` backward exploduje přes `recip(Inf)`,
delegace na `candle_nn::ops::silu`).

## Přenositelný pattern

Pattern aplikuju kdykoli debugovám hierarchický nebo řetězovitý systém,
kde failure se projevuje *na konci* a chci vědět *kde* uprostřed se
něco rozbilo. Konkrétní recept:

1. **Identifikuj lineární strukturu.** Forward pass přes vrstvy. Pipeline
   přes ETL kroky. Build přes phases. Render přes passes. Cokoli, co
   je řetězec A → B → C → ... → Z s pozorovatelnou hodnotou na každé
   hraně.

2. **Přidej "stop point" mechanismus.** Enum nebo flag, který říká
   "běž jen do bodu K, vrať mezivýstup". Z perspektivy productivity
   to vypadá jako overkill, ale často je to 30 řádků kódu a šetří
   hodiny při každém budoucím bugu.

3. **Binary search v lineárním prostoru.** Zkoušej cut=N/2. Pokud
   broken, hledej v [0, N/2]. Pokud OK, hledej v [N/2, N]. Po `log₂(N)`
   spuštěních máš lokalizaci na úrovni jednoho kroku.

4. **Když stop point není možný (irreversible step), zkus diff
   testing.** Místo "běž jen do K" porovnávej výstup tvé implementace
   s referencí po každém kroku. První divergence ti řekne, kde
   vznikla nepřesnost.

Pattern se přenáší napříč doménami: ML training (per-layer gradient
norm), build systems (`make -d` pro krok, kde to selhává), database
migrations (apply N-1 migrations, observe stav, apply N), distributed
systems (correlation IDs napříč hops), reverse engineering (instruction
trace s breakpointy). Všude tam, kde je *řetěz* a chceš vědět *který
článek* to praskl, je binary search rychlejší než lineární scan a méně
chybový než guessování.
