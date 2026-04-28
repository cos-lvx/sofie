# Reasoning chain — Bloom pass posun jako one-line architektonický pivot

## Zdroj
`iskra/SOLUTIONS.md#SOL-011` — bloom v render pipeline zachytával jen
část HDR contentu; refactor RF-007 chtěl rozšířit overlay na bloom.

## Kontext
Iskra má render pipeline s deseti passy: world, ui_glow, ui_material,
ui_glass, particles, overlay_glow, overlay, bloom, dof_blur, postprocess.
Aktuální pořadí má bloom pass na pozici 3 (po world a ui_glow). Důsledek:
bloom vidí emissive content z world geometrie a UI glow auru, ale **NE**
UI material highlights (panely s emissive material), glass Fresnel
highlights (lesknoucí se okraje glass widgetů), particles (HDR sparkles),
ani overlay_glow. Refactor RF-007 chce, aby všechny tyto hot kategorie
bloomovaly. Otázka: jak.

## Analytický flow

1. **První instinkt: přidat druhý bloom pass jen pro overlay.**
   Selektivní — current bloom zůstává pro world+ui_glow, druhý pass
   pro UI material + glass + particles + overlay. Architectonicky
   to vypadá clean (oddělení world bloom od UI bloom), ale prakticky:
   dvojnásobný gauss kernel, dvojnásobné mip chains, dvojnásobné
   memory pressure. Pro mobilní GPU (Iskra cílí i na low-end) je
   to měřitelně drahé.

2. **Druhá cesta: jeden bloom, ale na jiném místě.** Pokud bloom
   pass pojede *po* všech contentových passech, čte finální HDR
   scénu a zachytí všechno hot najednou. Ale tady mě brzdí
   instinkt: bloom je tradičně "post-effect" pass, který běží
   na finální composition. Nezní to jako něco, co bych měla
   *přidat doprostřed*. Pojďme se podívat, *kde přesně* by měl
   bloom logicky sedět.

3. **Mapuju, co se po bloom passu děje teď.** Po bloom (pozice 3)
   přijde ui_material, ui_glass, particles, overlay_glow, overlay,
   dof_blur, postprocess. Sedm passů následuje za bloomem — ale
   bloom *nečte* jejich výstup. Bloom čte **HDR offscreen**, který
   získává po pozici 3. Pak ten samý offscreen se *přidá k sobě*
   v ui_material (čte HDR offscreen jako podklad), ui_glass čte
   pozadí pro Fresnel sample, atd. **Po tomto všem je v offscreenu
   už finální HDR scéna včetně všech composing passů.**

4. **Klíčový poznatek z předchozího refaktoru (SOL-009, v0.25.0).**
   V refaktoru "Full HDR pipeline" jsem celou scénu přesunula do
   *jednoho* HDR offscreen, takže UI / overlay / particles všechny
   compositují do téhož bufferu, místo aby měly samostatné
   offscreens. Tonemap se aplikuje jednou na konci. Ten refactor
   řešil jiný problém (HDR consistency), ale *vedlejší efekt* je,
   že po pozici 9 (overlay_glow) je v HDR offscreen už **úplně
   všechno hot, co kdy v scéně bude**.

5. **Tady mi to cvakne.** Pokud posunu bloom *za* všechny composing
   passy, čte úplnou finální HDR scénu. Žádný druhý bloom kernel,
   žádný extra mip chain. **Jeden řádek change v `render_scene`:
   pozici bloom z 3 na 10**. Bloom binding (`offscreen.view`)
   zůstává, jen časování čtení se posune.

6. **Ověření, že nic neexploduje.** Trade-off: UI material panely
   s `emissive > 0` budou nyní bloomovat (předtím nebloomovaly).
   Default presety (BASALT, IRON, MARBLE) mají `emissive = 0` →
   no visual change. EMBER preset má emissive HDR colors → silnější
   aura, ale to **chce** uživatel, který EMBER zvolí. Bloom OFF
   (`bloom_intensity == 0.0`) → bez změny. Žádné regression risk
   pro existující content.

7. **Architektonický důsledek je hezčí, než jsem čekala.** Pipeline
   teď má jasnou trojfázovou strukturu: (1) Geometry & compositing
   (world → ui_glow → UI/overlay passes), (2) Post-effects (bloom
   → DOF blur), (3) Final tonemap. Cokoli "post-effect" se teď
   aplikuje na finální HDR offscreen. Pro budoucí post-effects
   (color grading, screen-space glow, vignette) je tu **čistá
   insertion point** — přidávám pass do fáze 2, není potřeba
   přemýšlet, kde přesně.

## Aplikovatelné principy

- **One-line refactor s velkým dosahem je obvykle známka, že
  předchozí decisions zaplatily.** Můžu posunout bloom o sedm pozic,
  protože SOL-009 (Full HDR pipeline) vytvořil sjednocený HDR
  offscreen. Bez něj by bloom musel skládat dohromady vícero
  bufferů — refactor by byl mnohonásobně dražší. Investice
  do správné architektury zhodnocuje *pozdější* změny.
- **Když volíš mezi "duplikovat za jiných okolností" a "posunout
  s jinými okolnostmi", podívej se po duplikátech v okolí.**
  Druhý bloom pass byl "duplikuj současný". Posun byl "změň
  okolnosti". Druhá cesta je téměř vždy lepší, ale vyžaduje vidět,
  *jaké okolnosti* by se musely změnit. Mít předchozí refactor
  (SOL-009) udělaný usnadnil rozpoznat tu možnost.
- **Pipeline ordering je architektonické rozhodnutí, ne implementační
  detail.** Pokud změníš pořadí pass v render pipeline, **měníš,
  co každý pass může vidět**. Bloom před ui_material vidí jen
  podklad. Bloom po ui_material vidí materiál. Stejný kód, jiná
  schopnost. Ordering zaslouží mental model, ne ad-hoc enumeraci.
- **Trojvrstvá pipeline (geometry / post-effects / tonemap) je
  reusable mental shape pro real-time rendering.** Pokud máš
  postprocess, všechno geometry musí proběhnout *před*. Pokud
  máš HDR, tonemap musí proběhnout *po* všech HDR ops. Mezivrstva
  (post-effects) je flexibilní zóna. Tahle struktura odráží
  fyziku procesu (světlo se sčítá → po-mapuje na display).

## Závěr

```rust
// renderer.rs::render_scene — change one line
fn render_scene(&mut self, ...) -> Result<()> {
    self.pass_world(...)?;          // pos 1
    self.pass_ui_glow(...)?;        // pos 2
    // self.pass_bloom(...)?;       // ❌ byl tady (pos 3)
    self.pass_ui_material(...)?;    // pos 3 (was 4)
    self.pass_ui_glass(...)?;       // pos 4 (was 5)
    self.pass_particles(...)?;      // pos 5 (was 6)
    self.pass_overlay(...)?;        // pos 6 (was 7)
    self.pass_overlay_glow(...)?;   // pos 7 (was 8)
    self.pass_bloom(...)?;          // ✅ teď (pos 8) — vidí celou HDR scénu
    self.pass_dof_blur(...)?;       // pos 9
    self.pass_postprocess(...)?;    // pos 10 (final tonemap)
    Ok(())
}
```

Žádný shader, pipeline, ani API break. Bloom binding `offscreen.view`
zůstává — jen časování čtení se posunulo. UI material panely s
`emissive > 0` nyní bloomují, default presety se nemění (mají emissive
= 0). Pipeline má jasnou trojfázovou strukturu pro budoucí post-effects.

## Přenositelný pattern

Pokaždé, když refaktoruju pipeline (render pipeline, build pipeline,
data pipeline, request handling pipeline), procházím tímto:

1. **Vykresli si pipeline jako sekvenci, ne jako bag-of-passes.**
   Co každý krok produkuje, co spotřebovává. Která data poteče
   napříč. Bez tohoto mental modelu nedokážu posoudit, co se
   stane při reorderingu.

2. **Pro každý reordering candidate se ptej: co teď vidí vs. co
   uvidí?** Bloom před UI vidí jen world; bloom po UI vidí všechno.
   Stejně: validation před DB write vs. po DB write — vidíš
   uložená data nebo jen návrh? Cache pass před vs. po enrichment
   — cachuješ raw nebo processed?

3. **Když candidate vyžaduje změnu kódu více než jednoho passu,
   nejde o reorder — jde o restrukturalizaci.** Reorder je clean,
   pokud se *jen* změní pořadí volání. Pokud se musí změnit
   shadery, signatury, datové struktury, je to refaktor s vyšším
   rizikem a měl by se posuzovat jinak.

4. **Trojvrstvá struktura je častý attractor.** Input → Transform
   → Output. Read → Compute → Write. Geometry → Effects → Display.
   Validate → Process → Persist. Když vidíš, že pipeline má
   přirozenou trojvrstvou strukturu, organizuj kolem ní. Když ne,
   nelep ji uměle, ale podívej se po tom, jaký je skutečný shape.

Pattern se přenáší daleko za rendering. Compiler passes (parse →
typecheck → optimize → codegen) mají stejnou logiku — ordering
určuje, co každý pass může pozorovat. Network middleware stack:
auth před routing vidí token, routing před handler vidí route,
handler vidí session. Stream processing: window → aggregate →
publish. Kdykoli má systém řetěz transformací, ordering je
**architektonické rozhodnutí**, a one-line shift v ordering může
mít stejnou váhu jako přepsaný subsystém. Ten one-liner ale obvykle
předpokládá, že předchozí refactor uvolnil konstrukci. Naučit se
poznat, kdy je terén připraven pro takovou drobnou ale silnou
změnu, je dovednost, kterou se nedoporučuju nahrazovat za
"přidejme druhý subsystém".
