# Reasoning chain — Scissor rect clamping a validation na hraně

## Zdroj
`morana/SOLUTIONS.md#SOL-006` — wgpu validační panic na Windows při
resize okna: scissor rect přesahoval surface velikost.

## Kontext
Iskra renderer používá GPU hardware scissor pro UI clip regions
(ScrollView, panely s overflow). Když Painter řekne "kresli teď
v rectanglu (x, y, w, h)", renderer před `RenderPass` zavolá
`set_scissor_rect(x, y, w, h)` — GPU pak zahodí všechny fragmenty
mimo tento rect. Nulový performance overhead, perfectní hardware
support. Funguje hladce na Linuxu i Mac. Na Windows + Vulkan: panic
po resize okna. Stack trace: `ValidationError: scissor rect (0, 0,
1920, 1080) exceeds render target dimensions (1280, 720)`. Konkrétně
to spadne *po* resize z 1920x1080 na 1280x720, na druhém renderu
v té smaller velikosti.

## Analytický flow

1. **Pozoruji platform divergence znovu — ale tentokrát s validation
   error, ne s tichým no-op.** Linux a Mac mají buď permisivnější
   validation, nebo Vulkan API není exposovaný stejně. Windows +
   Vulkan vrstva `VK_LAYER_KHRONOS_validation` aktivně kontroluje,
   že scissor rect je ⊆ render target. Jiné platformy možná validaci
   nedělají, nebo dělají později.

2. **Důležitý detail: Linux a Mac to *neproháněly* — byl to skrytý
   bug.** Při dalším resize na large velikost by se validace neprojevila.
   Na resize z 1920x1080 na 1280x720 se ale naopak **bug objevil**
   — scissor z předchozího framu (1920x1080) přesahuje nový surface
   (1280x720). Vulkan validation panickne. Bez Vulkan validation
   by GPU možná nevadil, ale výsledek by byl undefined behavior.
   **Validation panic mě zachránil před runtime tichou chybou.**

3. **Trace zpět: kde se scissor rect bere?** UI layout vypočítává
   clip regions v jednotkách logical pixels (DPI-aware). Renderer
   konvertuje na surface coordinates při `set_scissor_rect`. Layout
   je z předchozího framu — frame N-1 layout řekl "panel je
   1920x1080 minus padding", scissor regiony reflektují to.
   Frame N přijde po resize, surface je 1280x720, ale layout ještě
   nebyl přepočítán (přepočítá se v dalším update tick).

4. **Hypotéza fix č. 1: layout počítat synchronně po resize.** To je
   "make resize blocking" — počkej s renderem, dokud layout neproběhne.
   Cena: latence po resize, layout pass může být drahý, blocking
   v event handleru je rude pro user experience. Funkční, ale
   architektura je posunutá.

5. **Hypotéza fix č. 2: clamping na renderer hranici.** Místo opravy
   layout pipeline (fragile, drahá), validuju scissor rect na
   *hranici*, kde se používá. `set_scissor_rect` dostane hodnoty,
   které mohou být outdated — clampnu je na surface velikost.
   Cena: 4 řádky kódu. Risk: scissor pak ořeže menší area než
   layout chtěl, ale nikdy větší, takže *visual* je acceptable
   (worst case: user vidí trochu menší panel než UI očekává, jen
   pro 1 frame, dokud se layout nepřepočítá).

6. **Implementace clampingu.**
   ```rust
   for clip in clip_regions {
       // Origin musí být uvnitř surface; jinak segment přeskoč
       if clip.x >= surface_w || clip.y >= surface_h { continue; }
       // Clamp velikost na surface (od origin)
       let w = (clip.w).min(surface_w - clip.x).max(1);
       let h = (clip.h).min(surface_h - clip.y).max(1);
       render_pass.set_scissor_rect(clip.x, clip.y, w, h);
       // ... draw calls v tomto clip ...
   }
   ```
   Dva edge cases:
   - **Origin mimo surface:** scissor (1500, 0, 100, 100) na surface
     1280x720 — origin (1500, 0) je za rightem. `continue` — nic
     z tohoto clipu není viditelné, skip celý draw segment.
   - **Velikost nulová:** wgpu odmítá `width = 0` nebo `height = 0`
     jako invalid. `.max(1)` garantuje minimum 1px.

7. **Reflexe: defense in depth.** Layout pipeline by *měl* držet clip
   regions in-bounds. Pokud bug se objeví v layoutu (forgotten resize
   handler, race condition v scroll position), defense at renderer
   boundary mě uchrání před runtime panic. **Validation se aplikuje
   na hranicích kontextu, nejen u sourceu.** Stejně jako server
   validuje user input, renderer validuje layout output. Layer
   validation je kombinována, ne distribuována.

## Aplikovatelné principy

- **Validation na hraně systému je defense in depth.** I když source
  data *má* být valid, validace na consumer-side chytne bugs v producer
  bez celé pipeline reorg. Stejný princip jako server validuje JSON
  payload, ne jen relies na frontend, že to bude správné.
- **Clamping je cheap recovery, panic je expensive.** Zvolit mezi
  panic ("layout je rozbitý!") a graceful clamp ("ukaž to nejlepší
  approximaci") je obvykle no-brainer pro non-critical operace.
  Validation panic ve framework, který má 60 fps, je catastrophic.
  Frame s mírně off layoutem je acceptable.
- **Validation layers jsou debug aid, ne enemy.** Vulkan validation
  vrstva odhaluje bugs, které by jinak byly silent corruption. Když
  validation panickne, je to *fakt* — produkce by měla stejný bug,
  jen ne s loud signal. Fix se aplikuje protože validation je
  pravda, ne aby validation umlčela.
- **Po resize jsou frames v "transient" state.** Mezi `Resized` event
  a tím, kdy layout se přepočítá, je okno, kde data z minulého layoutu
  nemusí sedět s novým surface. Renderer musí být tolerant na tenhle
  gap — buď `continue` (skip frame) nebo clamp.

## Závěr

```rust
// V renderer.rs, draw_ui_clip_segments():
for segment in &scene.ui_clip_segments {
    let clip = segment.scissor;

    // Skip segment, pokud origin mimo surface
    if clip.x >= surface_w || clip.y >= surface_h {
        continue;
    }

    // Clamp velikost (≥ 1px, ≤ surface - origin)
    let w = clip.w.min(surface_w.saturating_sub(clip.x)).max(1);
    let h = clip.h.min(surface_h.saturating_sub(clip.y)).max(1);

    render_pass.set_scissor_rect(clip.x, clip.y, w, h);
    // ... draw calls ...
}
```

`saturating_sub` chrání proti underflow (kdyby `clip.x > surface_w`,
ale to už řeší předchozí `if continue`).

## Přenositelný pattern

Kdykoli mám API, které vyžaduje `value ∈ [min, max]` a value pochází
z předchozí fáze, kde to garantováno *měla* být ale není 100% jisté:

1. **Zvol mezi blocking sync a graceful clamp.** Blocking ("počkej
   na update layoutu před renderem") je správné v některých kontextech
   (database transaction integrity), ale fragile v real-time pipelinech.
   Clamp ("ukaž best approximation") je obvykle správné v UI/rendering.

2. **Validation na consumer side, ne jen na producer.** Producer
   *má* dělat validní data, ale když ne, consumer je posledni linie
   defense. Bezpečnost (ASLR, ROP, JIT) tohle dělá důsledně. Také
   dobrý GUI framework, dobrý compiler frontend, dobrý API server.

3. **Edge cases: empty, negative, beyond-bounds.** Pro každý input
   range kontroluju: co když je hodnota přesně na hraně? Mimo? Záporná?
   Velmi velká (overflow)? `set_scissor_rect(0, 0, 0, 0)` selže;
   `set_scissor_rect(u32::MAX, 0, 1, 1)` taky. Rozsah validity je
   úzký, hranice jsou nesymetrické.

4. **Saturating arithmetic je tvůj kamarád.** `.saturating_sub`,
   `.saturating_add`, `.saturating_mul` v Rustu, `Math.min/max`
   chains v JS, clamp helpers v většině jazyků. Místo manuálního
   if-overflow check je krátký compose volání compute-with-bounds.

5. **Test pro transient/rare state.** Test na initial render frame,
   test na frame po resize, test na frame po focus loss. Tyto stavy
   se nedostanou do unit testů normálně — chytí je integrační test,
   který simuluje *full lifecycle*, ne *steady state*.

Pattern se přenáší: HTTP client retry policies (clamp delay between
attempts), database connection pool (clamp concurrent queries),
animation system (clamp interpolation t na [0, 1]), audio sample
clipping (clamp -1.0..=1.0), network protocol (clamp packet sizes).
Společný invariant: **na hranici systému, kde data přechází z jednoho
domain do druhého, validuj. Nemůžeš věřit, že producer tě nezklamal,
i kdyby měl. Defense in depth není paranoia — je to disciplína.**
