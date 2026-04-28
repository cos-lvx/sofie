# Reasoning chain — DOF blur reuse glass Kawase pattern

## Zdroj
`iskra/SOLUTIONS.md#SOL-008` — Depth-of-field blur v postprocess shaderu
byl omezený na ~3px offset; pro silnější DOF chtěl rozšíření, ale inline
expanze nebyla cesta.

## Kontext
Iskra renderer má depth-of-field efekt — UI elementy v "back" elevation
vrstvě se mírně rozmazávají, takže se "dostávají do hloubky". Aktuální
implementace: inline 5-tap cross blur v `postprocess.wgsl`. Pro `dof_strength
≤ 0.3` to vypadá OK. Pro `dof_strength > 0.5` (silnější rozostření) se
začínají objevovat aliasing artefakty — kernel je moc úzký na to, aby
sample-oval dost daleko od centra. Naivní cesta: rozšířit kernel z 5-tap
na 9-tap nebo 13-tap. Tím se ale narůstá fragment shader cost lineárně
s počtem tapů, a 3 fullscreen passy s 13-tap sample by byly drahé.

## Analytický flow

1. **Co existuje v repu, co řeší podobný problém?** Iskra už má **glass
   pipeline** — context menu, palette, toast používají Kawase dual-filter
   blur (3-level downsample/upsample chain) přes mip chain. Kawase je
   exactly designovaný pro vysoký kernel-equivalent při low per-sample
   cost. Glass blur dává equivalent ~10-12px Gaussian při ~7 fullscreen
   passes celkem, místo lineárně rostoucího single-pass kernelu.

2. **Otázka: aplikuje se Kawase pattern stejně na DOF?** Glass blur:
   surface format (sRGB), kopíruje surface → blur source, blur, mix
   s tint + Fresnel. DOF blur: HDR offscreen format (Rgba16Float),
   blur celé scény, mix s sharp version podle elevation. Strukturálně:
   **stejný blur algorithm, jiný format a jiná destination**. Pokud
   sdílím blur shadery a pipeline shape, mám identitu pattern reuse.

3. **Co je jiné a proč.**
   - **Vstup:** Glass musí kopírovat (surface není texture-bindable
     directly). DOF má offscreen už jako texture-bindable HDR
     buffer. *DOF nepotřebuje copy step* — to je win.
   - **Format:** Glass = surface sRGB, DOF = HDR Rgba16Float. Shader
     ale samply je generic — nezávisí na format, jen na fragment
     pipeline configuration. **Stejný `ui_blur.wgsl`, jiná pipeline
     descriptor.**
   - **Upsample blend mode:** Glass uses replace (compose blurred
     pozadí s glass tint). Bloom uses additive (akumulace energie
     mezi mip levels). **DOF chce *replace*, jako glass — energy
     conservation, ne accumulation.** Chcu mít kontrolovaný blur,
     ne zesvětlit.
   - **Trigger condition:** Glass se spouští pouze pokud `glass_shapes`
     existuje. DOF se spouští pouze pokud `dof_strength > 0.001`.
     Stejný "skip if not used" pattern.

4. **Implementace: structural copy s small overrides.** Vytvořím
   `DofBlurPipeline` v `renderer/dof.rs`. Skoro identický s
   `UiBlurPipeline` (glass), jen s:
   - HDR format místo surface format v texture descriptors
   - `BlendState::REPLACE` v upsample (stejný jako glass; verbatim)
   - Skip condition na `dof_strength`
   - Vstup direct na `offscreen.view` (žádný copy step)
   Vlastně ~80 % kódu je copy-paste z glass pipeline. Sdílený
   `ui_blur.wgsl` shader (`kawase_down_fs`, `kawase_up_fs`) zůstává
   beze změny — funguje pro libovolný pixel format.

5. **Postprocess shader: změna z 5-tap inline na single sample
   blurred buffer.** Aktuální postprocess dělal:
   ```wgsl
   let s0 = textureSample(offscreen, uv);
   let s1 = textureSample(offscreen, uv + vec2(1px, 0));
   let s2 = textureSample(offscreen, uv - vec2(1px, 0));
   let s3 = textureSample(offscreen, uv + vec2(0, 1px));
   let s4 = textureSample(offscreen, uv - vec2(0, 1px));
   let blurred = (s0 + s1 + s2 + s3 + s4) * 0.2;
   let factor = smoothstep(0.0, 1.0, dof_strength * elevation);
   let final = mix(s0, blurred, factor);
   ```
   Po fix:
   ```wgsl
   let sharp = textureSample(offscreen, uv);
   let blurred = textureSample(dof_blur_result, uv);
   let factor = smoothstep(0.0, 1.0, dof_strength * elevation);
   let final = mix(sharp, blurred, factor);
   ```
   **Jeden textureSample navíc místo 5 samples** — fragment shader
   je levnější. Ale upfront mám 6 dodatečných fullscreen passes
   (3 down + 3 up). Pro typical scénu kde DOF je zapnutý jen na
   některých framech, je to good trade-off.

6. **Trade-off explicit: kdy to vyhraje, kdy ne.**
   - **Vyhraje:** Když je DOF zapnutý často a chci silný blur. Kawase
     dává ~8-12px gaussian equivalent stable, žádné artefakty.
     Inline 5-tap + dof_strength = 0.7 dává viditelné aliasing.
   - **Nevyhraje:** Když je DOF *vypnutý* většinou. Pak inline 5-tap
     by aspoň nepotřeboval extra mip chain pamět (~6MB on 4K HDR).
     Skip condition na `dof_strength < 0.001` ale řeší — když OFF,
     pass se vůbec nespustí.

7. **Reflexe: pattern reuse je multiplikátor.** Místo dvou samostatných
   blur implementací mám jeden shared shader file, dvě structurally-similar
   pipelines, každá s vlastním small subset specifik. Když refaktor
   blur algoritmu (RF-016 zlepšuje Kawase weights), refactoruje se
   na jedním místě a obě pipelines benefit. Když přijde třetí blur
   use case (motion blur, shadow soft edges), structure je tam.

## Aplikovatelné principy

- **Repository je knihovna patternů, ne jen kódu.** Když řeším problém,
  první otázka: máme už pattern, který řeší podobný shape problému?
  Glass blur byl pro úplně jiný cíl (UI overlay translucency), ale
  *jeho shape* (Kawase dual-filter mip chain s replace blend) řešil
  můj problém taky.
- **Reuse ≠ generic abstraction.** Místo abstrahování `BlurPipeline<T>`
  s generic format/blend, dvě specific pipelines sharing shader.
  Kód duplikace ~50 řádků, abstrakce by byla 200 řádků generic
  trickery. Verbatim copy s strategic differences vyhrává v
  read-clarity i maintenance.
- **"Just expand kernel" je fragile shortcut.** Když existuje principled
  way k řešení (Kawase je proven post-process technika), lepší cesta
  je investovat do principled, ne do shortcut. Shortcut by se vrátil
  jako "DOF nestačí silný" znova za měsíc.
- **Sdílený shader, různé pipeline descriptors.** WGSL shader je
  format-agnostic většinou — `textureSample(t, uv)` returns vec4
  bez ohledu na underlying format. To umožňuje sdílet shader napříč
  pipelines s different texture formats, just by binding different
  textures. Klíčový enabling pattern.

## Závíer

```rust
// renderer/dof.rs (mostly mirror of renderer/glass.rs)
pub struct DofBlurPipeline {
    down_pipeline: wgpu::RenderPipeline,
    up_pipeline: wgpu::RenderPipeline,
    mips: [DofMip; 3],
    bind_groups_down: [wgpu::BindGroup; 3],
    bind_groups_up: [wgpu::BindGroup; 3],
}

impl DofBlurPipeline {
    pub fn new(device: &Device, format: TextureFormat) -> Self {
        let shader = device.create_shader_module(include_wgsl!("ui_blur.wgsl"));
        // ... format = Rgba16Float (HDR) ...
        // Same shader, different format binding than glass pipeline.
    }

    pub fn execute(&self, encoder: &mut CommandEncoder, src: &TextureView) {
        // 3 down passes, 3 up passes, REPLACE blend (energy conserving).
    }
}

// renderer.rs::render_scene
if dof_strength > 0.001 {
    self.dof_blur.execute(encoder, &offscreen.view);
}
self.postprocess(encoder, &offscreen.view, &dof_blur_result.view, ...);

// postprocess.wgsl
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sharp = textureSample(offscreen_tex, linear_sampler, in.uv);
    let blurred = textureSample(dof_blur_tex, linear_sampler, in.uv);
    let factor = smoothstep(0.0, 1.0, dof_strength * elevation);
    return mix(sharp, blurred, factor);
}
```

## Přenositelný pattern

Kdykoli stojím před implementací nového feature, ptám se:

1. **Existuje v repu už pattern, který řeší *shape* tohoto problému?**
   Ne hledám identitu (přesně tento feature), ale shape (struktura
   problému). Mip chain blur je shape — glass, DOF, screen-space
   reflection, motion blur všechny mohou ten shape sdílet.

2. **Co je v podobném patternu *jiné* a proč?** Format, target,
   blend mode, trigger condition. Pokud tato seznam je krátký (3–5
   bodů), reuse je správná odpověď. Pokud je dlouhý (10+ bodů),
   nový pattern je správná odpověď.

3. **Sdílím *implementaci* nebo *abstrakci*?** Sdílení implementace =
   copy code, lokálně modify. Sdílení abstrakce = generic type/trait,
   parameterized by differences. Pro 2 use cases obvykle implementation
   sharing vyhrává; pro 5+ use cases abstrakce začíná dávat smysl.

4. **Můžu reuse zlepšit pro both consumers?** Při refaktoru
   shared shader nebo pipeline, oba consumeři benefit. Pokud refactor
   vyžaduje rozdělení (jeden chce A, druhý chce B), pattern už není
   shared — je to coincidence.

5. **Trade-off explicit, ne implicit.** Pro DOF: 6 fullscreen passes
   navíc, 6MB extra HDR memory, no inline shader. Win: stable kernel
   na libovolný blur strength. Když píšu PR / commit message, trade-off
   jde do textu. Implicit trade-off se vrací jako bug report o
   performance later.

Pattern se přenáší daleko za rendering. Database query optimization
(reuse query patterns napříč endpointy), API design (consistent
pagination patterns), state management (reuse reducer shapes), build
systems (shared CI templates). Společný princip: **kód, který vypadá
podobně, by měl být podobně, a kód, který je *fundamentálně* podobně,
by měl být sdílený.** Disciplína v rozeznávání podobnosti shape vs.
jen visuální podobnosti je hluboká dovednost — když vidím repeat,
ptám se "je to opravdu repeat, nebo náhoda".
