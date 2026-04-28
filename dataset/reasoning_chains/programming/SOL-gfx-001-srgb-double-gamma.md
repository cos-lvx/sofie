# Reasoning chain — sRGB double-gamma a coordinate space mental model

## Zdroj
`sofie-gfx/SOLUTIONS.md#SOL-001` — barvy v UI vypadají vyblitě a moc
světlé, ale `#3a4a6b` přiřazený jako primary color je correct hex.

## Kontext
Stavím UI rendering pipeline ve wgpu. Buttony, panely, pozadí mají
barvy přiřazené přes hex literály — `#3a4a6b` pro primary blue,
`#f5f5f0` pro background. V Photoshopu vypadají hluboké, sytě tmavě
modré a slonovinově béžové. V mé aplikaci jsou *fluorescenční* — jako
kdyby si někdo kuli pustil na monitor laser. Modrá je svítivá, béžová
je téměř bílá. Barvy jsou hex-correct, ale visualně mimo. Něco zdvojuje
intenzitu.

## Analytický flow

1. **První instinkt: bug v hex parsování.** Otevřu `hex_to_rgba()`
   utilitu — `r = u8::from_str_radix(&hex[1..3], 16)? as f32 / 255.0`
   atd. Pro `#3a4a6b`: r=0.227, g=0.290, b=0.420. Tyto hodnoty jdou
   přímo do shaderu. Číselně OK, sedí to s tím, co Photoshop ukazuje
   jako float reprezentaci. **Bug nemůže být v parsování.**

2. **Hypotéza číslo dvě: GPU swap channels.** wgpu na různých backends
   může mít BGRA vs RGBA. Ověřuju surface format — `wgpu::TextureFormat::
   Bgra8UnormSrgb`. Channel order pro mé hex je správně RGBA, render
   pipeline vertex output má `[r, g, b, a]`. Swap by produkoval modrou
   z červené, ne svítivou modrou ze tmavé. **Bug taky není tady.**

3. **Pauza, podívat se na ten název: `Bgra8UnormSrgb`.** Surface format
   *má* sRGB v názvu. Co to konkrétně dělá? Otevřu wgpu docs:
   "Pixel values are interpreted as sRGB; sample reads convert to
   linear space, store writes convert back to sRGB." **Hardware mi
   automaticky aplikuje sRGB → linear na čtení a linear → sRGB na
   zápis.**

4. **Tady se mi rozsvítí.** Hex hodnota `#3a4a6b` je sRGB-encoded
   (to je standardní reprezentace barev pro displej, gamma 2.2 zhruba).
   Když ji parsuju jako `r=0.227` a pošlu do shaderu, **shader si myslí,
   že je to lineární hodnota**. Pak GPU při zápisu do `Bgra8UnormSrgb`
   surface aplikuje *další* linear → sRGB konverzi. **Dvojí gamma:**
   sRGB v interpretaci hex → "linear" v shaderu (chybně) → sRGB při
   zápisu = konečný pixel je dvakrát zesvětlený. Proto fluorescence.

5. **Kde fix patří.** Mám dvě cesty: (a) konvertovat sRGB → linear
   už při hex parsování, takže shader pracuje skutečně lineárně, GPU
   pak ze shader output do surface dělá jednu correct linear → sRGB
   konverzi; (b) změnit surface format na `Bgra8Unorm` (bez sRGB
   suffixu), nechat shader pracovat v sRGB end-to-end, žádná hw
   konverze. Cesta (a) je správná architecturally — pokud někdy
   přidám blending nebo alpha compositing, blending v lineárním
   prostoru je fyzikálně správný (lineární barvy se sčítají jako
   světlo), v sRGB prostoru produkuje nesprávné mixy.

6. **Implementace `hex_to_linear()`.** sRGB → linear konverze není
   prostě `^2.2`, je to piecewise function: pro `c ≤ 0.04045` lineární
   (`c / 12.92`), pro `c > 0.04045` mocnina (`((c + 0.055) / 1.055)^2.4`).
   Není to jen kosmetika — pro tmavé barvy (`#3a4a6b` má kanály ≤ 0.42)
   je piecewise nutná pro správnou hodnotu poblíž nuly.

7. **Validation: barvy teď vypadají jako v Photoshopu.** Tmavá modrá
   je tmavá, béžová je béžová. Žádná fluorescence. Side benefit:
   alpha blending teď produkuje fyzikálně realistické překryvy
   (overlay 50% white + 50% black = mid-gray, ne fluorescent
   gray-yellow).

## Aplikovatelné principy

- **Pixel format v graphics API je smluvní vztah, ne dekorace.**
  Když API říká `*Srgb`, *aplikuje* sRGB konverzi v hw. To není
  metadata, to je transformace. Pokud netuším, kdy se aplikuje,
  garantovaně skončím s double-gamma nebo missing-gamma.
- **Color space je coordinate system.** Stejný vector může mít různou
  hodnotu v různých systémech. RGB hex (`#3a4a6b`) v sRGB prostoru
  ≠ stejné `[0.227, 0.290, 0.420]` v lineárním prostoru. Než budu
  počítat s barvou, musím vědět, ve kterém prostoru se nacházím.
- **Compositing v lineárním prostoru je fyzikálně správný; v sRGB
  je tradiční ale rozbitý.** Sčítání barev jako fotonů funguje
  lineárně (světlo se přidává). sRGB compositing produkuje "muddy"
  blending — proto pre-2010s hry vypadaly tak nějak špinavě, když
  se míchaly textury. Industry shift k linear workflow byl právě
  o tomto.
- **Když něco vypadá `2× příliš silně`, podezřívej double application
  transformace.** Double gamma, double scale, double rotation, double
  alpha. Hodnota `2×` je signature, že nějaká transformace běží
  dvakrát místo jednou.

## Závěr

```rust
// hex_to_linear, ne hex_to_rgba
fn hex_to_linear(hex: &str) -> [f32; 4] {
    let srgb_to_linear = |c: f32| {
        if c <= 0.04045 { c / 12.92 }
        else { ((c + 0.055) / 1.055).powf(2.4) }
    };

    let r = u8::from_str_radix(&hex[1..3], 16).unwrap() as f32 / 255.0;
    let g = u8::from_str_radix(&hex[3..5], 16).unwrap() as f32 / 255.0;
    let b = u8::from_str_radix(&hex[5..7], 16).unwrap() as f32 / 255.0;
    [
        srgb_to_linear(r),
        srgb_to_linear(g),
        srgb_to_linear(b),
        1.0,
    ]
}
```

Surface zůstává `Bgra8UnormSrgb`, takže hw udělá lineární → sRGB při
finálním zápisu. Shadery pracují v lineárním prostoru, alpha blending
i emissive mixing teď fyzikálně dávají smysl.

## Přenositelný pattern

Pokaždé, když pracuju s graphics nebo vizuální výstup, kontroluju
**color space pipeline** explicitně. Otázky, které si klade jako první:

1. **V jakém prostoru přichází input?** Hex literál: sRGB. Texture
   z disk: závisí na formátu (PNG je obvykle sRGB, EXR lineární).
   Camera RAW: lineární. Photoshop CMYK: úplně jiný prostor s gamma.
2. **V jakém prostoru pracuje můj výpočet?** Modern GPU pipeline:
   shader v lineárním prostoru. Legacy pipeline: shader v sRGB
   prostoru. Wgpu / Metal / Modern Vulkan: lineární. Old OpenGL
   bez sRGB framebufferu: sRGB.
3. **V jakém prostoru očekává output můj backend?** Surface format
   `*Srgb` aplikuje konverzi automaticky. Surface bez sRGB potřebuje
   manuální linear → sRGB v shaderu. Tape recording / video out:
   typicky sRGB nebo Rec.709. HDR display: jiný prostor.

Pattern se přenáší daleko za color: do **jakékoli situace, kde existuje
interpretace dat napříč rozhraními**. Audio: dB vs lineární amplitude,
sample rate konverze. Geometrie: world space vs screen space vs NDC,
matrices musí cestovat ve správném pořadí. Time: UTC vs local timezone,
unix epoch vs ISO 8601. Money: minor units (cents) vs major units (dollars),
implicitní currency. Internationalization: UTF-8 bytes vs Unicode
code points vs grapheme clusters. **Každá doména má svůj coordinate
system, a každá hranice mezi systémy je místo, kde se vytváří chyba.**

Konkrétní disciplína: **na každé hranici typu si pojmenuj prostor**.
Místo `Vec3` raději `WorldPos` a `ScreenPos`. Místo `f32` raději
`SRgbColor` a `LinearColor`. Type system se pak postará, abych je
nemíchala. Když jazyk nepodporuje newtype patterny, dělám to alespoň
v komentářích a názvech funkcí (`hex_to_linear`, ne `hex_to_color`).
Disciplína v pojmenování dělá rozdíl mezi pipelinami, kde double-gamma
projde nepoznán, a těmi, kde compiler nebo recenzent okamžitě vidí
"ty mícháš sRGB s lineárním".
