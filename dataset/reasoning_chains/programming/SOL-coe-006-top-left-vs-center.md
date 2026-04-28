# Reasoning chain — Top-left vs center: konvence na hraně API

## Zdroj
`coe/SOLUTIONS.md#SOL-006` — panel renderoval malý čtverec v levém horním
rohu místo plné velikosti. `RoundedRect { rect: Rect::new(32, 32, 1216, 656) }`
projít přes `ScenePassBuilder::push` do Iskra.

## Kontext
COE staví UI nad Iskrou (renderer). COE má vlastní `Rect { x, y, w, h }`
s **top-left** konvencí — `(x, y)` je levý horní roh, jako HTML canvas,
CSS, většina UI frameworků. Iskra interně pracuje se shape primitivy
typu `ShapeInstance::rect(x, y, w, h, color)`, kde `(x, y)` je **center**
shape (důsledek toho, že Iskra cílí glyph quads a sprite particles —
center makes sense pro instancing). Mezi nimi `ScenePassBuilder::push`
v sofie-gfx v0.1.16 dělal **přímý mapping bez konverze** — předal
top-left souřadnice do Iskřinou center API. Pro panel s top-left (32, 32)
a velikostí (1216, 656) byl Iskra center skutečně (32, 32) → panel
center na okraji obrazovky → vidíme jen pravý dolní kvadrant uvnitř
viewportu.

## Analytický flow

1. **Pozoruju, co vidím vs. co očekávám.** Očekávaný panel: full-window
   minus 32px padding ze všech stran. Skutečný panel: maličký čtverec
   v levém horním rohu, asi 1/4 velikosti. **Visual signature je
   jasná: rohové bití, ne random offset, ne squashing.** Něco s
   konvencí pozice, ne s velikostí.

2. **Hypotéza č. 1: typo v rect coordinates.** Otevřu `build_hello_scene`,
   `Rect::new(32, 32, 1216, 656)`. Sedí. Žádný typo, žádný hack.

3. **Hypotéza č. 2: viewport / scaling.** Pokud framework škáluje DPI
   nebo aplikuje transform, mohlo by to vést k srolovanému layoutu.
   Ale potom by panel byl *menší*, ne *posunutý do rohu*. Nesedí.
   Také: ostatní UI prvky (text, knoflíky) jsou na správných místech
   — jen panel je rozbitý.

4. **Hypotéza č. 3: konvence top-left vs. center.** Myslím si: COE
   používá top-left, ten panel byste měli vidět full-screen. Iskra
   API volá `ShapeInstance::rect(x, y, w, h, color)`. **Doc komentář
   na position field říká: "World position of glyph quad center".**
   Tady to je. Iskra interpretuje `(x, y)` jako střed shape, ne levý
   horní roh.

5. **Verifikace: matematika sedí na pozorování.** Pokud Iskra dostane
   center (32, 32) s šířkou 1216 a výškou 656, shape se rozprostírá
   od (32 - 608, 32 - 328) = (-576, -296) do (32 + 608, 32 + 328) =
   (640, 360). Z toho viditelná část je v rectangle (0, 0) → (640, 360),
   což je pravý dolní kvadrant od top-left (0,0). **Matematika přesně
   sedí na pozorovaný visual.** Diagnóza potvrzená.

6. **Kde fix patří: na hranici mezi systémy s odlišnými konvencemi.**
   COE používá top-left interně (správně, většina UI frameworků taky).
   Iskra používá center interně (správně, optimální pro instanced
   primitives). **Fix patří do hranice — `ScenePassBuilder::push`,
   funkce, která mapuje COE Rect → Iskra ShapeInstance.** Tam udělám
   explicitní konverzi:
   ```rust
   let cx = rect.x + rect.w * 0.5;
   let cy = rect.y + rect.h * 0.5;
   let shape = iskra::shapes::ShapeInstance::rect(cx, cy, rect.w, rect.h, color);
   ```

7. **Po fixu: regression test, aby se to nikdy nevrátilo.**
   `push_rect_converts_top_left_to_center` ověří přesný numerický
   převod. To není test "vypadá to vizuálně OK" — je to test, že
   `Rect::new(32, 32, 100, 100)` se v Iskra position ulodí jako
   `(82, 82)`. Když se v budoucnosti někdo dotkne `ScenePassBuilder::push`,
   test selže okamžitě.

8. **Sekundární improvement: zlepšit Iskra API.** Zapsala jsem RF-036
   do Iskra REQUESTED-FEATURES.md — návrh přejmenovat parametry
   `ShapeInstance::rect(x, y, ...)` → `ShapeInstance::rect(cx, cy, ...)`,
   nebo přidat alternativní `rect_at_top_left()` variant. Cíl: typová
   signatura sama signalizuje konvenci, doc komentáře nejsou primary
   source of truth. Nízká priorita — workaround v COE funguje, ale
   pokud se Iskra někdy refaktoruje, parametr name by měl převzít
   semantický důraz.

## Aplikovatelné principy

- **Konvence (top-left vs. center, 0-based vs. 1-based, cm vs. pixel,
  zero-indexed time vs. timestamps) musí být explicitní v API hranicích.**
  Doc komentář je nedostatečný — code review ho přečte jednou, refactoring
  pass ho přehlédne, IDE ho nezvýrazní. Parametr name nebo type by měl
  konvenci nést.
- **Když mapuju mezi dvěma systémy s odlišnými konvencemi, vždy napiš
  regression test pro pixel pozici (nebo numerickou hodnotu) PŘED
  vizuálním testem.** Vizuální test je drahý a citlivý na okolní změny.
  Pixel test (nebo unit test na konvertor) je deterministický a chytí
  90 % regresí.
- **Visual signature často určuje typ chyby.** "Posunuté do rohu" =
  origin convention bug. "Zrcadlové" = axis flip. "Posunuté konstantním
  offsetem" = padding/margin issue. "Roztažené nebo squashed" = scaling.
  "Random posun" = uninitialized memory. Naučit se číst tyto signaturní
  vzory šetří hodiny debugování.
- **Typová signatura `pub fn rect(x: f32, y: f32, w: f32, h: f32, ...)`
  může znamenat top-left NEBO center podle implementace.** Parametr
  name `x`, `y` je sémanticky nezachycený. Lepší: `cx`, `cy`,
  `top_left_x`, `top_left_y`, nebo dedicated `Center { x, y }` /
  `TopLeft { x, y }` newtype (zero-cost v Rustu).

## Závěr

```rust
// Před (přímý mapping):
fn push_rect(&mut self, rect: Rect, color: Color) {
    let shape = iskra::shapes::ShapeInstance::rect(
        rect.x, rect.y, rect.w, rect.h, color,  // ❌ x,y interpretováno jako center
    );
    self.shapes.push(shape);
}

// Po (explicit konverze):
fn push_rect(&mut self, rect: Rect, color: Color) {
    let cx = rect.x + rect.w * 0.5;
    let cy = rect.y + rect.h * 0.5;
    let shape = iskra::shapes::ShapeInstance::rect(cx, cy, rect.w, rect.h, color);
    self.shapes.push(shape);
}

#[test]
fn push_rect_converts_top_left_to_center() {
    let mut b = ScenePassBuilder::default();
    b.push_rect(Rect::new(32.0, 32.0, 100.0, 100.0), Color::WHITE);
    assert_eq!(b.shapes[0].position, Vec2::new(82.0, 82.0));
}
```

## Přenositelný pattern

Pokaždé, když napojuju nezávislé systémy přes API hranici, procházím
tímto checklistem:

1. **Najmenuj všechny konvence v každém systému.** Top-left vs. center,
   pixel vs. logical units, 0-based vs. 1-based, sRGB vs. linear,
   y-axis up vs. y-axis down, big endian vs. little endian, unix
   timestamp vs. ISO 8601, count vs. capacity. Neptám se "fungujou
   spolu" — ptám se "v jaké jednotce každý mluví".

2. **Hraniční funkce explicitně konvertuje.** Adapter, wrapper,
   adapter funkce — *na hranici*, nikde jinde. Konverze rozprostřená
   napříč codebase je recept na drift; jeden centralized adapter je
   debuggovatelný.

3. **Test na konverzi, ne na vizuální výsledek.** Test pixel pozice
   po mapping, test color values po sRGB konverzi, test timestamp
   po formatting. Rychlé, deterministické, rezistentní k regresím
   v unrelated modulech.

4. **Typový systém by měl konvenci nést, kde to jazyk umožňuje.**
   `Vec2 { x, y }` je generic. `Center(Vec2)` a `TopLeft(Vec2)` jsou
   newtypes — Rust to dělá zero-cost. Compiler chytí každý "nezamýšlený
   přechod" před tím, než to způsobí runtime bug.

5. **Visual signature → hypotéza o typu chyby.** "Rohové" = origin.
   "Zrcadlové" = axis flip. "Mírně posunuté" = padding/border. "Příliš
   světlé" = double gamma. "Roztažené" = aspect ratio. Každý signature
   má typický kořen — naučím se reflexně mapovat symptom na hypotézu.

Pattern se přenáší napříč doménami. Vlastně je *všudypřítomný* každé
rozhraní mezi softwarovými systémy je hranice mezi konvencemi — někdy
zjevná (HTTP request body formát), někdy skrytá (anchored vs. left-aligned
v PDF rendering). Disciplína v identifikaci konvenců a explicitní konverzi
na hranici je rozdíl mezi softwarem, kde se bug objeví v produkci,
a softwarem, kde compiler bug nepustí dál.
