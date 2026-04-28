# Reasoning chain — MSDF font charset a tichý unicode fallback

## Zdroj
`coe/SOLUTIONS.md#SOL-007` — podtitulek "Kaiser Quarry Studios — „We Carve
Stories From Stone."" se vykreslil s otazníky místo českých uvozovek:
`?We Carve Stories From Stone?`.

## Kontext
COE používá Inter font přes MSDF (Multi-channel Signed Distance Field)
atlas. MSDF je technika, kde každý glyph je předgenerovaný do jedné
sdílené texturové mapy s SDF informací, takže shader umí kreslit ostré
hrany v libovolné velikosti bez resampling artifacts. Atlas se generuje
**offline** (`msdf-atlas-gen` tool) z TTF souboru — uživatel specifikuje
charset (které unicode codepointy se zahrnou). Atlas je teď konfigurovaný
na ASCII (32-126) + latin-extended pro českou diakritiku (`ě`, `š`, `ř`,
atd.) + některé typografické znaky (em dash `—`, en dash `–`, bullet `•`).
Český text "Kaiser Quarry Studios — „We Carve..." obsahuje **české
typografické uvozovky `„` (U+201E) a `"` (U+201D)**. Tyto v atlasu
**nejsou**.

## Analytický flow

1. **Pozoruju, co vidím.** Otazníky místo uvozovek. Konzistentně, pokaždé
   na pozici `„` a `"`. Žádný error v logu, žádný warning, žádný panic.
   Build prošel, test prošel (žádný test neoveřoval visual output),
   render prošel s pozitivním exit code.

2. **První otázka: kde se otazník vzal?** V mém kódu nikde nemám
   `?` v string literálu. Někdo ho tam vrazil. Něco v rendering pipeline
   ten `?` produkuje *namísto* znaku, který chybí. To je classic
   **fallback character** behavior — když font/atlas neumí glyph,
   nahradí ho placeholderem.

3. **Najít, kde fallback nastává.** Otevřu `iskra::text::TextLayout::
   layout_internal`. Pro každý codepoint v textu se volá lookup v
   `font.glyphs.get(&codepoint)`. Pokud glyph neexistuje, kód spadne
   na `font.glyphs.get(&'?')` jako fallback. **Takže když codepoint
   chybí v atlasu, render sloupne na otazník místo abych dostala
   error.** Visual fallback místo loud failure.

4. **Které codepointy přesně chybí?** Vypíšu si z textu unicode:
   `K(75)` `a(97)` `i(105)`... `'(34)` (ASCII quote)... a vidím
   v textu **`„` (U+201E, 8222) a `"` (U+201D, 8221)**. Otevřu
   `charset.txt` v atlas konfiguraci — 8222 ani 8221 tam nejsou.
   ASCII quote `"` (U+0022, 34) tam je. **Diagnóza confirmed.**

5. **Proč jsem si toho nevšimla dřív?** Otázka je *strukturální*,
   ne náhodná. Pipeline:
   - Build: žádná validace charset coverage. Build vidí jen Rust
     kompilaci, ne shader render output.
   - Test: žádný unit test volá `font.glyphs.contains_key(&'„')`.
     Testy jsou na logické chování (TextLayout pozice, line breaks),
     ne na charset coverage.
   - Render: chybějící glyph se nahradí otazníkem, pipeline pokračuje.
   - IDE: zvýraznění syntaxe, ne vizuální preview render output.
   **Žádný vrstva nezachytí absenci znaku v atlasu před tím, než to
   uvidím na obrazovce.** Klasický silent failure, ale v unicode space.

6. **Fix má dvě roviny: taktická a systémová.**

   **Taktická (teď):** Nahradit české typografické uvozovky obyčejnými
   ASCII `"` v `build_hello_scene`. Em dash `—` (U+2014, 8212) v atlasu
   je, ten zachovám. Přidat komentář v build_hello_scene s upozorněním
   na charset constraint pro budoucí texty. Quick fix, žádné rebuild
   atlasu.

   **Systémová (RF-037):** Rozšířit `charset.txt` o 5 codepointů
   (8216, 8217, 8220, 8221, 8222) — všechny typografické uvozovky
   včetně anglických. Regenerovat `inter_atlas.{json,png}` přes
   `msdf-atlas-gen`. Zapsáno do REQUESTED-FEATURES.md, nízká priorita
   — Phase 1.8 MVP zvládne ASCII workaround. Pro Phase 1.13 Polish
   (kvalitní typografie, lokalizace) bude potřeba.

7. **Pre-flight check pro budoucí texty.** Sepíšu pravidlo: každý nový
   text v COE projde mentální charset audit. *Nebo* — lepší — pre-flight
   test:
   ```rust
   #[test]
   fn build_hello_scene_chars_in_atlas() {
       let font = load_default_inter();
       let scene = build_hello_scene();
       for text in scene.text_glyphs() {
           for ch in text.chars() {
               assert!(font.glyphs.contains_key(&ch),
                   "Missing glyph for '{}' (U+{:04X}) in text: {}",
                   ch, ch as u32, text);
           }
       }
   }
   ```
   Test selže okamžitě, pokud někdo přidá text s ne-ASCII znakem
   nepokrytým atlasem. Pre-build hook, ne post-build oprava.

## Aplikovatelné principy

- **Visual fallback je silent failure pattern v unicode space.** Když
  chybějící znak se "odkliká" otazníkem (nebo `□`, nebo `?`, nebo
  invisible), pipeline pokračuje, log nekřičí, ale výstup je rozbitý.
  Stejná povaha jako `padding: var(--missing-token)` padá na 0, jako
  `try { ... } catch { return defaults }` v build kontextu.
- **Atlas je *fixed* charset, ne *open* charset.** Generuje se offline
  z explicitně specifikovaného listu. Pokud do textu vstupuje něco
  nového (lokalizace, user content, copy/paste z designeru), musí se
  charset rozšířit a atlas regenerovat. Není to runtime extensible.
- **"Co by mě nástroj neřekl, kdyby byl špatně" je centrální otázka.**
  Compiler nezachytil, atlas nezachytil, test nezachytil, IDE nezachytil.
  Jen vizuální kontrola na produkčním renderu. Pokud doménová pipeline
  má takový blind spot, je třeba jej zaplnit pre-flight testem.
- **Lokalizační pipeline je side-effect-rich.** Český text potřebuje
  háčky, čárky, typografické uvozovky, em dash, vlastní quote rules.
  Anglický text typografické uvozovky používá jiné (`'` `'` `"` `"`).
  Polský má `„` `"` (jiný horní). Každá lokalizace expandnoucí množinu
  znaků vyžaduje nová testace charset coverage.

## Závěr

Taktický fix:

```rust
// fn build_hello_scene
// "Kaiser Quarry Studios — „We Carve Stories From Stone."" → ASCII quotes
"Kaiser Quarry Studios — \"We Carve Stories From Stone.\""
// Em dash `—` (U+2014, 8212) v atlasu je, zachován.
```

Pre-flight test:

```rust
#[test]
fn build_hello_scene_chars_in_atlas() {
    let font = load_default_inter();
    let scene = build_hello_scene();
    for text in scene.text_glyphs() {
        for ch in text.chars() {
            assert!(
                font.glyphs.contains_key(&ch),
                "Missing glyph for '{}' (U+{:04X}) in text: {:?}",
                ch, ch as u32, text,
            );
        }
    }
}
```

RF-037 (systémové řešení) zapsán do `iskra/REQUESTED-FEATURES.md`:
rozšířit `charset.txt` o codepointy 8216, 8217, 8220, 8221, 8222
+ regenerovat `inter_atlas.{json,png}`.

## Přenositelný pattern

Kdykoli pracuju s **fixed-set resource lookup** (font glyph atlas, image
atlas, sprite sheet, lookup table, enum-to-display map, i18n keys), ptám
se:

1. **Co je *open* a co *closed* set?** Atlas je closed (generuje se
   offline). Input text je open (přijde z designerského dokumentu,
   z user input, z lokalizace). Boundary mezi open input a closed
   resource je místo, kde vzniká silent failure.

2. **Co se stane, když open input obsahuje něco mimo closed set?**
   Render fallback character (`?`, `□`)? Tichý skip? Crash? Error
   propagation? Každá z těchto cest má jiný debugging cost. Tichý
   fallback je nejhorší — neuvidíš, dokud někdo vizuálně nezkontroluje.

3. **Existuje pre-flight check, který ověří coverage předtím, než
   pipeline doběhne?** Test, ktery iteruje přes všechny použité
   znaky a porovnává s atlasem. Test, který parsuje i18n bundles
   a ověřuje, že každý klíč má zdroj. Test, který kontroluje, že
   všechny enum-to-string zápisy mají reverse mapping.

4. **Pokud closed set se rozšiřuje (atlas regeneration, table expansion,
   localization update), je to atomický deploy?** Nebo se může stát,
   že kód odkazuje na charset, který v atlasu ještě není? Pokud je
   to možné, potřebuju version-stamping a deploy ordering pravidlo.

Pattern se přenáší: i18n missing keys (returning klíč jako string nebo
prázdno), enum SerDe missing variant (panicking nebo defaulting), DB
foreign key reference to non-existing row (referential integrity nebo
silent NULL), sprite atlas missing entry (rendering invisible nebo
default placeholder), API contract field optional vs. required, dokonce
i path traversal v file system (existence check vs. graceful 404).
**Univerzální disciplína: kdykoli systém *může* tiše degradovat na
default, *musí* mít pre-flight test, který degradaci zachytí jako
chybu před tím, než ji uživatel uvidí jako otazník.**
