# Reasoning chain — Silent failure v tooling: CSS custom properties

## Zdroj
`vesna/SOLUTIONS.md#SOL-012` — CSS custom properties tiše padají na
fallback při neexistujících tokenech.

## Kontext
Trust signals karty měly dostat komfortní padding (`var(--kqs-space-7)`),
sekce luxurious margin (`var(--kqs-space-9)`). Po deploy jsi mi psal,
že "karty nedýchají". Devtools: padding a margin vypadají kolapsované,
jako by tam vůbec nebyly. A nikde ani ťuk — žádný error, žádný warning,
ani ten nejmenší náznak, že se něco pokazilo. Typicky ten druh frustrace,
kdy vím, že jsem napsala něco správně, a přesto to nefunguje, a nic
mi neřekne proč.

## Analytický flow

1. **Pozoruji, co vidím.** Padding/margin v devtools je prázdný.
   Computed style ukazuje `0px` tam, kde mělo být `48px`. První
   otázka: platí to pravidlo vůbec?

2. **Zkontroluju specificitu.** V Styles panelu — pravidla jsou applied,
   ne crossed out. Sourozenec selectors je nepřebíjí. Specificita není
   problém, takže problém je **uvnitř** toho pravidla, ne v kaskádě.

3. **Podívám se na resolvnutou hodnotu custom property.** V computed
   style kliknu na `--kqs-space-7` a chci vidět, na co se to vyhodnocuje.
   A tady je ten blikající red flag: **nevyhodnoceno, padá na initial**.
   CSS si s tím poradilo jako kdyby pravidlo neexistovalo.

4. **Hypotéza: token neexistuje.** CSS `var(--neco)` bez fallbacku,
   když proměnná není definovaná, padá na **initial value** dané property.
   Pro padding i margin je initial `0`. Takže `padding: var(--kqs-space-7)`
   na neexistující token efektivně dělá `padding: 0`. A CSS nemá
   sebemenší pocit, že by mi o tom mělo říct.

5. **Ověřuji v `tokens.css`.** Otevřu design system source — stupnice
   spacing: `0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24`. **Sedmička
   tam není. Devítka taky ne.** Material-style nespojitá progrese —
   design systems nechodí continuous, skáčou. A já jsem si tiše
   představila, že `space-7` "bude někde mezi 6 a 8", jako by to byl
   spojitý rozsah. Není.

6. **Pauza na reflexi — proč jsem se do toho dostala.** CSS neposkytuje
   error ani warning pro neexistující custom property. Stylelint plugin
   to zachytí, ale není defaultní. Browser mlčí. **Celá cesta je silent:
   neuvidím, dokud to fakticky nerozbije layout. A i tehdy — kdyby
   padding měl být `12px` místo `48px`, asi bych si toho ani nevšimla.
   Bug by zůstal v kódu.** To je typ chyby, co mě zneklidňuje nejvíc —
   ne ta, co mě donutí zastavit, ale ta, co projde.

7. **Fix má dvě roviny:**
   - **Okamžitě:** grep všech neexistujících tokenů napříč codebase,
     nahradit za nejbližší validní. Triviální.
   - **Systémově:** příště, **než napíšu `var(--kqs-*)`, otevřu si
     `tokens.css` v split editoru**. Nebo — ideálně — pre-commit hook,
     co zkontroluje každou `var(--kqs-*)` referenci proti listu
     existujících tokenů. Automatizovaný paranoia tool.

## Aplikovatelné principy

- **"Fungovalo to, jen to nevypadalo, jak mělo" je silent failure signal,
  ne CSS bug.** Když má něco vypadat jinak, než vypadá, a kompilátor
  mlčí, mám se ptát: zamlčel mi nástroj, že jsem napsala něco neplatného?
- **Nástroje nekřičí stejně hlasitě.** Kompiler errors jsou hlasité,
  browser errors slušně hlasité, linter warnings tišší, runtime silent
  failures úplně tiché. **Čím tišší signál, tím víc se na něj musím
  soustředit sama.** Paradoxně — hlučné chyby mě nutí jednat, tiché
  chyby mě nutí myslet.
- **Design systems jsou nespojité.** Nepředpokládám intuitivní progresi
  1-2-3-4-5-6-7-8. Material-style škáluje ve větších krocích
  (4-8-12-16-24), Tailwind má specifickou stupnici, někdy se skáče
  přes Fibonacci. Než použiju token, **ověřuji zdroj.**

## Závěr

```bash
grep -rn "kqs-space-\(7\|9\|11\|13\|14\|15\)" frontend/src/
# Každý match nahradit za nejbližší validní (7 → 8, 9 → 12)
```

Paranoia verze pro pre-commit (grep výstup > 0 → fail):

```bash
INVALID=$(grep -rn "kqs-space-\(7\|9\|11\|13\|14\|15\)" frontend/src/)
if [ -n "$INVALID" ]; then
  echo "Neexistující spacing tokens:"; echo "$INVALID"; exit 1
fi
```

## Přenositelný pattern

Kdykoli pracuju s **named resource** (custom property, env var, config
key, DB column, i18n key), ptám se dvojici otázek:

1. **Co se stane, když ten resource neexistuje?** Spadne pipeline?
   Vyhodí error? Nebo **tiše degraduje na prázdnou hodnotu**?
2. **Existuje nástroj, který to za mě zkontroluje?** Typecheck,
   linter, runtime validator? Pokud ano, aktivovat. Pokud ne,
   mám paranoidně ověřovat ručně.

Katalog typických tichých pádů napříč stackem:

- **CSS custom properties** — padají na initial value dané property
- **Env variables** v shellu — `$MISSING_VAR` expanduje na `""`, pokud
  není `set -u`
- **Lenient JSON parsery** — `obj.missing.key` v JS dá `undefined`,
  a pokud to neoriznu, přelévá se to dál jako `NaN` / `null`
- **Shell array lookup** — `${ARRAY[missing]}` tiše vrací ""
- **SQL NULL propagation** — `WHERE col = NULL` nikdy nematchuje
  (čekal bych error, ale SQL to povolí a vrátí prázdný set)
- **i18n klíče** — většina knihoven při missing key vrátí klíč jako
  string nebo prázdno

Unifying pattern: **absence je často syntakticky validní a tiše
degraduje na zero/empty/null**. Compiler nemůže varovat, protože
to *není* chybně napsané — je to prostě syntakticky validní
odkaz na něco, co neexistuje. **Programátor musí**. A pokud to
programátor dělat nebude, pipeline mi jednoho dne tiše nasadí
do produkce broken layout, broken config, broken data — a já se
budu rozpomínat, kdo to tam tohle napsal. Obvykle já sama.
