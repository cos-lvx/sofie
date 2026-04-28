# Reasoning chain — CSS `:first-child` ignoruje text nodes

## Zdroj
`vesna/SOLUTIONS.md#SOL-018` — selektor `.blog-article__body p em:first-child`
měl cílit jen na úvodní kurzívní podtitul (jeden specifický element);
místo toho stylioval `<em>` uvnitř kterých dalších odstavců, kde byla
inline kurzíva. Tři blog články visuálně rozbité.

## Kontext
Vesna blog má markdown rendering. Article body má strukturu:
```html
<article class="blog-article__body">
  <p><em>Tagline / kurzívní podtitul pod nadpisem.</em></p>
  <hr>
  <h2>...</h2>
  <p>Lorem ipsum <em>important word</em> dolor sit amet.</p>
  <p>Druhý odstavec s <em>další zdůrazněnou frází</em>.</p>
</article>
```
Cíl CSS: úvodní `<em>` v prvním `<p>` (tagline) má `display: block;
font-size: 1.2rem; color: var(--secondary)`. Inline `<em>` v body
odstavcích zůstává inline, default style. Selector který napsal:
```css
.blog-article__body p em:first-child {
  display: block;
  font-size: 1.2rem;
  color: var(--secondary);
}
```
Intuice: "em is first-child of p" = "em that is the entire content of p".
**Nesedí.** Selector matchuje *všechny* `<em>` které jsou first
elementový child of any `<p>`. Body odstavce začínající `Lorem ipsum
<em>important word</em>` taky matchují (em je první element child;
text "Lorem ipsum " je text node, ne element child).

## Analytický flow

1. **Pozoruji: tři blog články mají broken styling.** Inline kurzíva
   v body odstavcích vypadá jako tagline (block, large, secondary
   color). Tagline na úvodu funguje OK. Něco s selectorem matchuje
   víc, než zamýšleno.

2. **Otázka: co přesně `:first-child` matchuje?** Otevřu MDN docs.
   `:first-child` matchuje element, který je **first child element
   of its parent**. Klíčové: *element child*. Text nodes (whitespace,
   plain text) se ignorují pro tento selector. `<p>Lorem <em>x</em></p>`:
   první child node je text "Lorem ", ale první *element* child je
   `<em>`. Takže `<em>` *je* `:first-child` of `<p>`.

3. **Diagnóza: selector říká něco jiného, než bylo zamýšleno.**
   Zamýšleno: "em který je celým obsahem p" — i.e., text content
   p je jen em, nic dalšího. Skutečně řekl: "em který je první
   element child of p". Tyto dvě jsou v `<p>Lorem <em>x</em>` *jiné*
   — text "Lorem " je tam, ale `:first-child` to ignoruje.

4. **Kde fix patří?** Tři možnosti:
   - **(a) Zkusit complex CSS selector, který lépe vyjadří záměr.**
     CSS bohužel nemá selector pro "this element is the entire content
     of parent". `:only-child` matchuje element, který je *jediné*
     element child — ale stejně ignoruje text nodes, takže
     `<p>Lorem <em>x</em></p>` má em jako jediný element child
     (= matches `:only-child`). Stejný problém.
   - **(b) Přidat třídu na úrovni rendereru.** Custom marked renderer
     (markdown → HTML) zachytí "this paragraph contains only em",
     přidá class `subtitle`. CSS pak styluje `.subtitle` — sémantická
     třída, ne strukturální guess.
   - **(c) JS post-processing po render.** Iterate paragraphs,
     check if children = single em, addClass. Funkční, ale runtime
     cost a fragile.

5. **Volím (b) — renderer.** Markdown to HTML conversion běží jednou,
   ne na frame loop. Custom renderer v `marked` v15:
   ```ts
   new Marked({
     renderer: {
       paragraph(token) {
         const inner = this.parser.parseInline(token.tokens);
         const onlyEm =
           token.tokens.length === 1 && token.tokens[0].type === "em";
         return onlyEm
           ? `<p class="blog-article__subtitle">${inner}</p>\n`
           : `<p>${inner}</p>\n`;
       },
     },
   });
   ```
   `token.tokens` je array of inline tokens v paragraph. Pokud array
   má length 1 a single token je `em`, paragraph je tagline. Add
   class.

6. **CSS becomes simple, semantic.**
   ```css
   .blog-article__subtitle em {
     display: block;
     font-size: 1.2rem;
     color: var(--secondary);
   }
   /* Body p em zůstává default — žádný override */
   ```
   No structural selector trickery. Class řekne sémantiku, CSS
   reaguje.

7. **Reflexe: CSS je o vzhledu, HTML o sémantice.** "Toto je tagline"
   je sémantická informace. Vyjádřit ji jen strukturálním CSS selectorem
   je fragile — strukturální shape (em as first-child of p) může
   shodovat s úplně jiným záměrem (inline em). Sémantika patří do
   HTML (class, role, semantic tag), ne do CSS.

## Aplikovatelné principy

- **CSS pseudo-class `:first-child` (a `:nth-child`, `:only-child`)
  ignoruje text nodes.** Říká "first/nth/only **element** child", ne
  "first/nth/only node". Pro element-only DOM je to OK; pro mixed
  content (text + element) může matchovat víc, než zamýšleno.
- **Strukturální selector pro sémantický záměr je fragile.** "Em is
  whole content of p" je sémantika. Strukturální guess (`:first-child`,
  `:only-child`) na to neodpovídá robustně. Lepší: sémantická třída.
- **Markdown renderer customization je často underused.** Marked,
  remark, MDX všechny umožňují custom renderer. Pokud potřebuju
  semantic decisions během markdown → HTML, renderer je right place
  — má access k AST, může add classes, transform output.
- **HTML nese sémantiku, CSS nese vzhled.** Když chci říct "tato část
  je jiná v meaning", patří to do HTML (`<aside>`, `<figcaption>`,
  class, ARIA). Když chci říct "vypadá to jinak", CSS. Mix těch dvou
  (sémantika přes pseudo-class) je past.

## Závěr

```ts
// frontend/src/lib/markdown.ts
import { Marked } from "marked";

export const markdownRenderer = new Marked({
  renderer: {
    paragraph(token) {
      const inner = this.parser.parseInline(token.tokens);
      const onlyEm =
        token.tokens.length === 1 && token.tokens[0].type === "em";
      return onlyEm
        ? `<p class="blog-article__subtitle">${inner}</p>\n`
        : `<p>${inner}</p>\n`;
    },
  },
});
```

```css
/* CSS pak je sémantická */
.blog-article__subtitle em {
  display: block;
  font-size: 1.2rem;
  color: var(--secondary);
  font-style: italic;
}
/* p em (inline) zůstává default */
```

## Přenositelný pattern

Kdykoli píšu CSS selector pro "this specific occurrence":

1. **Ptej se, *co* selector skutečně matchuje, ne *co bys chtěl*.**
   `:first-child` matchuje first element child (ignoring text). `>`
   matchuje direct child only. `*` matchuje any descendant.
   Konkrétně, ne intuitivně.
2. **Test selector v devtools.** Highlight matching elements,
   ověř že počet sedí. Pokud matchuje víc, než čekal jsem, selector
   je broader than intended.
3. **Pro sémantický záměr použij sémantickou třídu, ne strukturální
   selector.** "Toto je nadpis" → `class="heading"` + CSS. "Toto je
   tagline" → renderer adds class + CSS. Místo CSS guesswork.
4. **Custom renderer / preprocessor je legitimní layer.** Markdown
   render, JSX transform, template engine — všechny jsou place,
   kde můžeš přidat semantic information bez change source content.
5. **Test acceptance criteria: blog article s mixed content.**
   Markdown s tagline + body s inline em. Render. Ověř, že tagline
   styluje, body ne. To je test, který by chytil tento bug.

Pattern se přenáší: XPath selectors (mají různé node selection
semantics), jQuery `.first()` vs. `:first-child`, query selectors
v testech (CSS-based vs. role-based selectors v Testing Library),
templating systems s conditional rendering. Společný princip: **CSS
selektory mluví strukturálním jazykem (parent-child, siblings, position),
ne sémantickým (this-is-the-tagline). Pokud chci sémantiku, musím ji
*explicitně* vyjádřit (class, ARIA role, semantic tag), ne implicitně
hádat ze structure. CSS stylo, HTML sémantizuj.**
