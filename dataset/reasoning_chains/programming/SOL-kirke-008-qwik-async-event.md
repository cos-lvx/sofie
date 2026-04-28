# Reasoning chain — Qwik async event a sync vs. async timing

## Zdroj
`kirke/SOLUTIONS.md#SOL-008` — checkbox uvnitř `<Link>`. Klik na checkbox
měl togglovat výběr, ale Link triggeroval navigaci. `event.preventDefault()`
v Qwik `onClick$` handleru nezachytil event včas.

## Kontext
Kirké frontend — Qwik 1.x, multi-select v listu items. UI: každá `ItemCard`
je `<Link to="/items/{id}">` (klik na kartu = navigace na detail).
V rohu karty je `<input type="checkbox">` pro multi-select. Klik na
checkbox má **toggle-ovat výběr**, *ne* navigovat. Standardní React
intuice: `onClick={(e) => { e.stopPropagation(); toggle(); }}`. V Qwiku:
```jsx
<input
  type="checkbox"
  onClick$={(e) => { e.stopPropagation(); toggle(); }}
/>
```
**Nefunguje.** Klik na checkbox naviguje na `/items/{id}`. Click event
bublá k Linku, Link otevře navigaci, asynchronní `stopPropagation` přijde
pozdě.

## Analytický flow

1. **Pozoruji: ne-Qwik intuice nesedí.** V React/Vue/vanilla JS by
   `event.stopPropagation()` v synchronous handler functioned. V Qwiku
   ne. Něco je v Qwik event modelu jiné.

2. **Qwik specific: co dělá `$` suffix?** Qwik `$` označuje *lazy
   serializable closure*. `onClick$={...}` znamená "tato funkce se
   nezavolá synchronně při click event, ale lazy-loadne se z bundle
   a zavolá se *asynchronně*". To je core Qwik feature pro
   resumability — komponenty hydratují per-event, ne na page load.

3. **Důsledek pro event timing.** Standard JS event flow:
   ```
   capture phase → target → bubble phase
   ```
   V capture/bubble phase každý handler může volat
   `event.stopPropagation()` a tím přerušit dál bublání. Toto musí
   být *synchronní* — protože browser pokračuje další handler v
   sérii hned, jak vrátí control z předchozího.
   
   **Qwik `onClick$` handler je async.** Browser zavolá Qwik wrapper
   (sync), Qwik vrátí "OK, brzy bude resolved", pokračuje další
   handlery (Link's). Když konečně Qwik async handler doběhne a
   zavolá `event.stopPropagation()`, je už pozdě — Link's handler
   už se vykonal a navigace běží.

4. **Co tedy potřebuju: synchronous prevent default a stop propagation.**
   Qwik nabízí *HTML attribute* form: `preventdefault:click` a
   `stoppropagation:click`. Tyto se vyhodnotí browserem **synchronously**
   *před* tím, než se vůbec spustí jakýkoli `$` handler. Browser
   zaznamená "tento element má preventdefault na click", aplikuje ho,
   pak teprve dispatch handlerů.

5. **Implementation: wrapper s sync attributes + async handler.**
   ```jsx
   <span
     preventdefault:click
     stoppropagation:click
     onClick$={() => toggle()}
     class="checkbox-wrapper"
   >
     <input
       type="checkbox"
       checked={selected}
       readOnly
       tabIndex={-1}
       style={{ pointerEvents: "none" }}
     />
   </span>
   ```
   Klíčové detaily:
   - **Wrapper `<span>`** drží sync attributes a async handler.
     Browser stop propagation hned, async toggle proběhne pozdě
     ale to je OK (state update doesn't need synchronous timing).
   - **Native `<input>` má `pointer-events: none`** — sám o sobě
     neumí zachytit klik. Wrapper drží all interaction logic.
   - **`readOnly` + `tabIndex={-1}`** — input nepřijímá user input
     direct; je jen visual indicator. Wrapper handluje state.

6. **Ověř, proč input má `pointer-events: none`.** Pokud by input
   byl click-able, browser by toggle `checked` *před* tím, než async
   handler zavolá `toggle()`. Race condition: klik změní visible
   `checked` (browser default), pak async toggle změní state (kterýé
   se re-renderuje a override visible state). Output: blink, nebo
   inconsistent state. **`pointer-events: none` na input předává
   všechny clicks na wrapper, který má all the control.**

7. **Reflexe: Qwik vs. React mental model je fundamentálně jiný.**
   React: synchronous event flow, you control everything inside
   handler. Qwik: *lazy* event handlers, async by default, sync
   control musí být na HTML attribute level. Toto je konsekvence
   resumability — Qwik nepoužívá full hydration na page load, takže
   handlery jsou primary lazy. Jakýkoli *control* před dispatchem
   musí být declarative (HTML attributes), ne imperative (handler
   code).

## Aplikovatelné principy

- **Asynchronní handler nemůže synchronně řídit event flow.** Pokud
  framework dispatchuje events synchronously a tvůj handler je
  asynchronous, jakýkoli synchronní control (preventDefault,
  stopPropagation) přijde pozdě. Buď použij synchronous handler,
  nebo declarative attribute level.
- **Mental model frameworku je core; reflex z jiného nesedí.** React
  `onClick={fn}` a Qwik `onClick$={fn}` *vypadají* podobně, ale mají
  zásadně jinou execution model. Reflexivní zápis "to je jako React"
  nepomáhá; přečíst, jak konkrétní framework dispatchuje, je nutnost.
- **Wrapper je legitimní kompozice pro mixed control.** Když jeden
  element neumí všechno, co potřebuju (sync control + async logic),
  wrap ho v parent, který drží sync attributes, a ponech native
  element s `pointer-events: none`. Composition over specialization.
- **`pointer-events: none` + readOnly + tabIndex=-1 je triple guard**
  proti race conditions s native input behavior. Když přebírám control,
  musím na všech kanálech (mouse, keyboard, focus) převzít, ne jen
  na jednom.

## Závěr

```jsx
import { component$, useSignal } from "@builder.io/qwik";

export const ItemCard = component$<Props>(({ item, selected, onToggle }) => (
  <Link href={`/items/${item.id}`} class="item-card">
    <span
      preventdefault:click
      stoppropagation:click
      onClick$={() => onToggle(item.id)}
      class="checkbox-wrapper"
    >
      <input
        type="checkbox"
        checked={selected}
        readOnly
        tabIndex={-1}
        style={{ pointerEvents: "none" }}
      />
    </span>
    <h3>{item.name}</h3>
    <p>{item.description}</p>
  </Link>
));
```

`preventdefault:click` brání default link navigaci (z hierarchie),
`stoppropagation:click` zastaví bublání ke Linku. `onClick$` proběhne
async — ale state toggle nemusí být synchronní, jen UI consistency
musí být.

## Přenositelný pattern

Kdykoli pracuju s nested interactive elements ve frameworku:

1. **Identifikuj framework event model.** React: synchronous handlers.
   Qwik: async lazy handlers. Vue: synchronous handlers, async setup.
   Solid: synchronous, fine-grained reactivity. Každý má jinou
   timing semantics.
2. **Pro sync control (prevent default, stop propagation) použij
   declarative attribute level, ne imperative handler.** Browser
   atributes (preventdefault, stoppropagation) se aplikují synchronously
   před handlery; handler-level calls jsou subject to handler timing.
3. **Wrapper composition over native specialization.** Když native
   element nedrží potřebnou logic, wrap him v parent. Native element
   ztrácí pointer events, wrapper má control. Cleaner než override
   native behavior.
4. **Triple-guard against native input race.** `pointer-events: none`,
   `readOnly`, `tabIndex={-1}`. Všechny tři kanály (mouse, key, focus)
   musí být na wrapper, ne na native. Jinak race s framework state.
5. **Test on slow network / device.** Async race conditions se
   projeví ne na fast laptopu, ale na slow phone nebo s throttled
   CPU. DevTools "CPU 6× slowdown" nebo "Slow 3G" jsou must-test
   conditions.

Pattern se přenáší daleko za Qwik: React Server Components (lazy
hydration), HTMX (server-driven swaps), service worker fetch handlers
(intercept timing), iframe postMessage flow. Společný invariant:
**asynchronní logic nemůže synchronně řídit synchronně dispatched
events. Pokud chci synchronně řídit, musím to udělat declaratively
před dispatchem, ne imperativně uvnitř async handleru.** Disciplína:
přečtu framework docs o event timing, ne extrapoluju z reflexů.
