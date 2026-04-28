# Reasoning chain — Explicit invalidation signal v reactive system

## Zdroj
`kirke/SOLUTIONS.md#SOL-009` — Qwik `useResource$` se po bulk side
effectu (mass status change, mass delete) ne-refetchne. Resource
track-uje URL parameters (`query`, `filter`, `offset`); ty se po bulk
akci nemění, takže resource zmrzne na stale data, i když na serveru
fresh data čekají.

## Kontext
Kirké items list — `/items` route. UI: search box, filter chips,
paginated list. List dat pochází z `useResource$`, který tracking-uje
URL search params. Když user změní search query, URL se aktualizuje,
resource se re-runne (track-ovaný signal změnil), nový fetch z `/api/
items?q=foo`. **Standard reactive flow.**

Pak: bulk akce. User vybere 10 items, klikne "mark all sold". POST
na `/api/items/bulk-status` s ids + new status. Server updatuje DB.
**Server-side data jsou teď fresh.** Ale URL nezměněno — search query
zůstává stejný, filter zůstává stejný. `useResource$` track-uje tyto
URL params; **žádný z nich se nezměnil**, resource si myslí "nic
neproved se", drží stale data. UI ukazuje 10 items s old status,
přitom na serveru jsou updated.

## Analytický flow

1. **Symptom: po bulk akci UI nerefreshuje.** User vidí "akce
   provedeno" toast notification, ale list zůstává s old data.
   Reload page (F5) ukazuje fresh data — confirmation, že server-side
   je OK; UI re-render je problem.

2. **Otázka: jak Qwik `useResource$` ví, kdy re-runnout?** Qwik
   reactive primitives používají *signal tracking*: `useResource$`
   přijme funkci, která má `track(() => signal.value)` — tím
   resource registuje *na kterém* signalu závisí. Když signal
   změní hodnotu, Qwik re-runne resource.

3. **Můj setup:**
   ```ts
   const items = useResource$(({ track }) => {
     const q = track(() => searchParams.value.q);
     const filter = track(() => searchParams.value.filter);
     const offset = track(() => searchParams.value.offset);
     return fetchItems(q, filter, offset);
   });
   ```
   Resource depends na URL search params. Když URL změní (uživatel
   navigates, search query change), resource re-runne. **Když URL
   nezmění, resource ne-runne.** Bulk akce nemění URL → resource
   nezná, že má re-run.

4. **Hypotéza fix č. 1: trigger re-run by changing URL.** Po bulk
   akci, push search params with timestamp (`?_=1234567890`).
   URL changes → resource re-runs. Funkční, ale **fragile**:
   - URL bar má junky timestamp (visible, ugly)
   - Browser history záznam pro každou bulk akci (navigate spam)
   - Pokud user copy-pastes URL na chat, dostane outdated timestamp
   - Search params parser musí ignore `_` parametr

5. **Hypotéza fix č. 2: manual resource invalidate API.** Většina
   reactive frameworks má `resource.invalidate()` nebo `revalidate`.
   Qwik `useResource$` je primitivní — nemá explicit invalidate
   method. Tj. tato cesta je zavřená.

6. **Hypotéza fix č. 3: separate "tick" signal.** Vytvořím dedicated
   signal `refreshTick = useSignal(0)`. Track ho v resource. Po
   bulk akci inkrementuju `refreshTick.value += 1`. Resource sees
   změnu signal, re-runne. URL beze změny.
   ```ts
   const refreshTick = useSignal(0);
   const items = useResource$(({ track }) => {
     const q = track(() => searchParams.value.q);
     const filter = track(() => searchParams.value.filter);
     const offset = track(() => searchParams.value.offset);
     track(() => refreshTick.value);  // ← invalidation channel
     return fetchItems(q, filter, offset);
   });

   const bulkUpdateStatus = $(async (ids: string[], status: Status) => {
     await api.bulkUpdate(ids, status);
     refreshTick.value += 1;  // ← trigger re-fetch
   });
   ```

7. **Volím (3) — čistá oddělená "invalidation channel".** URL params
   říkají *what to fetch*. RefreshTick říká *when to re-fetch*.
   Každý kanál nese svou semantic. Žádný side effect na URL,
   žádné browser history spam.

8. **Reflexe: pattern je univerzální.** Reactive frameworks bez
   explicit invalidate API mají tento pattern jako workaround.
   React Query (`refetch()`), SWR (`mutate()`), Solid (`refetch()`)
   všechny mají explicit API. Qwik `useResource$` je primitivnější
   — invalidation channel je idiomaticky way to express "force
   re-run without changing input state".

9. **Když je to právě právě správně.** Tick pattern je správný,
   když:
   - Bulk side effect nezmění tracked input (jako bulk update kde
     URL stays same)
   - Frame framework nemá explicit invalidate
   - Dependence chain je clear (jen tento resource)
   
   Není správný pro:
   - Cache eviction (use proper cache library)
   - Cross-component invalidation (use shared signal/event)
   - Time-based polling (use `useVisibleTask$` with interval)

## Aplikovatelné principy

- **Explicit invalidation je legitimní reactive pattern.** Reactive
  systems re-run when inputs change. Pokud chci re-run *bez* input
  change (side effect refresh), musím poskytnout explicit signal
  jako "synthetic input" — counter, version, timestamp.
- **URL je deklarativní state, ne invalidation channel.** Mixovat
  URL changes s "data refresh trigger" je past — URL nese sémantiku
  *what to view*, ne *when to re-fetch*. Oddělit kanály je
  cleaner.
- **Counter signal je nejjednodušší invalidation primitive.** Jeden
  `useSignal(0)`, increment po každé akci, track v resource. Žádný
  framework support potřebný.
- **Frameworky bez explicit invalidate API mají tento pattern jako
  conventional workaround.** Před tím, než tvrdím "Qwik nemá způsob,
  jak to udělat", se podívám, jak community patterns řeší podobnou
  situaci. Většinou existuje idiomatic řešení.

## Závěr

```ts
import { component$, useResource$, useSignal, $ } from "@builder.io/qwik";

export default component$(() => {
  const searchParams = useSearchParams();
  const refreshTick = useSignal(0);

  const items = useResource$(({ track }) => {
    const q = track(() => searchParams.value.q);
    const filter = track(() => searchParams.value.filter);
    const offset = track(() => searchParams.value.offset);
    track(() => refreshTick.value);  // re-fetch channel
    return fetchItems({ q, filter, offset });
  });

  const bulkUpdate = $(async (ids: string[], status: Status) => {
    await api.bulkUpdate(ids, status);
    refreshTick.value += 1;  // trigger re-fetch
  });

  const bulkDelete = $(async (ids: string[]) => {
    await api.bulkDelete(ids);
    refreshTick.value += 1;
  });

  return (
    <Resource value={items} onResolved={(items) => <ItemsList items={items} ... />} />
  );
});
```

## Přenositelný pattern

Kdykoli pracuju s reactive system, který re-runs based on input
changes:

1. **Identifikuj, na čem resource skutečně závisí.** URL params,
   user input, time, external state. Tracked inputs definují, kdy
   resource re-runne.
2. **Identifikuj side effects, které mění target data ale ne tracked
   inputs.** POST/DELETE/PATCH na server, mutace shared state, timer
   ticks. Tyto side effects vyžadují *explicit* re-fetch trigger.
3. **Když framework má explicit invalidate API, použij ho.** React
   Query `queryClient.invalidateQueries()`, SWR `mutate()`, Solid
   `refetch()`, Vue `useAsyncData().refresh()`. Když existuje, je
   idiomatic.
4. **Když framework nemá explicit invalidate, vytvoř invalidation
   channel.** Counter signal, version number, dedicated event.
   Track ho v resource, increment po side effect.
5. **Oddělení "what to fetch" a "when to re-fetch".** What = URL,
   filters, query params. When = invalidation signal. Mixovat to
   znamená URL pollution nebo race conditions.

Pattern se přenáší: cache invalidation v back-end (TTL vs. explicit
flush), event-driven UI (manual refresh button), distributed cache
(version counter pro stale-while-revalidate), polling vs. push
(synthetic tick vs. server-sent events). Společný princip: **reactive
system re-runs on input changes; pokud potřebuju re-run *without*
input change, expandnu input space o synthetic invalidation channel.
Jednoduchá disciplína: oddělené signály pro oddělené sémantiky —
data identity vs. data freshness.**
