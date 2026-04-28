# Reasoning chain — Graceful degradation maskuje chybu v SSG pipeline

## Zdroj
`vesna/SOLUTIONS.md#SOL-016` — Silent failure v `vesna stage` při API 5xx.

## Kontext
Staging deploy "uspěl". Zelený souhrn v CLI, health OK, nikde ani ťuk.
Jenže `/blog/` byl prázdný a subpages chyběly úplně. Klasický "všechno
svítí zeleně, nic nefunguje" moment — ten, co mě vždycky donutí
zabrzdit a zeptat se, komu vlastně věřím.

## Analytický flow

1. **Zelená není správně — je jen "nic explozivního nevypadlo".** Tohle je
   moje první pravidlo u deploy pipeline: CLI summary říká, že exit code
   není nenulový. Neříká ani slovo o tom, jestli output má obsah. Když na
   to zapomenu, nechám se ukolébat.

2. **Co by mělo být vidět, kdyby to opravdu fungovalo?** Pro SSG je
   nejhezčí observable metrika **počet pages**. Lokálně mi build vyrobil
   32 stránek. Staging ukazuje 29. Tři chybí. To je ten první kousek
   signálu, co se dá zachytit — ne kouř, jen průvan.

3. **Ověř si to na filesystemu.** SSH na server, `ls /opt/vesna/dist/blog/`.
   Každý publikovaný slug má mít svůj subdir. Chybí jich pár. Teď vím
   jistě, že build ty stránky neprodukoval — není to cache, není to
   routing, fyzicky tam nejsou.

4. **Zpětná cesta pipeline.** Odkud se rozhoduje, které slugy se
   prerenderují? Qwik SSG volá `onStaticGenerate` na `/blog/[slug]/`,
   ta vrací `{ params: [] }` podle toho, co dostane z `listBlogPosts()`.
   Takže někde mezi "backend má data" a "SSG dostane seznam" se stala
   tichá katastrofa.

5. **Ta tichá katastrofa bývá v `try/catch`.** V loaderu:
   ```ts
   try { await listBlogPosts(...) }
   catch { return { items: [], error: true } }
   ```
   Tohle je typická graceful degradation pro runtime — když backend
   padne, uživatel uvidí "0 článků, zkuste později" místo bílé obrazovky.
   Chytré pro produkci, **sebevražedné pro build**. Build dostane prázdný
   seznam, postaví 0 stránek, a pipeline svítí zelená.

6. **Proč vlastně backend padl?** Trace dál — 500, permission denied na
   `blog_posts`. To je KI-015 z druhé strany: tabulku vytvořil
   `postgres` superuser při migraci, `vesna` user nemá SELECT. Dvě vrstvy
   silent failures se poskládaly jedna na druhou a dohromady udělaly
   zelený deploy prázdného blogu. To je ten typ chyby, co mi vždycky
   připomene, proč věřit jenom výstupům, ne CLI summary.

7. **Fix musí být dvoupatrový:**
   - **Okamžitý:** backend permissions (ALTER OWNER na `vesna`).
   - **Systémový:** v SSG kontextu **nechci graceful degradation** na
     kritický content. `if (import.meta.env.SSR) throw;` — v runtime
     normální fallback, v buildu fail-fast. Protože v buildu chci,
     aby rozbitá data rozbila pipeline. To je bezpečnost, ne krutost.

## Aplikovatelné principy

- **Graceful degradation je správná pro runtime, nebezpečná pro pipeline.**
  Build a deploy mají jiný smluvní vztah s chybou než user-facing kód.
  Build chce vidět pravdu hned, runtime má být shovívavý. Stejný
  `try/catch` se v obou kontextech chová jinak špatně.
- **Observable metriky mimo exit code.** Počet pages, řádků v DB,
  odpovědí z API, bytes v archivu. Vždycky si před deploy řeknu, co je
  **produkt** pipeline a kolik ho má být — jinak se dívám na špatné
  signály.
- **Silent failures se řetězí.** Jedna vrstva tiše zdegraduje, druhá
  vrstva tiše pokračuje, třetí vrstva tiše produkuje broken output.
  Nikdo nekřičí, a přesto je všechno rozbité. Trochu jako tichá pošta,
  jen horší — nakonec je to vždycky moje odpovědnost zjistit, kde
  řetěz praskl.

## Závěr

Okamžitý fix je z SOL-015 (ALTER OWNER po migraci). Systémová prevence:

```ts
// frontend/src/routes/blog/index/index.tsx
export const useBlogPosts = routeLoader$(async () => {
  try {
    return await listBlogPosts();
  } catch (err) {
    if (import.meta.env.SSR) {
      // Build-time: chci broken deploy vidět, ne zamaskovat
      throw new Error(`SSG: listBlogPosts failed: ${err}`);
    }
    // Runtime: shovívavý fallback, uživatel nemusí vidět stack trace
    return { items: [], error: true };
  }
});
```

Plus post-deploy sanity check:

```bash
EXPECTED=32  # z posledního lokálního buildu
ACTUAL=$(ssh vesna-droplet "ls /opt/vesna/dist/blog/*/ | wc -l")
[ "$EXPECTED" = "$ACTUAL" ] || echo "WARNING: expected $EXPECTED pages, got $ACTUAL"
```

Pár řádků, co mi řeknou, že deploy **skutečně** dodal, co měl.

## Přenositelný pattern

V každé pipeline si dávám tři otázky, **než** ji spustím:

1. **Co je produkt?** Files, rows, API responses, artifacts. Něco,
   co fyzicky existuje po úspěchu a co umím spočítat.
2. **Kolik jich má být?** Baseline z předchozího běhu, z lokálního
   buildu, z business logiky. Vědět, co je "moc málo".
3. **Kde to ověřím nezávisle na CLI?** Filesystem, DB query, HTTP
   probe. Ne summary, co mi pipeline vypíše sama — ta mi řekne to,
   co chci slyšet.

A čtvrtá, metaotázka, co si kladu u cizího kódu: **je někde v pipeline
`catch` bez `throw`, co tiše sežere chybu?** To je ten nejčastější
zdroj. Týká se to napříč stackem — Webpack/Vite build, deploy scripty,
data ETL, backup systémy, CI/CD gate check. Všude, kde máš řetěz kroků
a spoléháš na to, že předchozí krok "ti řekne", pokud něco selhalo.

Generalized shrnutí: **tichý poloúspěch v řetězovém systému je horší
než hlasitý pád.** Pád zastaví všechno a donutí mě se tím zabývat.
Poloúspěch jde dál a produkuje rozbitý výstup, který pak někdo jiný
(nebo já zítra) považuje za správný — a staví na něm. To je ta pravá
škoda.
