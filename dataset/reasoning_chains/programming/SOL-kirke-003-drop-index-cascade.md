# Reasoning chain — DROP INDEX cascade a kogentní dependencies v PG schemě

## Zdroj
`kirke/SOLUTIONS.md#SOL-003` — migrace v0.1.4 měla `ALTER TABLE items
DROP COLUMN brand` následovaný `DROP INDEX idx_items_search`. Migrace
selhala s `index "idx_items_search" does not exist`.

## Kontext
Kirké items table má FTS index `idx_items_search` postavený přes
`to_tsvector('czech', name || ' ' || description || ' ' || COALESCE(brand, ''))`.
Migrace v0.1.4 odstraňuje `brand` sloupec (refactor — značka je teď
samostatná entita). Migrace skript:
```sql
ALTER TABLE items DROP COLUMN brand;
DROP INDEX idx_items_search;
CREATE INDEX idx_items_search ON items USING GIN (to_tsvector('czech', name || ' ' || description));
```
Logická sekvence: smaž sloupec, smaž starý index (který závisel na
sloupci), vytvoř nový index bez sloupce. **Selže na druhém řádku:
`index "idx_items_search" does not exist`.**

## Analytický flow

1. **Symptom: index, který by měl existovat, neexistuje, když se ho
   pokusím dropnout.** Něco ho musí smazat *před* explicit `DROP
   INDEX`. Mezi řádkem 1 (`ALTER TABLE DROP COLUMN`) a řádkem 2
   (`DROP INDEX`) se index ztrácí. Otázka: kdo ho smazal?

2. **Hypotéza: PG cascade dependency.** PG má concept dependency
   tracking. Když objekt A závisí na B (např. index na column,
   constraint na column, view na table), drop B automaticky drops
   A. Tohle je `RESTRICT` vs. `CASCADE` semantika. Pro kolony default
   je `CASCADE` — drop column auto-drops dependent objects.

3. **Verifikace: pg_depend system catalog.** Před migrací (na DEV
   DB):
   ```sql
   SELECT classid::regclass, objid::regclass, refclassid::regclass, refobjid::regclass
   FROM pg_depend
   WHERE refobjid = 'items'::regclass;
   ```
   Output ukazuje, že `idx_items_search` má dependency na `items`
   columns including `brand`. **PG ví, že index potřebuje brand.
   Když drop brand, PG cascade-drops index.** Confirmed.

4. **Důsledek: explicit `DROP INDEX` po `DROP COLUMN` je redundantní
   a chybný.** PG už index dropnula sama. Můj `DROP INDEX` se zeptá
   na neexistující objekt → fail.

5. **Fix: odebrat explicit `DROP INDEX`.**
   ```sql
   ALTER TABLE items DROP COLUMN brand;
   -- PG automaticky dropne idx_items_search (cascade)
   CREATE INDEX idx_items_search ON items USING GIN (to_tsvector('czech', name || ' ' || description));
   ```
   Migrace projde. Index je rebuilt s novou definicí.

6. **Otázka: kdy *je* `DROP INDEX` nutný?** Kdy index nezávisí na
   columnu, který se dropá. Příklady:
   - Index na column, který *zůstává*. Pokud měním column type
     (`ALTER COLUMN ... TYPE`), index může zůstat nebo musí být
     rebuilt manually.
   - Index na multi-column kombinaci, kde dropuju jen jednu column.
     PG dropne index automaticky (závisí na všech columns), ale
     můžu chtít explicit drop pro clarity.
   - Index, který je *outdated* (neodpovídá nový query pattern),
     ale dropá se *bez* drop column.

7. **Reflexe na dependency awareness.** PG dependency tracking je
   *feature*, ne bug. Pokud bych explicit `DROP INDEX` *neodebrala*,
   PG by ho nedropla cascade-em (column dependency je primary), můj
   `DROP INDEX` by selhal. Buď respektuju PG cascade, nebo to
   override-uju s explicit ordering. Mid-way (pokus o duplicitní drop)
   je chyba.

8. **Generalize: cascade je rule.** Stejný princip pro:
   - `DROP CONSTRAINT` po `DROP COLUMN` — constraint na column
     cascade-dropuje
   - `DROP TRIGGER` po `DROP TABLE` — trigger cascade-dropuje
   - `DROP VIEW` po `DROP TABLE` — view depending cascade-dropuje
     (pokud table je ALL view source) nebo restrict-fails (pokud
     view depends ale je SELECTABLE without table)
   **Vždy si projdu `pg_depend` před tím, než píšu `DROP` statement,
   pokud je v migraci řetězec ALTER + DROP.**

## Aplikovatelné principy

- **PG cascade dependency je feature první, ne lapání chyba.** Když
  drop column kaskáduje na drop indexu, PG dělá svoji práci — udržuje
  schema consistent. Migrace by měla *spolupracovat* s cascade, ne
  duplicitně volat operace.
- **Migration script by měl obsahovat *jen* operace, které PG sám
  neudělá.** Pokud cascade dělá X, nedělej X explicitly. Pokud cascade
  nedělá Y, musíš Y explicitly. Test: spustit migraci na DEV/staging,
  ověřit, že schema je co chci, *před* aplikací na produkci.
- **`CREATE OR REPLACE` po cascade je správný recreation pattern.**
  Drop column → cascade dropne index → manually recreate index s
  new definition. Tříkrokový pattern: implicit drop, explicit
  recreate.
- **`pg_depend` je můj přítel pro pre-migration validation.** Před
  napsáním DROP statement, query pg_depend pro target objekt. Zjistím,
  co cascade-dropne, takže ne píšem redundantní DROP volání.

## Závěr

```sql
-- Migration v0.1.4 — drop brand, recreate FTS index
BEGIN;

ALTER TABLE items DROP COLUMN brand;
-- PG cascade automaticky dropne idx_items_search
-- (depends on brand column)

CREATE INDEX idx_items_search ON items
USING GIN (to_tsvector('czech', name || ' ' || description));

COMMIT;
```

Pre-migration validation:

```sql
-- Před napsáním migrace, ověř dependencies:
SELECT
    c.relname AS index_name,
    a.attname AS column_name
FROM pg_index i
JOIN pg_class c ON c.oid = i.indexrelid
JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
WHERE i.indrelid = 'items'::regclass;
```

## Přenositelný pattern

Kdykoli píšu DB migrace, která zahrnují DROP nebo ALTER:

1. **Identifikuj všechny závislé objekty.** Pro tabulku: indexes,
   constraints, triggers, views, foreign keys, sequences. Pro column:
   indexes, constraints, foreign keys odkazující na column. PG má
   `pg_depend` system catalog; mysql má `information_schema.
   key_column_usage` etc.
2. **Předpokládej cascade jako default behavior.** PG `DROP COLUMN`
   default-cascadekuje na dependent indexes/constraints. Mysql nějaké
   operace kaskádují, jiné fail-with-error. SQLite restricting.
   Specifika per-DB.
3. **Test migrace na DEV/staging před produkcí.** Migrace, která
   selže uprostřed (jako tento `DROP INDEX` po cascade), zanechá
   schema v inconsistent state — half-applied. Recovery vyžaduje
   manual repair. Lepší: catch on staging.
4. **Migration script má být minimal.** Každá řádka má smysl. Pokud
   řádka říká "do X", *musí* X potřebovat udělat explicitly. Implicit
   side-effects (cascade) jsou *neviditelná* část migrace, ale jsou
   skutečné.
5. **Recreation po implicit drop = `CREATE` v migraci.** Pokud chci,
   aby se po drop column re-created index s new definition, explicit
   `CREATE INDEX` v migrace. PG implicit drop neproduce automatic
   recreation.

Pattern se přenáší: filesystem operations (rmdir cascading vs. explicit
file removal), package management (uninstall package: cascading
removes its config files? data files? depending packages?), distributed
systems (cascading deletes in service A trigger updates in service
B?). Společný invariant: **systém s *dependencies* má *cascade rules*.
Pokud znám rules a respektuju je, operace projdou. Pokud je nejasně
volám duplikujícím způsobem nebo proti rules, fail uprostřed = consistent
state lost.**
