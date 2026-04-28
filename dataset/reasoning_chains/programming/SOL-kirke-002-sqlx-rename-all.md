# Reasoning chain — `rename_all` "lowercase" vs "snake_case" a slepá místa unit testů

## Zdroj
`kirke/SOLUTIONS.md#SOL-002` — `ItemCondition::VeryGood` se nepodařilo
INSERT do PG. API vracelo 500 `DATABASE_ERROR`. Postgres enum měl
hodnotu `'very_good'` (s podtržítkem), sqlx posílal `'verygood'` (bez).

## Kontext
Kirké backend — Rust + axum + sqlx + PostgreSQL. Doménový model
používá Rust enum pro každý fixní set hodnot: `ItemStatus`, `ItemCondition`,
`PlatformType`, `ListingStatus`, `JobType`, `JobStatus`. PostgreSQL
mirror typy s `CREATE TYPE item_condition AS ENUM ('new', 'like_new',
'good', 'very_good', 'fair', 'poor');`. SQL standard pro enum hodnoty
je `snake_case` (lowercase + underscore separator). Rust strana má:
```rust
#[derive(sqlx::Type)]
#[sqlx(type_name = "item_condition", rename_all = "lowercase")]
enum ItemCondition { New, LikeNew, Good, VeryGood, Fair, Poor }
```
Unit testy `ItemRepository::insert` prošly (testovaly `Good`, `Fair`,
`Poor` — single-word variants). API integrační test, který poslal
JSON s `"VeryGood"`, selhal s 500.

## Analytický flow

1. **Symptom: 500 v API, žádný 500 v unit testech.** To je první vodítko.
   Unit testy testovaly `ItemRepository::insert(item, pool)` — funkčí.
   API test postavil JSON, deserialized do Rust struct, předal repository
   — nefunguje. Mezi unit a API testem se něco mění.

2. **Diff mezi unit a API: jaký vstup mají?** Unit test používal:
   ```rust
   let item = Item { condition: ItemCondition::Good, ... };
   repo.insert(&item).await?;
   ```
   API test posílal JSON `{"condition": "VeryGood", ...}`, Rust to
   deserialized do `ItemCondition::VeryGood`. **Hodnoty enum jsou
   různé: unit používal `Good`, API používal `VeryGood`.** Single-word
   vs. multi-word.

3. **Hypotéza: jak `rename_all = "lowercase"` zachází s multi-word
   varianty?** Otevřu sqlx docs nebo vyzkouším:
   ```rust
   ItemCondition::Good     → "good"      ✓ (sedí s 'good')
   ItemCondition::VeryGood → "verygood"  ✗ (DB má 'very_good')
   ```
   `lowercase` jen převádí na lowercase, **bez separátoru**. Pro
   single-word je to bezvýznamné (`Good` → `good`, `Draft` →
   `draft`). Pro multi-word je to *špatně*. DB chce snake_case,
   sqlx posílá blob bez separátoru.

4. **Diagnóza confirmed.** `#[sqlx(rename_all = "lowercase")]` je
   wrong policy pro tenhle DB schema. Správná je `"snake_case"`,
   která produkuje:
   ```rust
   ItemCondition::Good     → "good"       ✓
   ItemCondition::VeryGood → "very_good"  ✓
   ItemCondition::LikeNew  → "like_new"   ✓
   ```
   Pro single-word identické s `lowercase`, pro multi-word přidá
   `_`.

5. **Fix: změnit ve VŠECH enum typech.** Grep `rename_all = "lowercase"`
   v repu — six matchů. Všechny změnit. Compile, run testy, run
   integrační test.

6. **Reflexe na unit test gap.** Proč tohle unit testy nezachytily?
   Unit testy testovaly *single-word variants*. Multi-word variants
   (LikeNew, VeryGood) nikdy nebyly v unit test fixtures. **Test
   coverage podle code path byl 100 % (testy prošly všemi větvemi
   v `insert`), test coverage podle data combinations byl <100 %**
   (chyběly multi-word fixtures). Unit testy jsou dobré pro logic,
   slepé pro data-driven divergence.

7. **Systémové učení: každý enum varianta musí být test fixture.**
   Pokud `ItemCondition` má 6 variant, test fixture by mělo zahrnovat
   všech 6. Property-based testing (proptest, quickcheck) by tohle
   pokrylo automatically — generuje random variants, ověřuje, že
   všechny round-trip přes DB. Bez property testing musí být explicit
   test pro každou variant.

## Aplikovatelné principy

- **Naming convention attribute na tváří v tvář DB schema je smluvní
  vztah, ne kosmetika.** sqlx `rename_all = "lowercase"` říká
  "concatenate words bez separátoru". DB schema říká "underscore-separated".
  Mismatch = hodnoty neexistují, runtime error.
- **Single-word variants jsou degenerate test cases.** `Good`, `Draft`,
  `Pending` všechny single-word — `lowercase` a `snake_case` produkují
  identické output. Test, který prochází jen single-word, nemůže
  rozlišit ty dvě naming conventions.
- **Test coverage podle code path ≠ test coverage podle data.** 100 %
  branch coverage neznamená 100 % data combinations covered. Pro
  enum-driven či data-driven systémy musí test fixtures zahrnout
  every variant, ne jen each branch.
- **`rename_all = "snake_case"` je default volba pro PG enum.** PG
  konvence je underscore-separated lowercase identifiers (table names,
  column names, enum values). `lowercase` je rare scénář pro typy,
  které mají "connected lowercase" hodnoty (např. tablename mapování
  na exact-string env var). Když jsem na pochybách, `snake_case`
  vyhrává.

## Závěr

```rust
// Před:
#[derive(sqlx::Type)]
#[sqlx(type_name = "item_condition", rename_all = "lowercase")]
enum ItemCondition { New, LikeNew, Good, VeryGood, Fair, Poor }

// Po:
#[derive(sqlx::Type)]
#[sqlx(type_name = "item_condition", rename_all = "snake_case")]
enum ItemCondition { New, LikeNew, Good, VeryGood, Fair, Poor }

// Test fixture, all variants:
#[tokio::test]
async fn insert_round_trips_all_conditions() {
    let pool = test_pool().await;
    for condition in [
        ItemCondition::New, ItemCondition::LikeNew, ItemCondition::Good,
        ItemCondition::VeryGood, ItemCondition::Fair, ItemCondition::Poor,
    ] {
        let item = Item { condition, ..Item::test_default() };
        let id = repo.insert(&item).await.unwrap();
        let loaded = repo.get(id).await.unwrap();
        assert_eq!(loaded.condition, condition);
    }
}
```

Aplikováno na všech 6 enum types: `item_status`, `item_condition`,
`platform_type`, `listing_status`, `job_type`, `job_status`. Plus
fixture, který round-tripuje every variant.

## Přenositelný pattern

Kdykoli pracuju s naming convention mappingem (Rust enum → DB enum,
JSON ↔ Rust struct, env var ↔ config field), procházím:

1. **Identifikuj naming convention na obou stranách.** Rust default:
   `PascalCase` typů, `snake_case` fields. PG default: `snake_case`
   všeho. JSON: `camelCase` napříč JS ekosystém. Env vars: `SCREAMING_SNAKE_CASE`.
   Mismatch convention = mapping headache.
2. **Pro každý mapping uveď, co konkrétně se konvertuje.** "lowercase"
   ≠ "snake_case" ≠ "kebab-case" ≠ "camelCase" ≠ "PascalCase". Tyhle
   jsou diskrétní hodnoty, ne synonyma. Doc check.
3. **Test fixture musí zahrnovat single-word i multi-word variants.**
   Single-word je degenerate case (mnoho conventions se chová
   identicky); multi-word odhaluje rozdíly.
4. **Round-trip testy odhalí asymmetric mapping.** Insert → load →
   compare. Pokud serializace a deserializace nejsou inverse, pattern
   se objeví v round-trip.
5. **Property-based testing pro enum-driven systémy.** `proptest::
   prop_oneof![]` generuje random variant, test ověří, že všechny
   round-trippují. Catches "I forgot to handle this variant" cases.

Pattern se přenáší: i18n key naming (translation files vs. code
references), API field naming (request body vs. response body vs.
internal type), config file ↔ runtime struct (TOML keys, YAML keys),
URL path vs. handler signature. Společné: kdekoli existuje *mapping*
mezi name spaces, je tam past mismatch convention. Disciplína:
explicitní convention statement, fixture pro multi-word, round-trip
test.
