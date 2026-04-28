# Reasoning chain — Test DB isolation a "config je pro app, ne pro testy"

## Zdroj
`kirke/SOLUTIONS.md#SOL-006` — integrační testy běžely proti DEV DB,
ne proti dedikované TEST DB. Každý test volal `DELETE FROM items`
před spuštěním. Než se objevila reálná data v DEV DB, problém zůstal
neviditelný.

## Kontext
Kirké backend má integrační testy v `backend/tests/` — `health_test`,
`items_api_test`, `bulk_test`, `stats_test`. Každý vytvoří axum
test server, požádá o tasted endpoint, ověří výsledek. Konvencionální
setup: test volá `KirkeConfig::load()`, který přečte `kirke.toml`
+ aplikuje `DATABASE_URL` env override. Pak `pool.connect()` na
adresu z configu. Pak `DELETE FROM items` (clean state pro test).
Pak test logika.

V `.env` mám `DATABASE_URL=postgres://localhost/kirke` (DEV DB).
V CI mám stejné, jen DB je dedicated CI postgres. **TEST DB
`kirke_test` existuje, ale config soubor ji nepoužívá.** Testy
proto sjely proti DEV. Dokud byla DEV DB prázdná, problém se
neprojevil. První spuštění aplikace se seedem 100 items, pak run
testů → `DELETE FROM items` smázl všech 100 items v DEV. **Test
data destruction proti aplikační DB.**

## Analytický flow

1. **Diagnóza: kde testy berou DB connection?** Otevřu `health_test.rs`,
   přejdu na setup. `let config = KirkeConfig::load();` — to čte
   `kirke.toml` a override z env. `let pool = PgPool::connect(&config.
   database_url).await?;` Pohodlí: jeden setup volání. Bug je v tom
   pohodlí — config je pro **aplikaci**, ne pro test prostředí.

2. **Otázka: proč to nikdo nezachytil dřív?** Tři důvody:
   - DEV DB byla většinou prázdná během vývoje. `DELETE FROM items`
     na prázdné tabulce je no-op. Test prochází.
   - Žádný developer neměl důvod si DEV DB seed prepopulovat trvale.
   - Test runtime je rychlý (~10s), nikdy ne tak pomalý, aby vyvolal
     pochybnost "co to právě dělá".
   **"Funguje" v sense pass/fail, ale logická správnost je
   masquerade.**

3. **Otevřená otázka: ověřit, že tahle struktura není i v jiných
   repos.** Vesna (sister repository) má pattern `tests/common/mod.rs`
   s `test_pool()` který explicitně preferuje `TEST_DATABASE_URL`.
   Vesna se to naučila *předtím* (při auditu Vesna fáze 0.2). Kirké
   ten pattern nepřevzala při forku — zkopírovala Cargo.toml,
   skipla testovací patterny. **Lekce: při forku nového workspace
   z proven repository, kopíruj test infrastructure stejně jako
   build infrastructure.**

4. **Návrh fixu: dedicated `tests/common/mod.rs` s guarded `test_pool()`.**
   Klíčová logika:
   ```rust
   pub async fn test_pool() -> PgPool {
       let url = std::env::var("TEST_DATABASE_URL")
           .or_else(|_| {
               let fallback = std::env::var("DATABASE_URL").ok()?;
               // Bezpečnostní guard
               if fallback.contains("_test") {
                   Ok(fallback)
               } else {
                   Err(())
               }
           })
           .expect("TEST_DATABASE_URL must be set, OR DATABASE_URL must contain '_test'");
       PgPool::connect(&url).await.unwrap()
   }
   ```
   Logika ve dvou krocích:
   - **Primary:** `TEST_DATABASE_URL` (explicit). Pokud existuje,
     použij.
   - **Fallback:** `DATABASE_URL` *ale jen pokud obsahuje `_test`*.
     Tj. `postgres://localhost/kirke_test` projde, `postgres://
     localhost/kirke` panickne s jasným hlášením.

5. **Proč ten guard `_test` v URL?** Defense in depth. Pokud někdo
   omylem nastaví `TEST_DATABASE_URL=DATABASE_URL` (typo, copy-paste),
   nebo jen použije jediné DB pro vše (zkušený DBA, ale jeden DB host),
   guard chytí, že název DB neobsahuje `_test`, a zkrachne s explicit
   error: "TEST DB URL must contain '_test' in name to prevent
   accidental data destruction." **Cheap safety net proti typu
   misconfiguration, který bych neviděla v CI flow.**

6. **Sdílený setup: `spawn_server()`, `cleanup_items()`.** Po fix
   každý test file začíná `mod common;` a používá `common::test_pool()`,
   `common::spawn_server()`, `common::cleanup_items()`. **Centralizace
   = jediný moudrý loc kontroly DB connection.** Když se config
   změní, mění se na jednom místě.

7. **Reflexe: "config" je polysémické slovo.** *Application config*
   říká "jak má aplikace běžet v produkci/staging/dev". *Test
   config* říká "kde test isolovaně vytvoří a strhne svůj svět".
   Mixování těchto dvou je past — application config nese
   production credentials, test config musí jasně oddělit. Pokud
   `KirkeConfig::load()` byl pro produkci, neměl by se v testech
   vůbec použít.

## Aplikovatelné principy

- **Integrační testy nesmí sdílet DB s aplikací.** Test runs vykonávají
  destructive operations (DELETE, TRUNCATE, ALTER) jako součást
  isolation. Aplikační DB obsahuje lidský data — never the twain
  shall meet. Dedicated test DB je *strukturální* prevence, ne
  *procedurální* (na kterou se zapomene).
- **Cheap safety net je investice s neomezeným return.** Guard
  `URL musí obsahovat _test` je 5 řádků kódu. Pokud kdykoli v
  budoucnu chrání před wipe produkční DB, ROI je nekonečný. Tyto
  kontroly se ne-amortizují, jsou to insurance.
- **Při forku z proven repository kopíruj infrastrukturu, ne jen
  závislosti.** Cargo.toml kopíruje knihovny. `tests/common/mod.rs`
  kopíruje test discipline. `.github/workflows/` kopíruje CI
  patterns. Setting up new repository znamená import battle-tested
  shapes, ne re-design from scratch.
- **Strukturální prevence > procedurální disclamer.** "Pamatuj
  spustit testy proti TEST_DATABASE_URL" v README je procedurální.
  Guard který panickne při wrong URL je strukturální. Procedurální
  závisí na tom, že developer pamatuje. Strukturální závisí na tom,
  že compiler/runtime checkuje. Druhé vyhrává.

## Závěr

```rust
// backend/tests/common/mod.rs
use sqlx::PgPool;

pub async fn test_pool() -> PgPool {
    let url = test_database_url();
    PgPool::connect(&url).await.expect("test DB connect failed")
}

fn test_database_url() -> String {
    if let Ok(url) = std::env::var("TEST_DATABASE_URL") {
        return url;
    }
    if let Ok(url) = std::env::var("DATABASE_URL") {
        if url.contains("_test") {
            return url;
        }
        panic!(
            "DATABASE_URL='{}' nesmí být použit jako TEST DB \
             (musí obsahovat '_test' v názvu).\n\
             Nastav TEST_DATABASE_URL=postgres://.../kirke_test",
            url
        );
    }
    panic!("TEST_DATABASE_URL nebo DATABASE_URL='..._test' musí být nastaveno");
}

pub async fn cleanup_items(pool: &PgPool) {
    sqlx::query!("DELETE FROM items").execute(pool).await.unwrap();
}

// V každém integration test souboru:
mod common;
use common::{test_pool, cleanup_items};

#[tokio::test]
async fn list_items_returns_empty_initially() {
    let pool = test_pool().await;
    cleanup_items(&pool).await;
    // ... test ...
}
```

## Přenositelný pattern

Při psaní integračních testů nebo CI infrastructure obecně:

1. **Testovací data isolation začíná u DB connection.** Před `setUp`
   musí být jasné, že test má vlastní DB, vlastní filesystem prefix,
   vlastní message queue topic. Sdílená DB je sdílená state — testy
   se navzájem ruší, race conditions, sporadic failures.

2. **Guard misconfiguration explicit panic, ne silent fallback.**
   Testy které tiše použijí špatnou DB jsou nejhorší výsledek.
   Test, který panickne s "TEST_DATABASE_URL not set" mě donutí
   to nastavit; test, který funguje proti aplikační DB mě nenechá
   pochopit, dokud nezničí data.

3. **Centralize setup do shared module.** `tests/common/mod.rs`
   v Rustu, `tests/conftest.py` v pythonu, `test_helper.exs`
   v Elixiru, `setup.ts` v TypeScriptu. Jeden modul s
   `setup_test_env()`, `teardown_test_env()`, helpers. Změna
   v setup logice = jeden edit.

4. **Při forking ze známého good repo, kopíruj test patterns.**
   Kopírování test infrastructure je drahší krátkodobě (víc kódu),
   levnější dlouhodobě (méně bugs ze začátku). Vesna měla pattern,
   Kirké si ho mohla převzít, neudělala — a strávila auditem
   znovu objevit, co Vesna už věděla.

5. **Test discipline je technická diskuze, ne stylistická.** "Testy
   běží proti vlastní DB" není opinion, je to invariant. Pokud
   to není dodrženo, nedělají testy svou práci — pretendují, že
   isolují, ale ne.

Pattern se přenáší: file system tests (use tempdir, ne real paths),
network tests (mock external APIs, nebo dedicated test endpoints),
state tests (reset state před každým testem), distributed tests
(spawned containers per test). Společný invariant: **test musí mít
plnou kontrolu nad svým prostředím — bez sdílených side effects, bez
zodpovědnosti za nadřazený context. Pokud test může poškodit data
mimo svůj scope, není isolated, je to bomba s odpočítáváním.**
