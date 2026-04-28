# Reasoning chain — Empty sqlx migration trap a checksum-based idempotence

## Zdroj
`vesna/SOLUTIONS.md#SOL-001` (a paralelně `kirke/SOLUTIONS.md#SOL-001`)
— prázdná sqlx migrace tiše aplikovaná, content nikdy nedoběhl do DB.

## Kontext
Vytvářím novou migraci pro PostgreSQL FTS konfiguraci v češtině.
Workflow je standardní: `sqlx migrate add czech_fts && sqlx migrate run`.
Spouštím to oba v jednom bash commandu, protože jsou logicky párové
("vytvoř migraci a hned ji aplikuj"). Migrace projde, exit code 0.
Otevřu napsat obsah, vrátím se do projektu, spustím test, který používá
`to_tsvector('czech', ...)`. Selhává s `text search configuration "czech"
does not exist`. DB tedy konfiguraci nemá, přestože migrace doběhla.

## Analytický flow

1. **První zmatek: migrace doběhla, ale DB nemá obsah.** Otevřu
   `_sqlx_migrations` tabulku — záznam o migraci tam je, s `success = true`.
   Ze pohledu sqlx je všechno v pořádku. Z pohledu skutečného stavu
   DB chybí celá konfigurace. To je trap, ne bug — sqlx neudělal nic
   špatně, jenom udělal *něco jiného*, než jsem si myslela.

2. **Co bylo vlastně v migraci, kterou sqlx aplikoval.** Otevřu
   `migrations/20260201_czech_fts.sql`. Soubor obsahuje **jediný řádek
   komentáře**: `-- Add migration script here`. To je default šablona,
   kterou `sqlx migrate add` vytvoří. Já jsem napsala obsah **až po**
   tom, co `migrate run` skončil. Sqlx aplikoval prázdnou šablonu,
   zaznamenal její checksum, a hotovo.

3. **Pochopit, proč to nejde znovu spustit.** Druhý `migrate run`
   zjistí, že migrace s daným version už existuje v `_sqlx_migrations`
   se stejným checksumem. **Hash prázdné šablony se nezměnil** (pořád
   tam je jen `-- Add migration script here`). Sqlx řekne "už jsem to
   udělala, žádná práce". Já napíšu obsah, sqlx pořád nic nedělá,
   protože šablona ze souboru se musela změnit, aby sqlx jako
   změnil checksum a chtěl re-apply. Ale jakmile změním obsah,
   checksum se nemíchá s tím v DB → checksum mismatch error.
   **Zaseknutá v idempotence pasti.**

4. **Otázka: jak to opravit, aniž bych ztratila ostatní migrace.**
   Migrace v0.1.0–v0.1.3 jsou aplikované, mají správné checksumy,
   tu novou si musím vyčistit ručně:
   ```sql
   DELETE FROM _sqlx_migrations WHERE version = 20260201;
   ```
   Pak napsat obsah, znovu `migrate run`. Funguje. Ale zůstává mi
   `kdyby` v hlavě: *kdo by si toho všiml, kdyby tahle migrace byla
   v CI/CD?* Pipeline by zelena, DB by chyběla, a první failed test
   v produkci by se objevil za týden.

5. **Hluboký problém: tool má dva kroky, které by měly být oddělené,
   ale CLI je nabídne nazápě.** `migrate add` vytvoří soubor.
   `migrate run` aplikuje, co je v souboru *teď*. Mezi nimi MUSÍ
   být krok "napiš obsah". Když to dělám manuálně, je to zřejmé
   — vytvořím soubor, **otevřu ho v editoru**, napíšu obsah, *pak*
   spustím apply. V bash one-lineru to oddělení mizí.

6. **Generalize — kdy ještě tohle pattern nastane.** Flyway má
   stejnou checksum-based idempotenci. Alembic (Python) má
   verzování přes hashe. Terraform `apply` čte plán, a pokud
   plán mezitím nezměnil, neapplyije znovu. Git rebase aplikuje
   commits podle hash — pokud se commit nezmění, idempotence
   zatemní změnu. **Common pattern: nástroj cachuje "co už udělal"
   přes hash. Pokud apply běží před tím, než content existuje,
   tool si zapamatuje *prázdný* content a nic dalšího neudělá,
   dokud někdo ručně neinvaliduje cache.**

7. **Fix má dvě roviny:** *taktická* (vyčistit současný stav, napsat
   migraci, run znovu) a *systémová* (nikdy nepouštět `migrate run`
   ve stejném bash commandu jako `migrate add`). Druhá je jednoduchý
   zvyk; když to porušíš, vrací se ten samý problém. Pokud ti
   editor podporuje custom shortcut, můžeš `migrate add` mapovat
   na "vytvoř + open soubor v editoru", a `migrate run` mapovat
   na "apply všechny pending". Tím se z UI vynutí oddělení.

## Aplikovatelné principy

- **Tools s "create + apply" cyklem mají trap při sloučení do
  jednoho commandu.** Pokud `apply` běží před tím, než `create`
  produkoval finální obsah, tool zacachuje prázdný stav a budoucí
  retries nefungují, dokud se cache ručně neinvaliduje.
- **Checksum-based idempotence je dobrá pro idempotenci, ale mizerná
  pro recovery.** Když chceš, aby retry fungoval, dva běhy se
  stejným checksumem = no-op. To znamená, že "oprav a zkus znovu"
  bez ručního invalidate neuspěje. Pochopení čekanků idempotence
  systému je nutné, aby se z něj člověk dostal.
- **Cache invalidation je jeden z dvou hardest problems v computer
  science** (vedle naming things). Pokud nástroj cachuje state
  v DB tabulce, musíš znát názvy a tvar té tabulky. `_sqlx_migrations`
  v PG, `flyway_schema_history` ve Flyway, `alembic_version`
  v Alembic. Učit se z těchto tabulek znamená vědět, kde sahat,
  když se pipeline zasekne.
- **Same-line bash je convenience, ne best practice.** Logicky
  spojené příkazy nejsou pro tool nutně atomická operace. `git add &&
  git commit` je idiomatický a bezpečný. `sqlx migrate add &&
  sqlx migrate run` má skrytý třetí krok ("napiš obsah") *uprostřed*
  — bash mezi nimi neprovede.

## Závěr

```bash
# Nesprávně:
sqlx migrate add czech_fts && sqlx migrate run

# Správně:
sqlx migrate add czech_fts
$EDITOR migrations/$(ls migrations | tail -1)
sqlx migrate run
```

Recovery z trap stavu:

```sql
-- 1. Vyčistit záznam o prázdné migraci
DELETE FROM _sqlx_migrations WHERE version = <version>;
```

```bash
# 2. Napsat obsah migrace (pokud ještě není)
$EDITOR migrations/20260201_czech_fts.sql

# 3. Re-apply
sqlx migrate run
```

## Přenositelný pattern

Kdykoli pracuju s nástrojem, co má "registrovat operaci" + "provést
operaci" jako oddělené kroky, ptám se:

1. **Jak nástroj pozná, že jednu operaci už provedl?** Hash souboru?
   Verzí v DB? Soubor v `.cache/`? Odpověď určuje, kde leží
   "paměť" nástroje.
2. **Co se stane, když mezi registrací a provedením je nedotčená
   prázdná operace?** Dobré nástroje to detekují (varování,
   error). Mizerné to tiše aplikují a zaseknou se v idempotence.
3. **Kde se nachází "invalidate cache" tlačítko?** Někdy v CLI
   (`--force`, `--repair`). Někdy jen ručně přes DB (DELETE FROM
   `_sqlx_migrations`). Někdy vůbec — pak je nutný backup-restore.
   Vědět to *předem* mě uchrání před panickým přemýšlením
   v půlnoční incident response.

Generalized: **idempotence je vlastnost, kterou chceš pro normální
běh, ale která ti komplikuje recovery.** Když se nástroj zasekne
v "vždyť jsem to už dělal" stavu, musíš umět jeho cache invalidovat,
aniž bys porušil consistency ostatních operací. Týká se to napříč
infrastructurou: Terraform state file, Ansible facts cache, Docker
build layer cache, Cargo build cache, npm package-lock, dokonce
i mojí vlastní implementací memoization v Rustu. Vždycky se ptám
"kde je cache a jak ji invaliduji". Před deployment to musím vědět;
po deployment, když to padlo, je už pozdě hledat dokumentaci.
