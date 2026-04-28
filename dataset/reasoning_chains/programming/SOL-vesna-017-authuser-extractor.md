# Reasoning chain — AuthUser extraktor a "TODO opravit ve vX.Y.Z" amnesie

## Zdroj
`vesna/SOLUTIONS.md#SOL-017` — privilege escalation: handlery org-scoped
endpointů četly `org_id` z query parametru, ne ze session. CRIT-01
z bezpečnostního auditu 2026-04-04.

## Kontext
Vesna backend má org-scoped endpointy: `/api/projects?org_id=<uuid>`
vrací projekty dané organizace, `/api/assets` umožňuje upload do
projektu organizace. Autentifikace přes session cookie, autorizace
přes membership v dané organizaci. V kódu existoval `AuthUser` extraktor
(cookie → SHA-256 hash → `sessions.org_id` lookup), který už od
v0.3.5 měl vracet trusted `org_id` ze session. **Routes ho ale nevolaly.**
Místo toho četly `org_id` z URL query nebo z request body, věřily
hodnotě poslané klientem. V komentářích u handlerů stálo
`// TODO: nahradit org_id z query za AuthUser session lookup ve v0.3.5`.
Ten komentář tam byl od v0.3.4. v0.3.5 prošla, v0.3.6 prošla, audit
v0.4.0 odhalil, že fix se nikdy neimplementoval.

## Analytický flow

1. **Začni od konkrétního útoku, ne od abstraktního principu.**
   Autentizovaný uživatel s cookie ke svému `org_id = aaa` může
   poslat `GET /api/projects?org_id=bbb` a backend mu vrátí projekty
   organizace `bbb`. Autentifikace prošla (cookie je validní), ale
   autorizace selhala — handler nezeptal "patří přihlášený uživatel
   k orgu, jehož data čte?". Důsledek: privilege escalation napříč
   organizacemi. To není bug v jedné route, to je systemic failure
   v autorizační vrstvě.

2. **Najít všechny dotčené endpointy.** Grep `org_id` napříč
   `routes/`. Sedmnáct match. Některé jsou legitimní (interní
   admin endpointy, kde `org_id` má smysl jako parametr), většina
   (12) je org-scoped a chybí `AuthUser`. Bez kompletního auditu
   bych mohla opravit dvě a tři jiné nechat — přesně to, jak vznikla
   původní díra.

3. **Princip fix: trust boundary leží na hraně serveru.** Klient je
   nedůvěryhodný a může poslat libovolný `org_id`. Server *zná*,
   ke kterému orgu uživatel patří, ze své vlastní session DB.
   **Identita autorizační decision MUSÍ pocházet ze server-side
   state, ne z client input.** AuthUser extraktor přesně tohle
   dělá — čte cookie, validuje session v DB, vrací `org_id` ze
   session row. Hodnota poslaná klientem je v tomto kontextu
   nerelevantní.

4. **Pořadí extraktorů v Axum handler signature je důležité.**
   Axum extraktory běží v pořadí, v jakém jsou v signatuře.
   Body-consuming extraktory (Json, Multipart) **konzumují
   request body**, takže musí být *poslední*. Pokud bych dala
   `Json(payload)` před `AuthUser`, body je už pryč, AuthUser
   nemá co číst (cookie je v headers, ne body, ale cookie helper
   v některých variantách čte celé requeste). Bezpečné pořadí:
   `State → AuthUser → Path/Query → Json/Multipart`. Zapsat
   jako pravidlo a držet konzistentně.

5. **Co dělat s `org_id` polem ve struct ListParams a CreateProjectRequest.**
   Aktuálně každá DTO má `org_id: Uuid`. To je strukturální pozvánka
   k chybě — pokud pole existuje, někdo ho někdy použije. Smažeme
   ze všech DTOs, takže ani omylem nebude k dispozici. Volání
   `repository::list(user.org_id, ...)` místo `repository::list(params.org_id, ...)`.
   Compiler teď fail-fast každého, kdo by se pokusil vrátit zpět
   k `params.org_id` — pole už neexistuje.

6. **Regression testy proti přesně tomuto útoku.** `bez_session_401`
   ověří, že žádost bez cookie spadne s 401. `invalidni_session_401`
   ověří, že žádost s neplatnou cookie taky spadne. Klíčový
   security test je třetí: `cizi_org_id_neobsahuje_projekty_kqs`
   — autentifikovaná žádost s `?org_id=<cizí UUID>` se musí
   chovat **stejně**, jako kdyby `org_id` v query nebyl. Test
   formálně dokumentuje, že trusted source je session, ne client
   input. Bez něj fix přežije přesně do první refactoring iterace.

7. **Pohled na `// TODO opravit ve v0.3.5` v kódu.** Ten komentář
   tam byl od v0.3.4. Tři minor verze. Každá merge byla opportunity
   chytit to a fixnout. Žádná to neudělala. **Komentář nemá
   compliance mechanismus** — code review se nedívá na všechny
   TODO, CI nehlídá, že TODO mizí ve své target verzi. Když říkám
   "opravím to později", musím to *zapsat někam, kde ten závazek
   přežije moji paměť*. Buď to opravím teď, nebo to jde do
   ROADMAP / KNOWN-ISSUES s konkrétní deadline. **TODO komentář
   v produkčním kódu je technický dluh s amnezií** — vznikl
   s úmyslem, neexistuje s vědomím.

## Aplikovatelné principy

- **Trust boundary je tam, kde data přechází z untrusted do trusted
  zóny.** V web aplikaci hranice leží mezi HTTP request (untrusted)
  a server logic (trusted). *Vše* z requesty je untrusted, dokud
  server explicitně nevalidoval. Identita uživatele NIKDY nepatří
  z request body nebo query parametru — vždycky z session, JWT
  signature, mTLS cert, nebo jiného server-validated source.
- **Server-side state má autoritu nad client-side input.** Pokud
  server zná `user.org_id` ze session, klient nemůže to override.
  Pokud server zná `user.role` z DB, klient nemůže to override.
  Stejný princip pro `user.id`, `user.permissions`, atd. Cokoli,
  co rozhoduje o autorizaci, musí pocházet z trusted source.
- **Strukturální absence je silnější než runtime check.** Smazat
  `org_id` field z DTO znemožňuje regrese typu "někdo se vrátí
  k starému patternu". Runtime check by to musel hlídat. Compiler
  hlídá zadarmo.
- **TODO komentáře jsou závazek bez compliance mechanismu.** Jediný
  TODO, který se reálně opraví, je TODO s deadline + tracking
  systémem (issue tracker, ROADMAP, KNOWN-ISSUES). TODO bez tracking
  se promění v "věčné TODO" — žije v kódu, dokud někdo nehledá
  v auditu, co divného tam je.

## Závěr

```rust
// PŘED:
async fn list_projects(
    State(state): State<AppState>,
    Query(params): Query<ListParams>,  // ❌ obsahuje org_id z klienta
) -> impl IntoResponse {
    let projects = state.projects.list(params.org_id, ...).await?;
    Json(projects)
}

// PO:
async fn list_projects(
    State(state): State<AppState>,
    user: AuthUser,                    // ✅ org_id ze session
    Query(params): Query<ListParams>,  // ✅ ListParams bez org_id pole
) -> impl IntoResponse {
    let projects = state.projects.list(user.org_id, ...).await?;
    Json(projects)
}
```

Pořadí extraktorů: `State → AuthUser → Path/Query → Json/Multipart`.
Body-consuming extraktor poslední.

Test pattern (`tests/auth_test.rs`):

```rust
#[tokio::test]
async fn cizi_org_id_neobsahuje_projekty_kqs() {
    let app = spawn_test_app().await;
    let (token, user_org) = authed_session(&app, "alice@kqs").await;

    let resp = send_authed(
        &app, &token,
        Request::get(format!("/api/projects?org_id={}", OTHER_ORG_UUID))
            .body(Body::empty()).unwrap(),
    ).await;

    assert_eq!(resp.status(), StatusCode::OK);
    let projects: Vec<Project> = json_body(resp).await;
    // Vrátí projekty user.org_id (KQS), NE OTHER_ORG_UUID
    assert!(projects.iter().all(|p| p.org_id == user_org));
}
```

## Přenositelný pattern

Při review nebo psaní route handleru s autorizační dimenzí procházím
tímto checklistem:

1. **Kdo identifikuje uživatele?** Cookie? JWT? mTLS? API key?
   To je jasné. Žádná z těchto věcí nemá být v request body
   ani query parametru. Identifikace patří do header / cookie /
   transport layer.

2. **Kdo autorizuje akci?** Server-side, na základě identifikace.
   "Tento uživatel patří do orgu X, route vyžaduje membership
   v X" — tahle decision se dělá *po* identifikaci, *před*
   business logikou.

3. **Co z requesty mám právo věřit?** *Nic.* Query parametry,
   body fields, headers, kromě těch, které jsem podepsala
   nebo validovala. Pokud handler dostane `org_id` z query,
   musím se ptát: byl validován? Pokud ne, je to attack surface.

4. **Compiler-enforced > runtime-checked > comment-noted.** Pokud
   můžu odstranit nebezpečné pole z DTO, udělám to. Pokud musí
   zůstat (legitimní use case), validuji ho na vstupu.
   Komentář "tohle je nebezpečné" je nejslabší ochrana.

5. **TODO bez tracking neexistuje.** Pokud řeknu "opravím to
   ve v0.3.5", musí to být v ROADMAP nebo issue trackeru.
   Komentář v kódu nestačí. Code review nečte všechny komentáře,
   CI nehlídá termíny v komentářích, nový vývojář komentář
   vůbec neuvidí.

Pattern se přenáší do každé bezpečnostní rozhodovací situace:
file upload (path traversal), SQL queries (parameterization),
deserialization (untrusted input), inter-service auth (mTLS,
service tokens), webhooks (signature verification). Všude tam,
kde data přechází z untrusted na trusted zónu, je hranice **explicit
boundary** s **explicit validation**. Komentáře, intuice, "asi to
bude OK" — to nejsou bezpečnostní mechanismy. Test, který by
attack reálně provedl a ověřil, že selže, je. Pokud test neexistuje,
attack pravděpodobně někdy projde.
