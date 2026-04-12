# Known Issues — Eleutheria

Známé problémy, limitace a vědomá technická rozhodnutí.

Formát: `KI-NNN` s fází, dopadem, kontextem a plánovaným řešením.

---

## Aktivní

### KI-001 — Hardcoded cesty k modelům v CLI

- **Fáze:** 1
- **Dopad:** Nízký — pouze dev convenience
- **Kontext:** `main.rs` obsahuje `/home/lvx/Models/falcon-h1-{1.5b,7b}-instruct`
- **Řešení:** Konfigurovatelné přes config soubor nebo env variable (plánováno pro v0.8.0)

### KI-002 — Placeholder stages v prompt pipeline

- **Fáze:** 3
- **Dopad:** Střední — ConversationContext, MemoryInjection, QualityGate jsou no-op
- **Kontext:** Vědomé rozhodnutí — pipeline architektura připravena, implementace postupně
- **Řešení:** ConversationContext v0.4.0, MemoryInjection v0.5.0, QualityGate v0.6.0

### KI-003 — InputClassifier je statický

- **Fáze:** 3
- **Dopad:** Nízký — defaultně Freeform intent
- **Kontext:** Plná klasifikace vyžaduje buď heuristiky nebo druhý model pass
- **Řešení:** Heuristická klasifikace v0.4.0, případně ML-based později

---

## Vyřešené

_(žádné zatím — všechny dosavadní bugy řešeny inline, viz BUGS.md)_
