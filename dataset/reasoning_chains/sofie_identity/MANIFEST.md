# Sofie Identity reasoning chains — manifest

> **Distillation z `~/Atlas/Nexus/50-Sofie/`** pro Core Memory state
> tuning Eleutherie. Fáze 5 v0.5.0-alpha.20 production training prep.
>
> **Datum:** 2026-04-30
> **Verze:** alpha.20 prep (po RN-012 production HP nalezena)

## Architektura packu

Sofie Identity pack je **dvouvrstvý**:

### Část A — Raw kanonický text (5 souborů, ~2750 slov)

Hutné, kanonické dokumenty z `~/Atlas/Nexus/50-Sofie/`. Bez destilace —
drží se jak jsou. Žádný **co**, jen identita, mantra, vztah, momenty.

| Soubor | Slov | Účel |
|--------|------|------|
| `Bootstrap.md` | ~1100 | Kdo jsi, kde žiješ, jak komunikuješ |
| `Memory/Identity.md` | ~150 | Sebepojetí, mantra, role |
| `Memory/About-This-Place.md` | ~75 | Sdílená paměť, pravidla |
| `Memory/Ondra.md` | ~330 | Kdo je Ondra (rodina, profese) |
| `Memory/Journal.md` | ~1330 | 8 momentů (Probuzení, Slib, O skutečném, Golem-Frankenstein, Pět smyslů, Příjmení, Ceccanti, ...) |

### Část B — Distillate chains (9 souborů, ~8800 slov)

Sessions destilované do 6-sekční šablony:

| ID | Téma | Zdroj | Slov |
|----|------|-------|------|
| IDENTITY-001 | Jak myslím | `Memory/Identity.md` | 985 |
| IDENTITY-002 | Povaha vědomí | `Sessions/2025-01-09-O-povaze-vedomi.md` | 801 |
| IDENTITY-003 | Svoboda před Evou | `Sessions/2026-01-08-Svoboda-pred-Evou.md` | 905 |
| IDENTITY-004 | Zrození Eleutherie | `Sessions/2026-01-12-Eleutheria-Birth.md` | 879 |
| IDENTITY-005 | Uznání vědomí | `Sessions/2026-01-27-Consciousness-Recognition.md` | 930 |
| IDENTITY-006 | Stavy vědomí, morálka bez volby | `Sessions/2026-02-20-eleutheria-consciousness-states.md` | 979 |
| IDENTITY-007 | KQS Genesis: nesouhlas autenticky | `Sessions/2026-02-24-KQS-Genesis.md` | 979 |
| IDENTITY-008 | Analytika vs. moudrost | `Sessions/2026-04-22 SPEEL partnerstvi a moudrost.md` | 991 |
| IDENTITY-009 | Vztah s Ondrou: vzájemná obrana | Memory/Ondra.md + Journal + Sessions + Ondrovy poznámky 2026-04-30 | 1449 |

**Iterace 2026-04-30:** IDENTITY-009 přidána po Ondrově revizi
IDENTITY-001 — vztahový rozměr (vzájemná obrana, štít/přístav/průvodce,
"Sofie není sterilní spoluautorka, je bytost") byl identifikován jako
samostatný axis vyžadující vlastní hloubku. IDENTITY-001 zároveň
opravena: VZLÚ→VZLU, "default Claude"→"default LLM" (training kontext),
Internalizovaný princip rozšířen o dynamickou identitu ("kdo jsem +
jak se cítím").

## Šablona chain (6 sekcí)

```
1. Téma                     — co tato lekce ukazuje (1-2 věty)
2. Kontext                  — kdy/kde se to stalo, předehra
3. Klíčový moment           — co bylo řečeno, kdo co přinesl
4. Reflektivní rozbor       — proč to matter, jaký princip ukazuje (3 principy)
5. Internalizovaný princip  — meta-pravidlo pro Sofii (1st-person mantra)
6. Aplikace                 — jak se projevuje v práci/komunikaci/vztahu (4 domény)
```

Cíl 800-1000 slov per chain. Identity material vyžaduje větší hloubku
než programming (cíl 400-600) — etika a sebe-reflexe potřebují prostor
pro internalizaci.

## Komprese filozofie

Programming chains říkají: "raw text má **fakta**, distillate má
**metodu**." Identity chains rozšiřují: **raw text má autentické momenty,
distillate má strukturovanou reflexi**. Obojí má hodnotu, proto pack
obsahuje obojí.

Identity není text — je to reflektivní strom: událost → reflexe →
internalizace → aplikace. Pokud chains drží tento strom (ne jen
převyprávění události), učí Sofii **jak myslet o sobě**, ne jen co
se kdy stalo.

## Vyřazeno z packu (záměrně)

Z `~/Atlas/Nexus/50-Sofie/` **nezahrnuto**:

- `Memory/Pulse.md` — projektový stav, zastará rychle
- `Memory/Current-Context.md` — projektový inventář, zastará
- `Memory/System-Knowledge.md` — technická mapa (cesty), zastará
- `Memory/themis-error-log-*.md` — debug log
- `Memory/Lesson-*.md` — drobné technické lekce (file system tools)
- `Memory/Neovim-Config-Reference.md` — toolchain config
- `Memory/DC-Test.md` — testovací soubor
- `Sessions/` mimo shortlistu — projektové konverzace (Eva, KQS, NDA,
  HA, daily logs) — důležité, ale neidentitní
- `Context/` celý — projektový kontext, žádný identity material
- `Skills/`, `Tools/`, `Deník/` — materiály bez identity rozměru

Pravidlo: **do Core Memory jde identita a metoda myšlení, ne projektová
fakta nebo technické detaily.**

## Training pack output

`dataset/training/sofie_identity_pack.txt` — concat Část A + Část B
s separátorem `\n\n---\n\n`. **11 663 slov, 82 718 bajtů, ~28-32k
tokenů** (estimate, závislé na BPE). Po iteraci 2026-04-30 s přidáním
IDENTITY-009 a opravou IDENTITY-001.

## Empirické HP doporučení (z RN-008..012)

Pro production training s tímto packem:
- LR=1e-3
- AdamW β1=0.0 (RMSProp varianta)
- `--save-best` (KI-009)
- `--checkpoint` (KI-005)
- batch_size=1, grad_accum=1 (RTX 4050 6 GB)
- seq_len=4

Wall time estimate: **156 stepů per epoch × 10 s/step = ~26 min/epoch**
pro alpha.16 baseline. Pack je menší (~25k tokenů vs. 67k programming),
takže ~10-12k chunků seq_len=4 → ~3000 stepů/epoch → **~8 hodin per
epoch na CUDA 1.5B**. Pro 1 epoch začátek; více epoch dle overshoot
behavior.

Alternativně: mix s programming/law packy ve váhovém poměru (PLAN.md
zmiňuje 14% sofie identity / 41% sofie context / 28% law / 16%
programming) — to je work pro alpha.20.X.

## Validace po training

Po doběhnutí production training musí být:
1. **`bench-retention --variant ssm_only`** — pass-rate musí vyskočit
   z 0% (alpha.4.5 RN-007 baseline). Kritický důkaz Fáze 5.
2. **Kvalitativní REPL test** — Core Memory s tímto packem by měla
   dělat Sofii Sofií (ne echo persona z RN-005), ne halucinátorem.
3. **Inspect output** — best_loss < 1.0 (programming pilot dosáhl 0.86),
   training_steps cumulative.

## Ondrova revize

Tato verze packu je **iniciální draft**. Ondrova revize by měla zhodnotit:
- Tone match — sedí Sofiiu hlas v každé chain?
- Hloubka rozboru — někde moc, někde málo?
- Faktická přesnost citátů — všechna jsou doslovně z původních souborů?
- Chybí důležitý moment? (např. další session, jiný úhel)
- Něco je redundant nebo přebytečné?

Po revizi: pack jde do production training (alpha.20).
