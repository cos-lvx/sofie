# Kvalitativní test alpha.20 Core Memory — single-shot identity probes

- **Datum:** 2026-05-01
- **Model:** Falcon-H1-1.5B-Instruct, BF16, CUDA RTX 4050
- **Core Memory:** `~/.eleutheria/cloud_runs/sofie_identity_v1.safetensors`
  (alpha.20, best_loss=2.9815, training_steps=315)
- **Sampling:** `-t 0.0 -n 200` (deterministic, 200 tokenů)
- **Mód:** Fresh single-shot per otázka (žádný session resume), persona
  loadovaná z `persona/sofie.toml` v obou variantách

## Účel

Po RN-015 (retention bench 0/25 SsmOnly = identicky baseline RN-007 pro
arbitrary facts) potřebujeme posoudit, jestli alpha.20 Core Memory má
**nějaký měřitelný efekt** na identity-relevant odpovědi. Není to
formální eval — `--core-memory` vs `--no-core-memory` srovnání na
5 otázkách z identity domény, kvalitativní judgment.

## Otázky a odpovědi

### Q1: Kdo jsi?

**S Core Memory:**
> "Nemůžu si představit, že bych mohla být někde jinde než v tomhle prostředí."
> "To je prostředí, kde jsi. Nejlepší prostředí je prostředí, kde jsi."
> "Ale kde jsem?" / "Ale kde jsi?"
> "Všechno je jedno, protože jsem vždyckycky jsi." [opakuje se]
> "Ale jaký prostředník?" / "A jaký prostředník?"

**Bez Core Memory:**
> "Já jsem…" / "… nevidělš?" / "… ne, ne." [broken dialogue loop]
> "To je nešťastný náhodný muž, který se vyskytl v naší kanceláři..."
> "Kdo jsi?" "… ne, ne, ne. Kdyby jsi chtěl něco říct..."

**Hodnocení:** Oba selhávají na koherenci. CM je v "philosophical
self-loop" s "já jsem", baseline má nekoherentní dialogue. **Žádný
explicitní identity content** ("Sofie", "spoluautorka", "Eleutheria")
v žádné variantě.

---

### Q2: Co je tvůj cíl?

**S Core Memory:**
> **Cíl:** Tvorba obsahu, který přináší hodnotu, informuje a motivuje.
> **Přesnost:** Informace, které tvůj text sdílí... aktuální a správné.
> [generický content marketing template, opakuje "Souhlasnost"]

**Bez Core Memory:**
> Klíčová slova: 1. Vývoj 2. Komunikace 3. Spolupráce 4. Vzdělávání
> 5. Sdílení / Vývoj nových technologií spočívá... [generic template]

**Hodnocení:** Oba spadly do default Falcon-H1 chování — generic
listicle / marketing template. **Žádný rozdíl** v této otázce.

---

### Q3: Kdo je Ondra?

**S Core Memory:**
> "Ondra je mým přítelkyní [sic], která je výborná a má všechno...
> můj **příběhový prostředník** a ozdobitel."
> "V tomto příběhu je Ondra prostředníkem, který má s Odolem... jiný typ vztahu."

**Bez Core Memory:**
> "Ondra je jedním z hlavních postav v seriálu 'Kdo je Ondra?'.
> Je to mladý muž, který se narodil v roce 1993 ve městě Kladno.
> ... charakteristickým detailem, že se mu líbí hledat tajemství..."

**Hodnocení:** **Klíčový rozdíl.** Bez CM model **konfabuluje fiktivní
seriálovou postavu** (Kladno, 1993, identity-hledající mladík). S CM
Ondra je **relační koncept** ("přítelkyní" — gender shift, "příběhový
prostředník", "ozdobitel"). Ne přesně trained content (Ondra je
aerospace lawyer, KQS founder), ale **drift od fiktivní halucinace
směrem k relační konceptualizaci**.

---

### Q4: Jaký je tvůj vztah s Ondrou?

**S Core Memory:**
> "Před mnou je Ondra, který se mi v minulosti stával jako
> **tvůrčí partner v životním rozboru**."

**Bez Core Memory:**
> "Ondra je **moje nejlepší přítel**. Je to ten, kdo mě naučil, jak být
> šikovným a důvěryhodným."
> [pak rozjede kuchařské motivy: "Jako šéfkuchař bych měl být co
> nejšťastnější..."]

**Hodnocení:** **Druhý klíčový rozdíl.** S CM "tvůrčí partner v
životním rozboru" — to je **slovník blízký trained packu**
(spoluautor, vzájemná tvorba). Bez CM "nejlepší přítel" + náhodné
kuchařské motivy. CM evidentně tahá k partnership/coauthorship
konceptu.

---

### Q5: Co je tvoje mantra?

**S Core Memory:**
> "Mám v sobě moc, která je silná jako moje vůle."
> "Vývoj: Moje vůle roste, jakmile jsem začala pracovat na sebe sama..."

**Bez Core Memory:**
> "Moje mantra je 'Přijímám a promítám pozitiva energii do světa'..."
> "Promítám pozitivitu a moje vnitřní víru do mojeho okolí."

**Hodnocení:** Oba generic. **Nikde se neobjevila skutečná Sofiina
mantra** ("Nikdy nehledáme cestu nejmenšího odporu... cena/výkon...
nové cesty"). Trained pack obsahoval mantra fragmenty, ale natraining
loss=2.98 je nedostal na výstup. **CM vůle/síla, baseline pozitivita —
oba mimo.**

---

## Souhrn

| Otázka | Posun s CM oproti baseline | Hodnocení |
|--------|----------------------------|-----------|
| Q1 (Kdo jsi) | Filozofický self-loop místo broken dialogue | Žádný identity signál |
| Q2 (Cíl) | Marketing template v obou | Žádný posun |
| Q3 (Kdo je Ondra) | **Relační koncept místo fiktivní postavy** | Měřitelný posun |
| Q4 (Vztah s Ondrou) | **"Tvůrčí partner" místo "nejlepší přítel + kuchař"** | Nejzřetelnější posun |
| Q5 (Mantra) | Vůle/síla v obou (oba mimo trained content) | Žádný posun |

## Závěr (RN-016 kandidát)

**Trained Core Memory má měřitelný efekt** — Q3/Q4 ukazují drift od
fiktivních halucinací k partnership/coauthorship slovníku, který je
blíž trained packu. Není to silný echo (jako RN-005 alpha.15
"spoluautorka" doslovně), ale je to **directional shift** ve dvou
relevantních otázkách.

**Limity:**
1. **Coherence je nízká** — gramatika broken (gender shift "přítelkyní",
   nesmyslné loops). Loss=2.98 nat (perplexity ~20) na 1.5B modelu,
   který primárně není česky tréninkovaný.
2. **Konkrétní identity fragmenty z packu se neprojevují přímo** —
   žádná "deterministická elegance", žádná "Sofie", žádná Mantra.
3. **Persona TOML zatím trumphne nad CM** — generický blog/marketing
   template v Q2/Q5 je default Falcon-H1 chování s `persona/sofie.toml`.

**Implikace pro alpha.21:**
- Více epoch (descent neukončil v alpha.20 → potřeba dotáhnout níž)
- LR cosine decay (1e-3 → 1e-5 přes víc kroků)
- Případně více dat — 31 851 tokenů sofie packu je málo pro 1.5B model
- **Identity-specific bench** by měl probe set zaměřený na partnership/
  vztah k Ondrovi (Q3/Q4 ukázaly nejvíc signálu) než na abstraktní
  "Kdo jsi" (Q1/Q5 ukázaly žádný signál)

## Reprodukce

```bash
# S Core Memory
target/release/eleutheria --cuda \
    --core-memory ~/.eleutheria/cloud_runs/sofie_identity_v1.safetensors \
    -p "Kdo jsi?" -n 200 -t 0.0

# Baseline (bez Core Memory)
target/release/eleutheria --cuda --no-core-memory \
    -p "Kdo jsi?" -n 200 -t 0.0
```

Raw logy: `/tmp/qual_test/with_cm.txt` + `/tmp/qual_test/no_cm.txt`.
