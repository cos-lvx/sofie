# Reasoning chain — RNG double-borrow a "extract pure function s explicit state"

## Zdroj
`morana/SOLUTIONS.md#SOL-016` — borrow checker odmítl `ParticleSystem::update()`,
které iteruje přes `&mut self.emitters` a uvnitř volá `self.spawn_particle()`.

## Kontext
Particle system: struct `ParticleSystem { emitters: Vec<Emitter>, rng_state: u64,
particles: Vec<Particle> }`. V `update(dt)` mám iterovat přes mutable emitters
a každý emitter, který má `time_since_emit > interval`, má spawnnout částici.
Spawn částice potřebuje (a) konfiguraci emitteru, (b) RNG state pro náhodnou
počáteční rychlost, (c) zápis do `self.particles`. Naivní `for emitter in
&mut self.emitters { let p = self.spawn_particle(&emitter.config); ... }`
borrow checker odmítl: `self.emitters` drží mutable borrow a `self.spawn_particle`
chce další mutable borrow na `self`. Klasický "cannot borrow `*self` as mutable
more than once at a time" error.

## Analytický flow

1. **První instinkt: dát si na chvíli pauzu, ne začít hledat hacky.** Borrow
   checker není šikana, je to compilerova druhá ruka, která mě vede ke
   správnému ownership designu. Když odmítne, není problém v "Rustu", ale
   v tom, jak strukturuju data a operace. Než sáhnu po `RefCell` nebo
   `unsafe`, podívám se, *co konkrétně* checker namítá.

2. **Diagnóza: dvě cesty k *self skrz nestejný field.** První borrow pochází
   z `for &mut self.emitters`. Druhý z `self.spawn_particle()`, což je metoda
   na `&mut self`. Rust nemůže vědět, že `spawn_particle` *nepoužívá*
   `self.emitters` (kdyby používal, vznikla by data race i logicky). Rust
   neví, jaké fields metoda čte/zapisuje — vidí jen `&mut self`. To je
   konzervativní, ale správné.

3. **Teď řešení: jak zúžit "co spawn_particle potřebuje" tak, aby to nebylo
   `&mut self`.** Podívám se, co metoda dělá: čte konfiguraci (předanou
   parametrem), generuje náhodné číslo z `self.rng_state`, vrací nově
   sestavenou částici. Klíčový pohled: **většina práce nepotřebuje self,
   potřebuje jen rng_state**. Konkretně:
   - `&Config` → parametr (immutable)
   - `&mut u64` (rng_state) → parametr (mutable jen ten field)
   - Návrat: `Particle` struct (žádný side effect)

4. **Refaktor: extrahuju `spawn_particle` jako *free function*** s explicit
   parametry:
   ```rust
   fn spawn_particle_with_rng(config: &Config, rng: &mut u64) -> Particle { ... }
   ```
   Tahle funkce nemá žádný `self`. Volá se jako `spawn_particle_with_rng(
   &emitter.config, &mut local_rng)`. Pak v `update`:
   ```rust
   let mut local_rng = self.rng_state;
   for emitter in &mut self.emitters {
       if emitter.time > emitter.interval {
           let p = spawn_particle_with_rng(&emitter.config, &mut local_rng);
           self.particles.push(p);  // ← pozor, teď druhý borrow
           ...
       }
   }
   self.rng_state = local_rng;
   ```
   Stále zbývá problém: `self.particles.push(p)` je další borrow na self.
   **Stejný pattern: vytáhnu particles z self lokálně.**

5. **Druhý refaktor: collect spawn results, pak push v jedné dávce.**
   ```rust
   let mut new_particles = Vec::new();
   let mut local_rng = self.rng_state;
   for emitter in &mut self.emitters {
       if emitter.time > emitter.interval {
           new_particles.push(spawn_particle_with_rng(&emitter.config, &mut local_rng));
           emitter.time = 0.0;
       }
   }
   self.rng_state = local_rng;
   self.particles.extend(new_particles);
   ```
   Borrow checker je spokojený: jeden mutable borrow na `self.emitters`
   uvnitř `for`, žádný další borrow na `self`. Po skončení loopu se borrow
   uvolní a můžu zapisovat do `self.rng_state` a `self.particles`.

6. **Cena? Jeden Vec<Particle> alokace per frame.** Pro typické particle
   counts (desítky až stovky per frame) je to triviální. Pro extrémní
   case (tisíce per frame) by se dal preallocate `Vec::with_capacity(estimated)`
   a reuse mezi framy přes `mem::take`. Ale tohle je optimalizace, ne
   architektura — začínám se simple variantou.

7. **Reflexe: borrow checker mě vlastně přiměl k lepšímu designu.**
   Free funkce `spawn_particle_with_rng` je teď testable bez celé
   `ParticleSystem` instance (předám config a u64, dostanu Particle).
   Logika spawnu je oddělená od orchestrace iterace. Když se objeví druhý
   typ emitteru s jiným spawnem, můžu mít `spawn_smoke_particle_with_rng`,
   `spawn_spark_particle_with_rng` — žádné nové metody, jen funkce.

## Aplikovatelné principy

- **Rust borrow checker vede ke správnému ownership designu, neobchází
  ho `RefCell`.** Pokud checker odmítne, má pravdu, že designově je něco
  divně. `RefCell` posune chybu z compile time do runtime — nezachrání
  bug, jen ho odsune. Lepší cesta: refactor, který compileru ukáže, že
  konflikt neexistuje.
- **"Method on `&mut self` zachycuje *celý* self."** Compiler nemůže vědět,
  že metoda potřebuje jen jeden field. Pokud mám konflikt, extrahuju metodu
  na free funkci s explicit field parametry. Funkce má pak striktnější
  signaturu, kterou může compiler verifikovat na fine-grained level.
- **"Collect now, apply later" pattern řeší borrow loops.** Když chci ve
  smyčce číst i zapisovat do téhož container, collectnu výsledky lokálně
  a apply až po skončení smyčky. Cena: jedna alokace. Výhoda: jasný
  oddělený "compute" a "commit" fáze, nezávislé borrow scopes.
- **Free funkce s explicit state jsou snazší k testování.** `fn(&Config,
  &mut u64) -> Particle` se otestuje s libovolným config a libovolným
  rng_state. Method on self vyžaduje vytvoření celé `ParticleSystem`
  instance. Test je menší, rychlejší, izolovanější.

## Závěr

```rust
// Před (nekompiluje):
impl ParticleSystem {
    fn update(&mut self, dt: f32) {
        for emitter in &mut self.emitters {
            if emitter.time > emitter.interval {
                let p = self.spawn_particle(&emitter.config);  // ❌ E0499
                self.particles.push(p);
                emitter.time = 0.0;
            }
            emitter.time += dt;
        }
    }
    fn spawn_particle(&mut self, c: &Config) -> Particle { ... }
}

// Po:
fn spawn_particle_with_rng(config: &Config, rng: &mut u64) -> Particle {
    let velocity = sample_velocity(config, rng);
    Particle { pos: config.origin, velocity, life: config.lifetime }
}

impl ParticleSystem {
    fn update(&mut self, dt: f32) {
        let mut new_particles = Vec::new();
        let mut local_rng = self.rng_state;
        for emitter in &mut self.emitters {
            if emitter.time > emitter.interval {
                new_particles.push(
                    spawn_particle_with_rng(&emitter.config, &mut local_rng),
                );
                emitter.time = 0.0;
            }
            emitter.time += dt;
        }
        self.rng_state = local_rng;
        self.particles.extend(new_particles);
    }
}
```

## Přenositelný pattern

Když Rust borrow checker odmítne metodu, postupuju touto sekvencí:

1. **Identifikuj, čeho se každý borrow dotýká.** Vypiš si, jaké fields
   jsou v každém scope mutable/immutable. Často konflikt vznikne mezi
   `&mut self` na úrovni metody a `&mut self.field` ve smyčce.

2. **Zkus refactor na free funkci s explicit field parameters.** Když
   metoda potřebuje jen `self.field_a` a `self.field_b`, ne celé self,
   napiš funkci `fn op(a: &mut FieldA, b: &mut FieldB)`. Compiler pak
   vidí přesně, co se borrowuje, a `disjoint borrows` jsou OK.

3. **Když to nejde, použij "collect now, apply later".** Vytvoř lokální
   buffer, naplň ho ve smyčce, applyuj po skončení smyčky. Cena: jedna
   alokace per call. Výhoda: jasné oddělení compute fáze a commit fáze.

4. **Pokud i to nejde, sáhni po `mem::take` / `mem::replace`.** Vytáhneš
   field do lokální variable, pracuješ s ním libovolně, po skončení vrátíš
   zpět. Funguje pro `Vec`, `HashMap` a další `Default` typy. Levné, čisté,
   nepotřebuje `unsafe`.

5. **Až jako poslední možnost zvaž `RefCell` nebo `unsafe`.** RefCell
   posouvá borrow check do runtime — mírná performance penalty + možný
   panic. `unsafe` ti dá libovolnou flexibilitu, ale ztrácíš compiler
   garance. Obvykle existuje *nějaká* z předchozích cest, jen je třeba
   ji najít.

Pattern se přenáší daleko za particle systémy. Game state update (oddělit
"compute new state" od "commit"), event handling (collect events, apply
batch), database transactions (build operations, commit at end), file
I/O (buffer writes, flush at boundary). Společný mental shape: **iterace
přes mutable věc nemůže současně mutovat jiné fields toho samého ownera.
Buď zúžit ownership (extract function), nebo oddělit compute od commit
(collect, apply later). Borrow checker odmítá to, co je obvykle i logicky
správné nedělat současně — důvěřuju mu.**
