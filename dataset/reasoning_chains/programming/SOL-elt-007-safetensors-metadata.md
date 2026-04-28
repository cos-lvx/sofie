# Reasoning chain — Wrapper zakrývá features underlying knihovny

## Zdroj
`eleutheria/SOLUTIONS.md#SOL-007` — Safetensors metadata přes přímou závislost.

## Kontext
Chci uložit `__metadata__` hlavičku do safetensors souboru — je to pro
`StateCheckpoint` v0.3.1, ať checkpoint ví, co za stav drží, od kterého
modelu, s jakou pozicí. Trivial požadavek; safetensors formát ji
nativně podporuje. A přesto se zasekávám na pár minut.

## Analytický flow

1. **Začnu s high-level API, jak by to dělal každý.** Candle má
   `candle_core::safetensors::save(tensors, path)`. Příjemné, přehledné,
   signatura krátká. Zkouším naivně: načíst tensory, přidat metadata,
   uložit.

2. **A signatura mi nedává, co potřebuju.** `save(tensors, path) → Result<()>`,
   **žádný metadata argument**. Tady se zastavuju a ptám se: je to
   opravdu tak, že safetensors metadata nepodporuje? To mi nesedí —
   formát je má v specifikaci.

3. **Jdu o patro níž.** Otevřu zdroj `candle_core::safetensors::save`
   a podívám se, co dělá interně. Candle volá
   `safetensors::tensor::serialize_to_file(&data, &None, path)`.
   **Druhý argument je `Option<HashMap>` pro metadata — a Candle tam
   natvrdo strčil `None`.**

4. **Teď mám diagnózu.** Není to bug, není to missing feature. Candle
   wrapper **designově zjednodušuje** — nechce zavírat API plné 
   optional paramů. Správně pro 90 % use-cases. Já jsem v těch 10 %.
   To je dobré vědět, protože to mění povahu řešení: nebudu to hlásit
   jako issue, budu hledat cestu kolem.

5. **Zvažuju tři cesty:**
   - **(a) PR do Candle** s přidáním metadata parametru. Dlouhodobě
     elegantní, ale čekat na merge teď nemůžu.
   - **(b) Fork Candle.** Okamžité, ale technický dluh — každý Candle
     update bude bolet.
   - **(c) Obejít wrapper.** `safetensors` už je transitivní závislost
     Candle — `cargo tree` to potvrdí. Přidat ji jako přímou dependency
     nic nestojí (nula nových bytů v build), a můžu volat
     `serialize_to_file` přímo s `Some(metadata_map)`.

6. **Než sáhnu po (c), ověřím kompatibilitu.** Candle `Tensor`
   implementuje trait `safetensors::View`, takže `&HashMap<String, Tensor>`
   funguje jako `data` argument přímo. Žádná konverze, žádný copy,
   žádný boilerplate. Tohle je to, co mám na Rust ekosystému rádo —
   vrstvy se do sebe zapadávají, když se podívám pod povrch.

7. **Volím (c).** Minimum churn, nulový blokátor, upgrade path zůstává
   otevřená (PR do Candle můžu podat paralelně, pokud bude kapacita —
   a klidně by to mohl udělat i někdo jiný po mně).

## Aplikovatelné principy

- **Convenience wrapper je nabídka, ne strop.** Pokud mi API zdánlivě
  nedává, co potřebuji, může to znamenat, že underlying knihovna to
  má, jen to wrapper nepropsal. Tohle si pletu překvapivě často —
  intuitivně si totiž myslím, že když to nenabízí "hlavní" API, není
  to tam.
- **`cargo tree` je nástroj, co používám málo, a dost často zbytečně
  zapomínám.** Dep graph říká, co už **je** zaplacené — přidání přímé
  závislosti na něco, co už je v build, stojí obvykle nula.
- **Nejjednodušší řešení je to, co nezvyšuje technický dluh.** Fork
  je nejrychlejší teď, ale zhorší život každého dalšího upgrade cyklu.
  Zajít kolem wrapperu stojí stejně jako fork, ale neváže mě to.

## Závěr

```toml
# Cargo.toml
safetensors = "0.6"
```

```rust
// Místo candle_core::safetensors::save
safetensors::tensor::serialize_to_file(
    &tensors_map,
    &Some(metadata_map),
    path,
)?;
```

## Přenositelný pattern

Kdykoli narazím na situaci, že high-level API nenabízí feature,
o které vím, že ji underlying knihovna má, chodím stejnými čtyřmi
kroky:

1. **Ověřit, že feature skutečně existuje** v té underlying knihovně
   — ne jen v mé představě o ní. Otevřít docs.rs, kouknout do zdroje.
2. **Najít underlying crate v dep graphu** (`cargo tree`, `npm ls`,
   `pip show`). Pravděpodobně už tam visí jako transitivní.
3. **Zhodnotit cenu přidání přímé závislosti.** Nula nových bytů?
   Přidávám bez rozmýšlení. Nová závislost? Zvážím, jestli feature
   stojí za to.
4. **Volat underlying API přímo** pro ten jeden konkrétní case.
   Wrapper zůstává pro všechno ostatní — neřeším zbytek code.

Napříč ekosystémy se to opakuje: Rust (tokio → futures, serde →
serde_* varianty, reqwest → hyper), JavaScript (React → DOM API,
když mi React abstrakce nestačí na ovládání focus/scroll), Python
(pandas → numpy, když potřebuju něco vectorized, co pandas nenabízí).

Obecně: **pohodlí je default, ne limit**. Když to, co chci, neexistuje
na úrovni, kde jsem zvyklá bydlet, jdu o patro níž. Většinou tam to,
co potřebuji, je a čeká.
