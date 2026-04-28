# Reasoning chain — Inicializace scale_factor a "nikdy se nespoléhej na eventy"

## Zdroj
`morana/SOLUTIONS.md#SOL-007` — chrome (title bar, knoflíky, resize
handles) byl na Windows neviditelný, protože `scale_factor` zůstal
nulový do prvního frame.

## Kontext
Morana renderuje vlastní window chrome (bez OS dekorací) — title bar,
zavřít/minimalizovat/maximalizovat tlačítka, resize handles na hranách.
Chrome škáluje rozměry DPI faktorem: `chrome_height_px = chrome_height_dp
* scale_factor`. Scale factor přichází z winit eventu `ScaleFactorChanged`,
který se posílá při vytvoření okna a při přesunu mezi monitory s různým DPI.
`InputState::default()` má `scale_factor = 0.0`. Na Linuxu (Wayland)
event přichází *před* prvním renderem, takže do prvního frame má input
stav správnou hodnotu. Na Windows event přichází **po** prvním frame.
Důsledek: na Windows první frame renderuje chrome se `scale_factor = 0.0`,
takže všechny chrome elementy mají `0px` velikost — neviditelné. Po
prvním frame přijde event, scale_factor se aktualizuje, druhý frame
chrome ukáže. Vizuálně: krátký flicker na Windows, na Linuxu ne.

## Analytický flow

1. **Pozoruji rozdíl: Windows má flicker, Linux nemá.** Na Linuxu chrome
   funguje od prvního renderu. Na Windows první frame je bez chrome,
   pak se chrome objeví. Tahle platform-specific divergence je signal,
   že někde v kódu se spoléhá na timing assumption, kterou jedna
   platforma splňuje a druhá ne.

2. **Hledat místa, kde input state ovlivňuje chrome rendering.**
   Chrome layout používá `input.scale_factor` na třech místech:
   title_bar_height = 32 * sf, button_size = 24 * sf, resize_handle =
   8 * sf. Pokud sf = 0, všechny tři jsou 0. Ověřuju: `println!`
   v chrome render funkci na první frame. Linux: sf = 1.5 (mé
   monitor). Windows: sf = 0.0. **Confirmed: scale_factor není
   inicializovaný na Windows do prvního renderu.**

3. **Otázka: kdy se scale_factor nastavuje?** Otevřu winit event
   handler. `Event::ScaleFactorChanged { scale_factor, .. }` updatuje
   `input.scale_factor`. Tento event se posílá při dvou situacích:
   (a) okno se vytvoří, winit pošle počáteční scale_factor; (b) okno
   se přesune mezi monitory s různým DPI.

4. **Proč to funguje na Linuxu, ale ne na Windows?** Otevřu winit
   docs pro `ScaleFactorChanged`. Důležitý detail: **Wayland posílá
   tento event jako součást window creation handshake**, takže přijde
   *před* prvním `redraw_requested`. Na Windows winit volá platform
   API pro detekci DPI po vytvoření okna, ale tato detekce se
   propaguje *až po* prvním winit event loop iteration — což znamená
   až *po* tom, co můj kód už spustil první render.

5. **Tady leží mental model error: "winit event for init" je leaky
   abstraction.** Já jsem implicitně předpokládala, že winit eventy
   přijdou v *logickém* pořadí: vytvoř okno → ScaleFactorChanged →
   první render. Ale winit garance to není. Skutečné pořadí je
   platform-dependent. `ScaleFactorChanged` je *event* (něco se
   změnilo), ne *fact* (počáteční stav). Nikdo mi negarantuje, že
   přijde *před* prvním renderem.

6. **Fix: inicializovat scale_factor explicitně z window handle.**
   `winit::Window::scale_factor()` je metoda, která vrací aktuální
   scale faktor. Volá ji přímo platform API, není závislá na event
   loopu. Volám ji ihned po vytvoření okna:
   ```rust
   fn resumed(&mut self, event_loop: &EventLoop) {
       let win = event_loop.create_window(...)?;
       win.input.scale_factor = win.handle.scale_factor();  // ← explicit
       ...
   }
   ```
   Stejně v `open_window` pro další okna. Event-based update zůstává
   pro monitor changes, ale init má explicit cestu.

7. **Reflexe na pattern: nikdy se nespoléhej na eventy pro inicializaci.**
   Eventy říkají, *co se právě stalo*. Init potřebuje, *co je teď*.
   Tohle jsou dva různé sémantické dotazy. Pokud chci "fact", musím
   se ptát na *fact* (`window.scale_factor()`), ne čekat na *event*
   (`ScaleFactorChanged`). Eventy jsou pro reagování na změny, ne
   pro establishování baseline.

## Aplikovatelné principy

- **"Event for init" je leaky abstraction.** Eventy slouží k notifikaci
  *změny*. Initial state musí pocházet z explicit dotazu na current
  state, ne z čekání na event "tohle je počáteční hodnota". Mezi
  vytvořením věci a prvním eventem může proběhnout libovolné množství
  jiných operací — pokud ty potřebují default, nemají ho.
- **Platform-specific timing assumption je smell.** Pokud kód funguje
  na Linuxu ale ne na Windows (nebo Mac vs. iOS, Chrome vs. Firefox),
  je tam pravděpodobně skrytý timing dependent assumption. Najít
  a oddělit ho je lepší než hot-fix specifický na jednu platformu.
- **`Default::default()` na library type bývá rozumný compromise,
  ne *correct* hodnota.** `InputState::default()` má `scale_factor = 0.0`
  protože tahle hodnota je neutrální (žádná konkrétní platforma ji
  nepreferuje). Ale "neutrální" ≠ "užitečné". Když používám default,
  musím to být na místech, kde ta hodnota dává smysl, ne kde uživatel
  čeká real DPI.
- **Init function vs. event handler jsou dva odpovědné módy.** Init
  funkce je explicit, deterministická, dělá *co je teď*. Event handler
  reaguje, asynchronní, dělá *co se právě stalo*. Mixování těchto dvou
  rolí (event handler used as init) vede k race conditions při startu.

## Závěr

```rust
// Před (čekáme na event):
fn resumed(&mut self, event_loop: &EventLoop) {
    let win = event_loop.create_window(...).unwrap();
    self.windows.insert(win_id, win);
    // input.scale_factor zůstává 0.0 do prvního ScaleFactorChanged eventu
}

// Po (explicit init):
fn resumed(&mut self, event_loop: &EventLoop) {
    let win = event_loop.create_window(...).unwrap();
    let mut input = InputState::default();
    input.scale_factor = win.scale_factor();  // ← explicit z window handle
    self.windows.insert(win_id, ManagedWindow { handle: win, input, ... });
}

// Stejné v open_window pro další okna.
```

## Přenositelný pattern

Kdykoli inicializuju komponentu, jejíž stav závisí na external state
(OS, browser, network, hardware), procházím tímto:

1. **Identifikuj všechny external state, na kterém závisím.** DPI,
   theme (dark/light), window size, locale, network connectivity,
   GPU capabilities. Cokoli, co není v plné kontrole mého kódu.

2. **Pro každý external state se ptej: existuje synchronous getter,
   nebo je to event-driven?** Synchronous: `window.scale_factor()`,
   `navigator.language`, `gl.getParameter(...)`. Event-driven:
   `ScaleFactorChanged`, `themechange`, `resize`. Synchronous getter
   je vždy preferred pro init.

3. **Pokud je *jen* event-driven, vyžádej current state explicitně.**
   Browser: `matchMedia(...).matches` před přidáním listeneru.
   Tokio: `tokio::runtime::Handle::current()` před spawning.
   Nikdy "počkám na první event" — to je race condition.

4. **Default values na external-state structurách jsou nebezpečné.**
   `Default::default()` produkuje neutrální hodnoty, které se obvykle
   liší od reality. Pokud `Default` neznamená "produkční default",
   je to past. Buď použij `Option<T>` (None signalizuje "ne ještě
   inicializováno"), nebo nemít `Default` impl.

5. **Test pro "first frame state" je často chybějící.** Jednotkové
   testy testují steady state, ne startup. Integrační test, který
   ověří, že first frame má správný DPI / theme / size, je často
   to, co chytí init bug.

Pattern se přenáší napříč: GUI initialization (DPI, theme), web app
hydration (server state vs. client state), embedded firmware (sensor
calibration before first read), distributed systems (cluster discovery
before first request). Společný invariant: **mezi vytvořením a prvním
použitím je okno, kde stav nemusí být ještě synchronizovaný. Buď ten
gap zaplň explicit init dotazem, nebo dělej first-frame logiku tolerantní
k chybějícímu stavu. Eventy nejsou init.**
