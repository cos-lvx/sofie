# Reasoning chain — Windows borderless resize a FFI escape hatch

## Zdroj
`morana/SOLUTIONS.md#SOL-008` — borderless okno na Windows (vlastní
chrome) odmítlo resize přes drag, na Linuxu fungovalo.

## Kontext
Morana má vlastní GPU-renderovaný window chrome (title bar, knoflíky,
resize handles) — záměr je pixel-perfect cross-platform vzhled,
nezávislý na OS dekoracích. Implementace: `with_decorations(false)`
v winit + vlastní hit testing (resize handles na hranách okna). Když
uživatel kliká na resize handle a táhne, voláme `window.drag_resize_window
(direction)` — winit posílá platform-specific resize message. **Na
Linuxu (Wayland) to funguje** — Wayland compositor zpracovává resize
přes vlastní protokol, nepotřebuje žádné zvláštní window flags. **Na
Windows to nefunguje** — kurzor se nemění, drag nedělá nic, okno se
nehne. Žádný error, žádný panic, jenom tichý no-op.

## Analytický flow

1. **Diagnostika: Windows vs. Linux divergence znovu.** Stejný kód,
   různé chování. To znamená, že někde je platform abstraction, která
   *vypadá* uniformně (`drag_resize_window` API), ale platform-specific
   implementace dělá jiné věci. Otevřu winit zdroj, hledám
   `drag_resize_window` Windows implementaci.

2. **Co dělá winit na Windows pro resize.** `drag_resize_window`
   posílá `WM_NCLBUTTONDOWN` message s `HTBOTTOMRIGHT` (nebo other
   resize hit-test code). Windows reaguje na tuhle message standardní
   resize logikou. **Pokud:** okno má `WS_THICKFRAME` style. Pokud
   ne, Windows tuto message ignoruje. Tichý no-op.

3. **Otázka: má moje okno `WS_THICKFRAME`?** winit s
   `with_decorations(false)` vytváří okno jako `WS_POPUP` (popup style,
   bez title baru a borders). `WS_POPUP` **neobsahuje `WS_THICKFRAME`**.
   To je rozumný default — popup okna by neměla být resizable. Ale
   já potřebuju popup okno (žádné dekorace) *plus* resize support.
   Win32 nemá tuto kombinaci out-of-the-box, je třeba ji ručně
   poskládat.

4. **Cesta č. 1: požádat winit o nový API option.** "Borderless plus
   resizable" je rozumný feature request. Otevřu winit issue tracker,
   vidím, že tato kombinace je dlouhodobě požadovaná, ale otevřená
   diskuze o správné API. Nemůžu blokovat svoje milestone na winit
   merge.

5. **Cesta č. 2: workaround přes FFI.** Windows má `SetWindowLongPtrW`,
   která umožňuje upravit window style po vytvoření. Mohu po winit
   vytvoření okna získat raw HWND handle a přidat `WS_THICKFRAME`
   manuálně. Cenu: minimal Win32 FFI volání (definované inline
   bez externího crate). Risk: změny v internal winit detail by
   to mohly rozbít, ale `raw-window-handle` crate poskytuje stabilní
   way k získání HWND.

6. **Implementace: `enable_borderless_resize()` cross-platform funkce.**
   Na Linux/Mac je no-op (žádné FFI volání potřeba). Na Windows:
   ```rust
   #[cfg(windows)]
   fn enable_borderless_resize(window: &Window) {
       use raw_window_handle::HasWindowHandle;
       let handle = window.window_handle().unwrap();
       let hwnd = match handle.as_raw() {
           RawWindowHandle::Win32(h) => h.hwnd.get() as HWND,
           _ => return,
       };
       unsafe {
           let style = GetWindowLongPtrW(hwnd, GWL_STYLE);
           let new_style = style | (WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX) as isize;
           SetWindowLongPtrW(hwnd, GWL_STYLE, new_style);
           SetWindowPos(hwnd, ptr::null_mut(), 0, 0, 0, 0,
               SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER);
       }
   }
   ```
   `WS_THICKFRAME` přidává resize support, `WS_MINIMIZEBOX |
   WS_MAXIMIZEBOX` pak zachovává funkčnost minimum/maximum gestures.
   `SWP_FRAMECHANGED` říká Windows "přepočítej non-client area" —
   bez toho by změna stylu se neaplikovala visually.

7. **Vedlejší efekt: tenký DWM resize border (~1px).** Windows DWM
   (desktop window manager) přidává malou aero-style hairline border
   k oknu se `WS_THICKFRAME`. Vizuálně téměř neviditelný, ale
   technicky tam je. Acceptable trade-off pro získání resize support.
   Alternativa by byla custom mouse capture + `SetWindowPos` během
   drag, ale to by replikovalo logiku, kterou Windows má built-in
   a kterou moji uživatelé očekávají (snap to edge, snap to corner,
   atd.).

8. **Reflexe: kdy je FFI escape hatch správná volba.** Tady je: API,
   které potřebuju (borderless + resizable), neexistuje v abstraction
   library; existuje na underlying platformě jako standard pattern;
   FFI volání je malé, dobře ohraničené, well-documented. Když
   underlying capability je tam a wrapper to nepropsal, je legitimní
   sáhnout dolů. (Stejný princip jako safetensors metadata workaround,
   jen s OS API místo Rust crate.)

## Aplikovatelné principy

- **Cross-platform abstrakce mají gaps. Wayland/X11/Win32/Cocoa mají
  fundamentálně různé event models.** Když API knihovna říká "drag
  resize", ne všechny platformy ten koncept mají identicky. Podpora
  borderless+resizable je velmi platform-specific.
- **`raw-window-handle` crate je standardní way k FFI escape.** Místo
  vlastního winit-internal-deps poskytuje stabilní abstrakci pro
  získání platform-specific handle (HWND na Windows, NSWindow na Mac,
  XCB na Linux). Pokud potřebuju FFI, jdu přes raw-window-handle.
- **`SetWindowPos(SWP_FRAMECHANGED)` je nutný invalidace po style
  change.** Windows cachuje window non-client area shape — bez
  invalidate by změna stylu nepropagovala. Podobně CSS reflow a
  invalidation u browser layoutů.
- **Tenký aero border je acceptable trade-off pro real Windows resize
  semantics.** Reimplementace OS resize logic (snap, magnetic edges,
  aero peek) by byla obrovská práce. Trade ~1px aero hairline za
  full Windows resize compliance je dobrý deal.

## Závěr

```rust
// Cross-platform stub
fn enable_borderless_resize(_window: &Window) {
    #[cfg(windows)]
    enable_borderless_resize_win32(_window);
}

#[cfg(windows)]
fn enable_borderless_resize_win32(window: &Window) {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use windows::Win32::Foundation::HWND;
    use windows::Win32::UI::WindowsAndMessaging::*;

    let handle = window.window_handle().unwrap();
    let hwnd = match handle.as_raw() {
        RawWindowHandle::Win32(h) => HWND(h.hwnd.get() as *mut _),
        _ => return,
    };
    unsafe {
        let style = GetWindowLongPtrW(hwnd, GWL_STYLE);
        let new_style = style
            | (WS_THICKFRAME.0 | WS_MINIMIZEBOX.0 | WS_MAXIMIZEBOX.0) as isize;
        SetWindowLongPtrW(hwnd, GWL_STYLE, new_style);
        let _ = SetWindowPos(
            hwnd, None, 0, 0, 0, 0,
            SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER,
        );
    }
}

// V resumed():
let win = event_loop.create_window(attrs.with_decorations(false))?;
enable_borderless_resize(&win);
```

## Přenositelný pattern

Kdykoli moje cross-platform knihovna nesplňuje požadovaný platform-specific
chování, procházím:

1. **Ověř, že underlying platform tu schopnost má.** Windows API,
   Cocoa, X11/Wayland — pokud na native úrovni to existuje jako
   standard pattern, mám reálnou cestu. Pokud ne, FFI mě nezachrání.

2. **Hledej standardní way přes `raw-window-handle` (nebo equivalent).**
   Místo vlastního FFI s winit internals, použij stabilní abstrakci.
   Na Rust side je to `raw-window-handle` crate; na Cocoa/Swift
   `NSView -> objc2_app_kit::NSWindow`; na web `WebGL -> canvas`
   element directly.

3. **Vykresli si platform matrix: kde to už funguje, kde nepotřebuje
   workaround.** Linux Wayland je často "naturally working" — bash
   neexistuje, X11 má vlastní mechanismus. Mac má Cocoa s vlastními
   patterny. Windows s Win32 je často ten, kdo potřebuje FFI escape.
   Workaround psát jen na platformách, kde to vyžaduje.

4. **Minimální FFI surface, dobře ohraničený.** Jedna funkce, jeden
   import set, jedno well-documented place. Když later upgrade
   knihovny obsoletizuje workaround, je to jednoduché smazat.

5. **Acceptable trade-off vs. perfect parity.** Někdy 99% parity
   stojí 5 % code. Posledních 1 % parity stojí 50 % code. Po určité
   úrovni je správná odpověď "pixelové rozdíly mezi platformami
   jsou OK", ne reimplementace native logiky.

Pattern se přenáší: cross-platform crypto (OS keychain access),
filesystem (case-sensitivity, path separators), audio (PortAudio vs.
CoreAudio), networking (raw sockets, IP_TOS). Vždycky existuje **vrstva
abstrakce**, která pokrývá 80 %, a **vrstva platform native API**,
která pokrývá zbytek. FFI escape hatch je legitimní, když je dobře
ohraničený a vede k underlying capability, ne k re-implementaci.
