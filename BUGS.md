# Bugs — Eleutheria

Aktivní bugy k opravě.

Formát: `BUG-NNN` s verzí nálezu, závažností (P1–P4), reprodukcí a stavem.

---

## Aktivní

_Žádné aktivní bugy._

---

## Vyřešené

### BUG-008 — BF16 temperature sampling crash ✓
- **Nalezeno:** v0.2.0 | **Závažnost:** P1
- **Reprodukce:** Jakýkoliv generate s temperature > 0 na CUDA
- **Příčina:** BF16 tensor ÷ f64 scalar není podporováno v Candle
- **Řešení:** Cast logits na F32 před dělením teplotou
- **Opraveno:** v0.2.0

### BUG-007 — RmsNormGated invertovaná logika ✓
- **Nalezeno:** v0.1.0 | **Závažnost:** P1
- **Reprodukce:** Model generuje garbage po ~5 tokenech
- **Příčina:** Gate/norm pořadí obrácené pro `norm_before_gate=false` — chyba se akumuluje přes 24/44 vrstev
- **Řešení:** Swap větví dle HF referenční implementace
- **Opraveno:** v0.1.0

### BUG-006 — Parallel prefill garbage output ✓
- **Nalezeno:** v0.1.0 | **Závažnost:** P1
- **Příčina:** Důsledek BUG-007
- **Opraveno:** v0.1.0

### BUG-005 — Špatný final norm key ✓
- **Nalezeno:** v0.1.0 | **Závažnost:** P1
- **Příčina:** `model.norm` místo `model.final_layernorm`
- **Opraveno:** v0.1.0

### BUG-004 — mlp_multipliers aplikovány na gate i up ✓
- **Nalezeno:** v0.1.0 | **Závažnost:** P2
- **Příčina:** `mlp_multipliers[0]` použit pro obě projekce místo jen gate
- **Opraveno:** v0.1.0

### BUG-003 — Chybějící mup_vector na in_proj segmentech ✓
- **Nalezeno:** v0.1.0 | **Závažnost:** P1
- **Příčina:** muP vektor nebyl aplikován na [z/x/B/C/dt] segmenty
- **Opraveno:** v0.1.0

### BUG-002 — ssm_in_multiplier na špatném místě ✓
- **Nalezeno:** v0.1.0 | **Závažnost:** P2
- **Příčina:** Aplikováno PŘED in_proj místo PO
- **Opraveno:** v0.1.0

### BUG-001 — Token duplikace v conv1d_step ✓
- **Nalezeno:** v0.1.0 | **Závažnost:** P2
- **Příčina:** State cat zahrnul token dvakrát
- **Řešení:** Roll-left + append pattern
- **Opraveno:** v0.1.0
