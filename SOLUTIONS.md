# Solutions — Eleutheria

Znalostní báze vyřešených problémů pro budoucí referenci.

Formát: `SOL-NNN` s problémem, kořenovou příčinou, řešením a ponaučením.

---

## SOL-001 — F32 upcast pro numericky citlivé operace

- **Problém:** BF16 má jen 7 mantissa bitů → akumulace chyb v normalizaci a aktivacích
- **Řešení:** F32 dočasný výpočet pro: RmsNorm, SiLU, softplus, RoPE, temperature sampling
- **Ponaučení:** BF16 je skvělé pro storage a matmul, ale citlivé operace vždy F32

## SOL-002 — muP multiplikátory jako f64 konstanty

- **Problém:** Falcon-H1 vyžaduje Maximal Update Parametrization pro správný výstup
- **Řešení:** Konstanty aplikované přes `affine()`: embedding 5.66, lm_head 0.0195, atd.
- **Ponaučení:** muP je kritický — bez něj model generuje nesmysly

## SOL-003 — conv1d_step roll-left pattern

- **Problém:** Token duplikace při naivním přidávání do conv state
- **Řešení:** Roll state vlevo, zapiš nový token, konvoluce přes celý state (HF reference)
- **Ponaučení:** Vždy ověř state management proti referenční implementaci

## SOL-004 — Weight key audit před implementací

- **Problém:** Safetensors key names se liší mezi modely (např. `model.final_layernorm` vs `model.norm`)
- **Řešení:** Inspekce safetensors souboru před psaním weight loading kódu
- **Ponaučení:** 5 minut inspekce ušetří hodiny debugování

## SOL-005 — Segmentace in_proj výstupu pro Mamba-2

- **Problém:** in_proj output se dělí na [z/x/B/C/dt] segmenty s per-segment muP
- **Řešení:** Explicitní split + mup_vector [0.354, 0.25, 0.177, 0.5, 0.354]
- **Ponaučení:** Hybrid modely mají jemnou granularitu škálování — čti paper pečlivě

## SOL-006 — Debugging garbage output layer-by-layer

- **Problém:** Model generoval nesmysly po parallel prefill
- **Řešení:** Systematické porovnání výstupů vrstva po vrstvě s HF referencí
- **Ponaučení:** Chyba v normalizaci je multiplikativní — akumuluje se přes vrstvy.
  Vždy kontroluj normy jako první

## SOL-007 — Safetensors metadata přes přímou závislost

- **Problém:** `candle_core::safetensors::save()` hardcoduje `None` pro metadata
  hlavičku — nelze uložit `__metadata__` do safetensors souboru
- **Řešení:** Přidat `safetensors = "0.6"` jako explicitní závislost (už je transitivní
  přes candle-core, nula nových bytů) a volat `safetensors::tensor::serialize_to_file()`
  přímo s `Some(metadata_map)`. Candle `Tensor` implementuje `safetensors::View`,
  takže `&HashMap<String, Tensor>` funguje jako `data` argument
- **Ponaučení:** Candle wrappery jsou pohodlné, ale občas zakrývají features
  underlying crate. Vždy zkontroluj zdroj wrapperu, ne jen jeho API
