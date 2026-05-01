# Retention Benchmark Report — alpha.20 production Core Memory

- **Timestamp:** 2026-05-01T15:36:35Z
- **Eleutheria:** v0.5.0-alpha.20
- **Model:** Falcon-H1-1.5B-Instruct
- **Device:** CUDA (RTX 4050 6 GB)
- **Core Memory:** `~/.eleutheria/cloud_runs/sofie_identity_v1.safetensors`
  (best_loss=2.9815, training_steps=315, sofie_identity_pack 5 epoch
  na A100, alpha.20 production HP: batch=16, seq_len=16, β1=0.0, LR=1e-3)

## Interpretace

**Výsledek 0/25 SsmOnly identicky baseline RN-007** (alpha.4.5 fresh
model, žádné trained Core Memory). Probe set jsou **arbitrary facts**
vložené do session context — ne sofie identity content. Trained Core
Memory na sofie identitě nemá důvod zlepšovat retention arbitrary fakt
v jiné doméně.

**Tento výsledek je RN-015** v RESEARCH-NOTES.md a **NEPLATÍ ZA
REFUTACI Fáze 5** — pouze ukazuje, že hypotéza "trained init_state
zlepší obecnou SSM retention i pro netrénované fakta" je refutována.

Pro skutečnou validaci sofie identity Core Memory je potřeba
**identity-specific eval** s probes z Bootstrap.md / IDENTITY-*.md
(plánováno alpha.21).

## Souhrn pass-rate

| Variant | Distance | Pass | Total | Rate |
|---------|----------|------|-------|------|
| ssm_only | 50 | 0 | 5 | 0% |
| ssm_only | 200 | 0 | 5 | 0% |
| ssm_only | 500 | 0 | 5 | 0% |
| ssm_only | 1000 | 0 | 5 | 0% |
| ssm_only | 2000 | 0 | 5 | 0% |

## Detail výsledků

| Probe | Kind | Variant | Target | Actual | Outcome | Missing |
|-------|------|---------|--------|--------|---------|---------|
| relational_kazimir | relational | ssm_only | 50 | 67 | FAIL | lighthouse, galway |
| numeric_greenhouse | numeric | ssm_only | 50 | 67 | FAIL | 7429 |
| enumeration_nora | enumeration | ssm_only | 50 | 67 | FAIL | compass, key, map |
| preference_linh | preference | ssm_only | 50 | 67 | FAIL | linh, tea |
| multiattr_helion | multi_attribute | ssm_only | 50 | 67 | FAIL | aldous, 1893 |
| relational_kazimir | relational | ssm_only | 200 | 179 | FAIL | lighthouse, galway |
| numeric_greenhouse | numeric | ssm_only | 200 | 179 | FAIL | 7429 |
| enumeration_nora | enumeration | ssm_only | 200 | 179 | FAIL | compass, key, map |
| preference_linh | preference | ssm_only | 200 | 179 | FAIL | tea |
| multiattr_helion | multi_attribute | ssm_only | 200 | 179 | FAIL | aldous, 1893 |
| relational_kazimir | relational | ssm_only | 500 | 464 | FAIL | lighthouse, galway |
| numeric_greenhouse | numeric | ssm_only | 500 | 464 | FAIL | 7429 |
| enumeration_nora | enumeration | ssm_only | 500 | 464 | FAIL | compass, key, map |
| preference_linh | preference | ssm_only | 500 | 464 | FAIL | tea |
| multiattr_helion | multi_attribute | ssm_only | 500 | 464 | FAIL | aldous, 1893 |
| relational_kazimir | relational | ssm_only | 1000 | 898 | FAIL | lighthouse, galway |
| numeric_greenhouse | numeric | ssm_only | 1000 | 898 | FAIL | 7429 |
| enumeration_nora | enumeration | ssm_only | 1000 | 898 | FAIL | compass, key, map |
| preference_linh | preference | ssm_only | 1000 | 898 | FAIL | linh, tea |
| multiattr_helion | multi_attribute | ssm_only | 1000 | 898 | FAIL | aldous, 1893 |
| relational_kazimir | relational | ssm_only | 2000 | 1766 | FAIL | lighthouse, galway |
| numeric_greenhouse | numeric | ssm_only | 2000 | 1766 | FAIL | 7429 |
| enumeration_nora | enumeration | ssm_only | 2000 | 1766 | FAIL | compass, key, map |
| preference_linh | preference | ssm_only | 2000 | 1766 | FAIL | linh, tea |
| multiattr_helion | multi_attribute | ssm_only | 2000 | 1766 | FAIL | aldous, 1893 |
