# Cloud GPU deployment workflow

> Cílový provider: Vast AI (single-GPU s ≥16 GB VRAM, ideálně 48+ GB).
> Funguje i na DigitalOcean, Lambda Labs, Runpod, atd.

## Předpoklady

1. **GitHub mirror** Eleutheria repa (temporary transport)
   - Default: `https://github.com/cos-lvx/sofie.git`
   - Po dokončení experimentů smazat (private temp, ne dlouhodobý hosting)
   - Push: `git -C /home/lvx/Dev/eleutheria push github main`

2. **Vast AI account + credit** ($5-20 stačí na první runs)
   - Registrace: <https://cloud.vast.ai/>

3. **SSH key** registrovaný ve Vast AI dashboard

## Doporučené Vast AI templates

**KRITICKÉ:** Vast AI nabídka má pole **"Max CUDA"** = host driver
verze. Template (Docker image) MUSÍ mít CUDA toolkit ≤ Max CUDA, jinak
build selže s "CUDA driver too old".

V Vast Console při výběru instance zkontroluj **Max CUDA: X.Y**, pak
vyber template odpovídající:

| Host Max CUDA | Template image |
|---------------|----------------|
| 12.0 | `nvidia/cuda:12.0.1-devel-ubuntu22.04` |
| 12.1-12.3 | `nvidia/cuda:12.1.1-devel-ubuntu22.04` |
| 12.4+ | `nvidia/cuda:12.4.1-devel-ubuntu22.04` |
| 13.x | `nvidia/cuda:13.0.0-devel-ubuntu22.04` |

`vast_setup.sh` automaticky detekuje host CUDA verzi z `nvidia-smi` a
nastaví `CUDARC_CUDA_VERSION` env var pro cudarc. **Pokud build selže,
zkontroluj že template CUDA toolkit ≤ host driver Max CUDA.**

NEvybírat:
- `runtime` varianty (chybí nvcc, candle build selže)
- CUDA 11.x (cudarc 0.18.2 nesupportuje)
- `slim` varianty (chybí dev tools)

## Doporučené HW (pro alpha.20 production training)

Z RN-012 víme: 16 GB stačí na batch=4 seq_len=8, 48 GB umožní batch=32+
(test RN-012 hypotézy o gradient noise).

| Varianta | VRAM | Cena/h | Vhodné na |
|----------|------|--------|-----------|
| RTX 4000 / A4000 | 16-20 GB | $0.5-0.8 | Single training, 1 epoch |
| RTX A6000 / RTX 6000 Ada | 48 GB | $0.8-1.5 | Batch ablace, paralelní HP |
| A100 40-80 GB | 40-80 GB | $1.5-3 | Production-grade, fast |
| **6× A4000 (96 GB total)** | 6×16 GB | $1.0 | Paralelní experimenty (6 procesů simultaneously) |

## Workflow

### Krok 1 — Provision Vast instance

V Vast Console:
1. Vyber instance dle tabulky výše
2. Template: `nvidia/cuda:12.4.1-devel-ubuntu22.04`
3. Storage: 100+ GB (model 13 GB + build artifacts + workspace)
4. Launch + počkej na running state
5. Zkopíruj **SSH command** z dashboardu (např. `ssh -p 12345 root@1.2.3.4`)

### Krok 2 — Bootstrap (na lokálu)

```bash
# Tailscale auth key
export TAILSCALE_AUTHKEY=tskey-auth-XXXXXXXXXX

# Upload setup script + spustit
scp -P <port> scripts/cloud/vast_setup.sh root@<vast-ip>:/tmp/
ssh -p <port> root@<vast-ip> \
    "TAILSCALE_AUTHKEY=$TAILSCALE_AUTHKEY bash /tmp/vast_setup.sh"
```

Setup udělá:
- Install Rust + build tools
- Install Tailscale + connect přes auth key
- Git clone Eleutheria z Forgejo přes Tailscale
- Cargo build --release --features cuda

Trvá ~10-15 min (z toho 5-10 min build).

### Krok 3 — Stáhnout Falcon-H1 model

Na Vast instanci (po setup):

```bash
ssh -p <port> root@<vast-ip>
pip install --user huggingface-hub
~/.local/bin/huggingface-cli download tiiuae/Falcon-H1-1.5B-Instruct \
    --local-dir /home/lvx/Models/falcon-h1-1.5b-instruct \
    --local-dir-use-symlinks=False
```

Trvá 3-5 min (~13 GB / 879 Mbps Vast).

### Krok 4 — Spustit training

```bash
ssh -p <port> root@<vast-ip>
cd ~/eleutheria

# Default = alpha.20 HP (LR=1e-3, β1=0, save-best)
bash scripts/cloud/vast_train.sh

# Override pro experimenty:
BATCH=16 SEQ_LEN=16 NOTES="batch ablation 16x16" \
    bash scripts/cloud/vast_train.sh
```

Estimate (1.5B model, sofie_identity_pack ~29k tokens):
- 16 GB GPU, batch=4 seq=8 grad_accum=2: **~6-8h/epoch**
- 48 GB GPU, batch=16 seq=16: **~2-3h/epoch**
- 80 GB GPU, batch=32 seq=32: **~1-2h/epoch**

### Krok 5 — Sync výstupů zpět (na lokálu)

```bash
# Z lokální mašiny (přes Tailscale)
bash scripts/cloud/sync_back.sh <vast-tailscale-hostname>

# Najdeš výstupy v ~/.eleutheria/cloud_runs/
ls -la ~/.eleutheria/cloud_runs/
```

### Krok 6 — Validace lokálně

```bash
# Inspect artefakt
cargo run --release --features cuda -- \
    --inspect-core-memory ~/.eleutheria/cloud_runs/sofie_identity_v1.safetensors

# Retention benchmark
cargo run --release --features cuda -- --cuda \
    --core-memory ~/.eleutheria/cloud_runs/sofie_identity_v1.safetensors \
    bench-retention --variant ssm_only \
    --output /tmp/bench_after_alpha20

# REPL kvalitativní test
cargo run --release --features cuda -- --cuda \
    --core-memory ~/.eleutheria/cloud_runs/sofie_identity_v1.safetensors
```

### Krok 7 — Stop Vast instance

V Vast Console: **Stop** nebo **Destroy**.

- **Stop:** šetří compute náklady, drží data + IP. $0.10-0.20/h za storage.
- **Destroy:** smaže vše. Model a buildy znovu provision. Pro ad-hoc experimenty.

Tailscale ephemeral node se automaticky smaže ~1h po stop.

## Backup řešení: Git bundle (offline)

Pokud Tailscale není možnost:

```bash
# Lokálně:
cd /home/lvx/Dev/eleutheria
git bundle create /tmp/eleutheria.bundle --all

# Upload na Vast:
scp -P <port> /tmp/eleutheria.bundle root@<vast-ip>:/tmp/

# Na Vast:
git clone /tmp/eleutheria.bundle ~/eleutheria
cd ~/eleutheria && git checkout main
```

Pak skip Tailscale step v setup, ostatní stejné.

Pro výstupy zpět: `scp` místo rsync, nebo `tar | base64` přes ssh.

## Cost optimization tipy

1. **Snapshot po prvním setupu** — Vast umožňuje vytvořit instance image
   po prvním successful build. Další provision z image = skip 10 min setup.
2. **Spot/Bid instance** — 50-70 % cheaper, ale interruptible. Pro
   ablation experimenty s `--save-best` je to OK (snapshot zachová best
   point i při interruption).
3. **Ephemeral storage** — Vast má levnější "ephemeral" disky (smazané
   po stop). Pro experimenty kde sync zpět hned, ne dlouhodobé hold.
4. **Bandwidth audit** — model download 13 GB = $0.65 (Vast accounts
   bandwidth). Snapshot model do persistent storage šetří $/run.

## Bezpečnost

- **Tailscale auth key:** preauthorized + ephemeral + 24h TTL.
  Ne reusable přes víc instancí (jeden key per instance ideálně).
- **SSH:** Vast instance je `root` accessible. Po skončení **destroy**,
  ne jen stop. Data nesmí zůstat na cizím stroji.
- **Forgejo credentials:** repo je public-clone přes Tailscale (žádné
  credentials), takže neuniknou. Pokud by byl repo private,
  použít deploy key, ne osobní SSH key.
- **Trained Core Memory artefakt:** `sofie_identity_v1.safetensors`
  obsahuje destilovanou Sofie identitu. Po skončení **explicitně
  stáhnout zpět + smazat z cloud instance** (rm + destroy).
