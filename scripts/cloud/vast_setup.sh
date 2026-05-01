#!/usr/bin/env bash
#
# vast_setup.sh — bootstrap Eleutheria na Vast AI / generic cloud GPU instance.
#
# Připojí instanci do Tailscale sítě (kde žije Forgejo), naklonuje
# Eleutheria repo, nainstaluje Rust toolchain a buildne s CUDA.
#
# Předpoklady:
# - Vast AI instance s CUDA 12.x devel image (ideálně
#   nvidia/cuda:12.4.1-devel-ubuntu22.04)
# - SSH přístup
# - Tailscale auth key (preauthorized, ephemeral) z
#   https://login.tailscale.com/admin/settings/keys
#
# Použití:
#   export TAILSCALE_AUTHKEY=tskey-auth-XXXXXXXXXX
#   curl -sSL <vast_setup_url> | bash
# nebo
#   scp scripts/cloud/vast_setup.sh root@<vast-ip>:/tmp/
#   ssh root@<vast-ip> "TAILSCALE_AUTHKEY=tskey-... bash /tmp/vast_setup.sh"

set -euo pipefail

if [[ -z "${TAILSCALE_AUTHKEY:-}" ]]; then
    echo "ERROR: TAILSCALE_AUTHKEY není nastaveno." >&2
    echo "Vygeneruj preauthorized ephemeral key:" >&2
    echo "  https://login.tailscale.com/admin/settings/keys" >&2
    exit 1
fi

FORGEJO_URL="${FORGEJO_URL:-https://git.nexus.lomsky.net/lvx/eleutheria.git}"
ELEUTHERIA_DIR="${ELEUTHERIA_DIR:-$HOME/eleutheria}"

# Detekce CUDA driver verze z hostu (Vast může mít různé verze podle
# hostů). cudarc vyžaduje CUDARC_CUDA_VERSION matching driver.
detect_cuda_version() {
    if command -v nvidia-smi &>/dev/null; then
        # nvidia-smi shows "CUDA Version: 12.0" v záhlaví
        local cuda_ver
        cuda_ver=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
        if [[ -n "$cuda_ver" ]]; then
            # 12.0 → 12000, 12.4 → 12040, 11.8 → 11080
            local major minor
            major=$(echo "$cuda_ver" | cut -d. -f1)
            minor=$(echo "$cuda_ver" | cut -d. -f2)
            printf '%d%03d\n' "$major" "$((minor * 10))"
        fi
    fi
}

echo "=========================================="
echo "Eleutheria cloud bootstrap"
echo "=========================================="
echo "Forgejo:       $FORGEJO_URL"
echo "Target dir:    $ELEUTHERIA_DIR"
echo "Tailscale key: ${TAILSCALE_AUTHKEY:0:20}..."
echo ""

# 1. Update apt + base tools
echo "[1/6] System update + base tools..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    curl wget git build-essential pkg-config libssl-dev \
    ca-certificates gnupg lsb-release sudo \
    >/dev/null

# 2. Install Tailscale
echo "[2/6] Tailscale install + connect..."
if ! command -v tailscale &>/dev/null; then
    curl -fsSL https://tailscale.com/install.sh | sh
fi

# Connect to Tailscale (ephemeral = node se smaže po stop)
tailscale up \
    --authkey="$TAILSCALE_AUTHKEY" \
    --ephemeral \
    --hostname="vast-eleutheria-$(hostname | tr -dc 'a-z0-9' | head -c 8)" \
    --accept-dns=true

echo "  Tailscale IP: $(tailscale ip -4)"
echo "  Hostname:     $(tailscale status --self --json | grep -o '"DNSName":"[^"]*' | cut -d'"' -f4 | head -1)"

# 3. Install Rust toolchain
echo "[3/6] Rust toolchain..."
if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
fi
source "$HOME/.cargo/env"
rustc --version

# 4. Clone Eleutheria from Forgejo (přes Tailscale)
echo "[4/6] Clone Eleutheria z Forgejo..."
if [[ -d "$ELEUTHERIA_DIR" ]]; then
    echo "  $ELEUTHERIA_DIR existuje, pulling latest..."
    cd "$ELEUTHERIA_DIR"
    git pull
else
    # Forgejo přes Tailscale — DNS jméno musí být resolvable
    # (Tailscale --accept-dns=true to zajistí)
    git clone "$FORGEJO_URL" "$ELEUTHERIA_DIR"
    cd "$ELEUTHERIA_DIR"
fi
echo "  HEAD: $(git log -1 --oneline)"

# 5. Build s CUDA features
echo "[5/6] Cargo build --release --features cuda..."
echo "  (toto může trvat 5-10 minut při prvním buildu)"
cd "$ELEUTHERIA_DIR"

# Detect host CUDA driver version a override cudarc CUDARC_CUDA_VERSION
# (lokálně je v .cargo/config.toml CUDARC_CUDA_VERSION=13010 pro CUDA 13.2;
# Vast hosti mají typicky CUDA 12.0/12.4 — musíme matchnout host driver).
CUDA_VER_NUM=$(detect_cuda_version)
if [[ -n "$CUDA_VER_NUM" ]]; then
    echo "  Detekována CUDA driver: $CUDA_VER_NUM (z nvidia-smi)"
    export CUDARC_CUDA_VERSION="$CUDA_VER_NUM"
    echo "  CUDARC_CUDA_VERSION=$CUDARC_CUDA_VERSION"
else
    echo "  WARN: nemohu detekovat CUDA verzi, default ze .cargo/config.toml"
fi

cargo build --release --features cuda 2>&1 | tail -20

# 6. Připravit Falcon-H1 model directory
echo "[6/6] Falcon-H1 model setup..."
MODEL_DIR="${MODEL_DIR:-/home/lvx/Models/falcon-h1-1.5b-instruct}"
mkdir -p "$(dirname "$MODEL_DIR")"
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "  Model dir $MODEL_DIR neexistuje. Stáhni model:"
    echo ""
    echo "  pip install huggingface-hub"
    echo "  huggingface-cli download tiiuae/Falcon-H1-1.5B-Instruct \\"
    echo "    --local-dir $MODEL_DIR --local-dir-use-symlinks=False"
    echo ""
    echo "  (~13 GB, 3-5 min na 879 Mbps Vast bandwidth)"
else
    echo "  Model existuje: $(du -sh "$MODEL_DIR" | cut -f1)"
fi

echo ""
echo "=========================================="
echo "Setup hotov."
echo "=========================================="
echo ""
echo "Další kroky:"
echo "  1. Stáhnout Falcon-H1 model (viz výše)"
echo "  2. Spustit training:"
echo "     cd $ELEUTHERIA_DIR"
echo "     bash scripts/cloud/vast_train.sh"
echo ""
echo "  3. Po skončení sync výstupů zpět na local:"
echo "     (z lokálu) bash scripts/cloud/sync_back.sh <vast-tailscale-ip>"
