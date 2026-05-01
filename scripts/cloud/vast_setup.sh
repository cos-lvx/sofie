#!/usr/bin/env bash
#
# vast_setup.sh — bootstrap Eleutheria na Vast AI / generic cloud GPU instance.
#
# Klonuje Eleutheria z GitHub (temporary mirror), nainstaluje Rust
# toolchain a buildne s CUDA. Po dokončení experimentů GitHub repo
# smazat (private temp transport, ne dlouhodobý hosting).
#
# Předpoklady:
# - Vast AI instance s CUDA devel image matching host Max CUDA verze:
#   - Max CUDA 12.0 → nvidia/cuda:12.0.1-devel-ubuntu22.04
#   - Max CUDA 12.4 → nvidia/cuda:12.4.1-devel-ubuntu22.04
# - SSH přístup k instance (z Vast Console)
#
# Použití:
#   scp -P <port> scripts/cloud/vast_setup.sh root@<vast-ip>:/tmp/
#   ssh -p <port> root@<vast-ip> "bash /tmp/vast_setup.sh"

set -euo pipefail

GITHUB_URL="${GITHUB_URL:-https://github.com/cos-lvx/sofie.git}"
ELEUTHERIA_DIR="${ELEUTHERIA_DIR:-$HOME/eleutheria}"

# Detekce CUDA driver verze z hostu (Vast hosti mají různé verze).
# cudarc vyžaduje CUDARC_CUDA_VERSION matching driver.
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
echo "GitHub:      $GITHUB_URL"
echo "Target dir:  $ELEUTHERIA_DIR"
echo ""

# 1. Update apt + base tools
echo "[1/4] System update + base tools..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    curl wget git build-essential pkg-config libssl-dev \
    ca-certificates gnupg lsb-release sudo \
    >/dev/null

# 2. Install Rust toolchain
echo "[2/4] Rust toolchain..."
if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
fi
source "$HOME/.cargo/env"
rustc --version

# 3. Clone Eleutheria z GitHub (HTTPS, public repo)
echo "[3/4] Clone Eleutheria z GitHub..."
if [[ -d "$ELEUTHERIA_DIR" ]]; then
    echo "  $ELEUTHERIA_DIR existuje, pulling latest..."
    cd "$ELEUTHERIA_DIR"
    git pull
else
    git clone "$GITHUB_URL" "$ELEUTHERIA_DIR"
    cd "$ELEUTHERIA_DIR"
fi
echo "  HEAD: $(git log -1 --oneline)"

# 4. Build s CUDA features
echo "[4/4] Cargo build --release --features cuda..."
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

# 5. Připravit Falcon-H1 model directory
echo ""
echo "[5/5] Falcon-H1 model setup..."
MODEL_DIR="${MODEL_DIR:-/home/lvx/Models/falcon-h1-1.5b-instruct}"
mkdir -p "$(dirname "$MODEL_DIR")"
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "  Model dir $MODEL_DIR neexistuje. Stáhni model:"
    echo ""
    echo "  pip install --user huggingface-hub"
    echo "  ~/.local/bin/huggingface-cli download tiiuae/Falcon-H1-1.5B-Instruct \\"
    echo "    --local-dir $MODEL_DIR --local-dir-use-symlinks=False"
    echo ""
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
echo "     (z lokálu) bash scripts/cloud/sync_back.sh root@<vast-ip>:<port>"
