#!/usr/bin/env bash
#
# detect-cuda.sh — detekuje host CUDA toolkit verzi a vypíše export
# command pro CUDARC_CUDA_VERSION.
#
# Použití:
#   source scripts/detect-cuda.sh
#   # nebo:
#   eval "$(scripts/detect-cuda.sh --export-command)"
#   # nebo jen pro audit:
#   scripts/detect-cuda.sh --report
#
# Detekuje přes `nvcc --version` (preferred, autoritativní) → `nvidia-smi`
# (driver report) → fail. Mapuje na cudarc 0.18.x supported verze
# (max 13.1 → 13010).
#
# cudarc je zpětně kompatibilní — pokud host má CUDA 13.2, exportujeme
# 13010 (cudarc max), build by měl projít.

set -euo pipefail

# ---------------------------------------------------------------------
# Detekční funkce
# ---------------------------------------------------------------------

detect_via_nvcc() {
    if ! command -v nvcc >/dev/null 2>&1; then
        return 1
    fi
    # "Cuda compilation tools, release 13.0, V13.0.48"
    nvcc --version 2>/dev/null \
        | grep -oP 'release \K[0-9]+\.[0-9]+' \
        | head -1
}

detect_via_nvidia_smi() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 1
    fi
    # "| NVIDIA-SMI 580.159.03  Driver Version: 580.159.03  CUDA Version: 13.0 |"
    nvidia-smi 2>/dev/null \
        | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' \
        | head -1
}

# ---------------------------------------------------------------------
# Mapování verze na CUDARC_CUDA_VERSION
# ---------------------------------------------------------------------

map_to_cudarc() {
    local version="$1"
    local major="${version%%.*}"
    local minor="${version#*.}"

    case "$major" in
        13)
            if (( minor >= 1 )); then
                echo "13010"
            else
                echo "13000"
            fi
            ;;
        12)
            if (( minor >= 8 )); then
                echo "12080"
            elif (( minor >= 6 )); then
                echo "12060"
            elif (( minor >= 4 )); then
                echo "12040"
            elif (( minor >= 2 )); then
                echo "12020"
            elif (( minor == 1 )); then
                echo "12010"
            else
                echo "12000"
            fi
            ;;
        *)
            echo ""  # unsupported
            ;;
    esac
}

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

mode="${1:---source}"

# Detekuj
host_version=""
detection_method=""
if version=$(detect_via_nvcc 2>/dev/null) && [[ -n "$version" ]]; then
    host_version="$version"
    detection_method="nvcc"
elif version=$(detect_via_nvidia_smi 2>/dev/null) && [[ -n "$version" ]]; then
    host_version="$version"
    detection_method="nvidia-smi"
else
    echo "ERROR: CUDA nedetekována (nvcc ani nvidia-smi nedostupné)" >&2
    exit 1
fi

cudarc_version=$(map_to_cudarc "$host_version")
if [[ -z "$cudarc_version" ]]; then
    echo "ERROR: CUDA $host_version není podporovaná v cudarc 0.18.x" >&2
    exit 2
fi

case "$mode" in
    --report)
        echo "Host CUDA: $host_version (přes $detection_method)"
        echo "CUDARC_CUDA_VERSION: $cudarc_version"
        echo ""
        echo "Pro export:"
        echo "  export CUDARC_CUDA_VERSION=$cudarc_version"
        ;;
    --export-command)
        echo "export CUDARC_CUDA_VERSION=$cudarc_version"
        ;;
    --source|*)
        # Default: pokud sourced ($0 ≠ shell běžící skript), exportuje;
        # jinak vypíše do stderr a vrátí export command na stdout.
        export CUDARC_CUDA_VERSION="$cudarc_version"
        echo "Host CUDA: $host_version → CUDARC_CUDA_VERSION=$cudarc_version (přes $detection_method)" >&2
        ;;
esac
