#!/usr/bin/env bash
#
# vast_train_7b.sh — wrapper pro 7B production training na Vast AI.
#
# Volání alpha.20-style HP setupu (RN-012 production HP) na Falcon-H1-7B
# místo 1.5B. Cíl: alpha.24 capacity ablace — testuje, zda větší
# model amplifikuje slabý positive signál CS reasoning self_app
# z alpha.23 (1.5B `(core − cold) = 0/+15%` s N=3, statistická slabost).
#
# Parametry per RN-012:
#   LR=1e-3, β1=0.0 (RMSProp), batch=16, seq_len=16, grad_accum=2,
#   epochs=5, --save-best, --save-best-every 5, --checkpoint
#
# Doporučená Vast GPU: A100-80GB nebo H100-80GB (15.5 GB 7B BF16 + Adam
# state + activations comfortably fit). RTX A6000 48 GB by mělo též.

set -euo pipefail

ELEUTHERIA_DIR="${ELEUTHERIA_DIR:-$HOME/eleutheria}"

# 7B-specific defaults
export MODEL_DIR="${MODEL_DIR:-$HOME/Models/falcon-h1-7b-instruct}"
export OUTPUT="${OUTPUT:-$HOME/sofie_identity_v2_7b.safetensors}"
export DATASET="${DATASET:-$ELEUTHERIA_DIR/dataset/training/sofie_identity_pack.txt}"
export NOTES="${NOTES:-alpha.24 7B capacity ablace, RN-012 production HP}"

# RN-012 production HP (alpha.20 batch=16 seq=16 setup)
export EPOCHS="${EPOCHS:-5}"
export BATCH="${BATCH:-16}"
export SEQ_LEN="${SEQ_LEN:-16}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export LEARNING_RATE="${LEARNING_RATE:-1e-3}"
export ADAM_BETA1="${ADAM_BETA1:-0.0}"
export SAVE_BEST_EVERY="${SAVE_BEST_EVERY:-5}"

echo "=========================================="
echo "Eleutheria 7B production training (alpha.24)"
echo "=========================================="
echo "Capacity ablace 1.5B → 7B post alpha.23 finding (b)"
echo ""

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: model dir $MODEL_DIR neexistuje" >&2
    echo "" >&2
    echo "Stáhni Falcon-H1-7B-Instruct z HuggingFace:" >&2
    echo "  export HF_TOKEN=hf_..." >&2
    echo "  pip install --user huggingface-hub" >&2
    echo "  ~/.local/bin/huggingface-cli login --token \$HF_TOKEN" >&2
    echo "  ~/.local/bin/huggingface-cli download tiiuae/Falcon-H1-7B-Instruct \\" >&2
    echo "    --local-dir $MODEL_DIR" >&2
    exit 1
fi

echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

exec bash "$ELEUTHERIA_DIR/scripts/cloud/vast_train.sh"
