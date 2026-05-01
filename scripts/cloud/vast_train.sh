#!/usr/bin/env bash
#
# vast_train.sh — wrapper pro production training na Vast AI instanci.
#
# Default parametry pro alpha.20 production training Sofie identity packu
# s identifikovaným HP setupem (RN-012):
#   LR=1e-3, β1=0.0 (RMSProp), --save-best, --checkpoint
#
# Pro 16+ GB VRAM (RTX A4000, RTX 4000, A6000, atd.) využíváme větší
# batch / seq_len než lokální RTX 4050 6 GB (KI-005).
#
# Override přes env vars:
#   DATASET=...           cesta k pack souboru
#   OUTPUT=...            cesta k output safetensors
#   BATCH=...             batch_size (default 4)
#   SEQ_LEN=...           seq_len (default 8)
#   GRAD_ACCUM=...        grad_accum_steps (default 2)
#   EPOCHS=...            počet epoch (default 1)
#   EXTRA_ARGS=...        další CLI argumenty (např. --warmup-steps 50)

set -euo pipefail

ELEUTHERIA_DIR="${ELEUTHERIA_DIR:-$HOME/eleutheria}"
cd "$ELEUTHERIA_DIR"

DATASET="${DATASET:-$ELEUTHERIA_DIR/dataset/training/sofie_identity_pack.txt}"
OUTPUT="${OUTPUT:-$HOME/sofie_identity_v1.safetensors}"
NOTES="${NOTES:-alpha.20 production cloud GPU run}"

BATCH="${BATCH:-4}"
SEQ_LEN="${SEQ_LEN:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EPOCHS="${EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
ADAM_BETA1="${ADAM_BETA1:-0.0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "=========================================="
echo "Eleutheria production training"
echo "=========================================="
echo "Dataset:       $DATASET"
echo "Output:        $OUTPUT"
echo "Epochs:        $EPOCHS"
echo "Batch:         $BATCH"
echo "Seq len:       $SEQ_LEN"
echo "Grad accum:    $GRAD_ACCUM"
echo "Learning rate: $LEARNING_RATE"
echo "AdamW β1:      $ADAM_BETA1"
echo "Notes:         $NOTES"
echo "Extra:         $EXTRA_ARGS"
echo ""
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Sanity: dataset existuje
if [[ ! -f "$DATASET" ]]; then
    echo "ERROR: dataset $DATASET neexistuje" >&2
    exit 1
fi

# Sanity: model directory existuje
MODEL_DIR="${MODEL_DIR:-/home/lvx/Models/falcon-h1-1.5b-instruct}"
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: model dir $MODEL_DIR neexistuje" >&2
    echo "Stáhni Falcon-H1 model nejdřív (viz vast_setup.sh)" >&2
    exit 1
fi

# Sanity: build hotový
BINARY="$ELEUTHERIA_DIR/target/release/eleutheria"
if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: binary $BINARY neexistuje, nejdřív cargo build --release --features cuda" >&2
    exit 1
fi

# Output dir
mkdir -p "$(dirname "$OUTPUT")"

START_TS=$(date +%s)
echo "Start: $(date -d @$START_TS)"
echo ""

"$BINARY" --cuda --no-core-memory \
    --model "$MODEL_DIR" \
    train-core-memory \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --seq-len "$SEQ_LEN" \
    --grad-accum "$GRAD_ACCUM" \
    --grad-clip 1.0 \
    --learning-rate "$LEARNING_RATE" \
    --adam-beta1 "$ADAM_BETA1" \
    --save-best \
    --checkpoint \
    --output "$OUTPUT" \
    --notes "$NOTES" \
    $EXTRA_ARGS

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo ""
echo "End:     $(date -d @$END_TS)"
echo "Elapsed: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo ""
echo "Output:"
ls -la "$OUTPUT" "${OUTPUT%.safetensors}.optim.safetensors" 2>/dev/null
echo ""
echo "Inspect:"
"$BINARY" --inspect-core-memory "$OUTPUT" || true
