#!/usr/bin/env bash
#
# sync_back.sh — rsync výstupů z Vast AI instance zpět na local.
#
# Pouští se NA LOKÁLNÍ MAŠINĚ (ne na Vast instanci).
# Předpokládá, že Tailscale je aktivní a Vast instance je dostupná.
#
# Použití:
#   bash scripts/cloud/sync_back.sh <vast-tailscale-hostname-or-ip>
#
# Příklad:
#   bash scripts/cloud/sync_back.sh vast-eleutheria-abc12345
#   bash scripts/cloud/sync_back.sh 100.x.y.z

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <vast-tailscale-hostname-or-ip>" >&2
    echo "" >&2
    echo "Vast instance hostname najdeš na Tailscale admin:" >&2
    echo "  https://login.tailscale.com/admin/machines" >&2
    exit 1
fi

VAST_HOST="$1"
VAST_USER="${VAST_USER:-root}"
LOCAL_DIR="${LOCAL_DIR:-$HOME/.eleutheria/cloud_runs}"

mkdir -p "$LOCAL_DIR"

echo "Sync z $VAST_USER@$VAST_HOST → $LOCAL_DIR"
echo ""

# Co stahujeme:
# - Core Memory artefakty (*.safetensors)
# - Tracing logy (pokud uživatel zapnul logging)
# - Případně dataset (jen pokud byl modifikován na cloudu)

rsync -avhP --stats \
    -e "ssh -o StrictHostKeyChecking=accept-new" \
    "$VAST_USER@$VAST_HOST:~/sofie_identity*.safetensors" \
    "$VAST_USER@$VAST_HOST:~/*.optim.safetensors" \
    "$VAST_USER@$VAST_HOST:~/*.log" \
    "$LOCAL_DIR/" 2>&1 || {
    echo ""
    echo "Některé soubory nemusely existovat (rsync ignoruje, pokračuj)."
}

echo ""
echo "Hotovo. Soubory v $LOCAL_DIR:"
ls -lah "$LOCAL_DIR/" | tail -20
