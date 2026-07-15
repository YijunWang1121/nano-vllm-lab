#!/usr/bin/env bash
# Pull engine-compare JSON from RunPod into experiments/results/.
#
# Preferred (scp works):
#   1) RunPod UI → Connect → enable / copy "SSH over exposed TCP"
#   2) Edit ~/.ssh/config Host runpod-tcp (HostName + Port)
#   3) bash experiments/pull_results_from_runpod.sh
#
# Fallback uses Host "runpod" proxy (often cannot scp).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/experiments/results"
mkdir -p "$DEST"

HOST="${RUNPOD_SSH_HOST:-runpod-tcp}"
REMOTE_DIR="${RUNPOD_RESULTS_DIR:-/nano-vllm-lab/experiments/results}"

FILES=(
  20260715_223901_sweep_default.json
  20260715_223801_sweep_quick.json
)

if ! ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" 'true' 2>/dev/null; then
  echo "Cannot ssh to '$HOST'."
  echo "If using TCP SSH: set HostName/Port under Host runpod-tcp in ~/.ssh/config"
  echo "  (RunPod UI → Connect → SSH over exposed TCP)."
  echo "Or: RUNPOD_SSH_HOST=runpod bash $0"
  exit 1
fi

for f in "${FILES[@]}"; do
  echo "pull $HOST:$REMOTE_DIR/$f"
  scp -O "$HOST:$REMOTE_DIR/$f" "$DEST/$f"
  wc -c "$DEST/$f"
done

echo "OK → $DEST"
ls -lh "$DEST"
