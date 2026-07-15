#!/usr/bin/env bash
# Compare course lab engine vs git main vs vLLM.
#
#   cd /nano-vllm-lab
#   git fetch origin main
#   bash experiments/run_engine_compare.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-$ROOT}"

MODEL="${NANOVLLM_TEST_MODEL:-$HOME/huggingface/Qwen3-0.6B}"
MAIN_PATH="${NANOVLLM_MAIN_PATH:-/tmp/nano-vllm-main}"
OUT_DIR="${COMPARE_OUT_DIR:-$HOME/engine_compare_results}"
mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

echo "lab root:  $ROOT"
echo "main path: $MAIN_PATH"
echo "model:     $MODEL"

if [[ ! -d "$MAIN_PATH/nanovllm" ]]; then
  echo "Creating worktree for main at $MAIN_PATH ..."
  git fetch origin main 2>/dev/null || true
  # Prefer local main; fall back to origin/main.
  if git show-ref --verify --quiet refs/heads/main; then
    git worktree add "$MAIN_PATH" main
  else
    git worktree add "$MAIN_PATH" origin/main
  fi
fi

echo
echo "=== package paths (sanity) ==="
PYTHONPATH="$ROOT" python -c "import nanovllm; print('lab ', nanovllm.__file__)"
PYTHONPATH="$MAIN_PATH" python -c "import nanovllm; print('main', nanovllm.__file__)"

ENGINES="lab,main"
if python -c "import vllm" 2>/dev/null; then
  ENGINES="lab,main,vllm"
  echo "vLLM: available"
else
  echo "vLLM: not installed (skip). Install with: pip install vllm"
fi

python experiments/compare_engines.py \
  --model "$MODEL" \
  --engines "$ENGINES" \
  --main-path "$MAIN_PATH" \
  --num-seqs 64 \
  --max-input-len 512 \
  --max-output-len 128 \
  --warmup \
  --skip-missing \
  --json-out "$OUT_DIR/${TS}_compare.json"

echo
echo "Done: $OUT_DIR/${TS}_compare.json"
