#!/usr/bin/env bash
# Reliable multi-config lab vs vLLM sweep (fixed lengths × repeats).
#
#   bash experiments/run_engine_sweep.sh              # suite=default, repeats=3
#   SUITE=quick REPEATS=1 bash experiments/run_engine_sweep.sh
#   SUITE=stress REPEATS=2 bash experiments/run_engine_sweep.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-$ROOT}"

_nv_root="${VIRTUAL_ENV:-$HOME/venv-nanovllm}/lib"
_nv_libs="$(find "${_nv_root}" -type d -path '*/site-packages/nvidia/*/lib' 2>/dev/null | paste -sd: - || true)"
if [[ -n "${_nv_libs}" ]]; then
  export LD_LIBRARY_PATH="${_nv_libs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset _nv_root _nv_libs

MODEL="${NANOVLLM_TEST_MODEL:-$HOME/huggingface/Qwen3-0.6B}"
OUT_DIR="${COMPARE_OUT_DIR:-$HOME/engine_compare_results}"
mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

SUITE="${SUITE:-default}"
REPEATS="${REPEATS:-3}"

# Same-kernel default when available.
if [[ -z "${NANOVLLM_ATTN_BACKEND:-}" ]]; then
  if python -c "import vllm.vllm_flash_attn" 2>/dev/null \
     || python -c "import vllm_flash_attn" 2>/dev/null; then
    export NANOVLLM_ATTN_BACKEND=vllm_flash
  elif python -c "from flash_attn import flash_attn_varlen_func" 2>/dev/null; then
    export NANOVLLM_ATTN_BACKEND=flash
  else
    export NANOVLLM_ATTN_BACKEND=torch
  fi
fi

echo "model:  $MODEL"
echo "attn:   $NANOVLLM_ATTN_BACKEND"
echo "suite:  $SUITE  repeats=$REPEATS"

python experiments/sweep_engine_compare.py \
  --model "$MODEL" \
  --suite "$SUITE" \
  --repeats "$REPEATS" \
  --engines lab,vllm \
  --enforce-eager \
  --warmup \
  --quiet \
  --json-out "$OUT_DIR/${TS}_sweep_${SUITE}.json"

echo
echo "Done: $OUT_DIR/${TS}_sweep_${SUITE}.json"
