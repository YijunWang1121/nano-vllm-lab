#!/usr/bin/env bash
# Isolate CUDA-graph effect: same shapes / same flash / prefix OFF,
# only toggle enforce_eager (eager vs graphs) for lab + vLLM.
#
#   bash experiments/run_graph_ablation.sh
#   SUITE=quick REPEATS=1 bash experiments/run_graph_ablation.sh
#   CASES=bs32_in256_out128,bs8_in128_out256 REPEATS=3 bash experiments/run_graph_ablation.sh
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

SUITE="${SUITE:-graph_ablation}"
REPEATS="${REPEATS:-3}"
CASES="${CASES:-}"

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

EXTRA_CASES=()
if [[ -n "$CASES" ]]; then
  EXTRA_CASES=(--cases "$CASES")
fi

echo "model:  $MODEL"
echo "attn:   $NANOVLLM_ATTN_BACKEND"
echo "suite:  $SUITE  repeats=$REPEATS  cases=${CASES:-all}"
echo "modes:  eager (enforce_eager=True) then graphs (enforce_eager=False)"
echo "prefix: OFF (isolate graphs only)"

EAGER_JSON="$OUT_DIR/${TS}_graph_ablation_eager.json"
GRAPH_JSON="$OUT_DIR/${TS}_graph_ablation_graphs.json"
MERGE_JSON="$OUT_DIR/${TS}_graph_ablation.json"

python experiments/sweep_engine_compare.py \
  --model "$MODEL" \
  --suite "$SUITE" \
  --repeats "$REPEATS" \
  --engines lab,vllm \
  --enforce-eager \
  --warmup \
  --quiet \
  "${EXTRA_CASES[@]}" \
  --json-out "$EAGER_JSON"

python experiments/sweep_engine_compare.py \
  --model "$MODEL" \
  --suite "$SUITE" \
  --repeats "$REPEATS" \
  --engines lab,vllm \
  --no-enforce-eager \
  --warmup \
  --quiet \
  "${EXTRA_CASES[@]}" \
  --json-out "$GRAPH_JSON"

python experiments/merge_graph_ablation.py \
  --eager "$EAGER_JSON" \
  --graphs "$GRAPH_JSON" \
  --json-out "$MERGE_JSON"

echo
echo "Done:"
echo "  $EAGER_JSON"
echo "  $GRAPH_JSON"
echo "  $MERGE_JSON"
