#!/usr/bin/env bash
# Full-config lab vs vLLM SLO compare (graphs + prefix cache + matched flash).
#
#   bash experiments/run_slo_compare.sh
#   WORKLOAD=independent bash experiments/run_slo_compare.sh
#   WORKLOAD=single_stream NUM_SEQS=16 bash experiments/run_slo_compare.sh
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

WORKLOAD="${WORKLOAD:-shared_prefix}"
NUM_SEQS="${NUM_SEQS:-32}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"

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

echo "model:    $MODEL"
echo "attn:     $NANOVLLM_ATTN_BACKEND"
echo "workload: $WORKLOAD  num_seqs=$NUM_SEQS output_len=$OUTPUT_LEN"
echo "full-config: CUDA graphs ON, prefix caching ON"

python experiments/bench_slo_compare.py \
  --model "$MODEL" \
  --engines lab,vllm \
  --workload "$WORKLOAD" \
  --num-seqs "$NUM_SEQS" \
  --prefix-len 512 \
  --suffix-len 128 \
  --output-len "$OUTPUT_LEN" \
  --gpu-memory-utilization 0.9 \
  --no-enforce-eager \
  --prefix-caching \
  --warmup \
  --slo-ttft-ms 500 \
  --slo-tpot-ms 50 \
  --skip-missing \
  --quiet \
  --json-out "$OUT_DIR/${TS}_slo_${WORKLOAD}.json"

echo
echo "Done: $OUT_DIR/${TS}_slo_${WORKLOAD}.json"
