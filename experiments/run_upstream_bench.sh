#!/usr/bin/env bash
# Match upstream nano-vLLM bench.py workload (lab vs vLLM, same tokens + same flash).
# Ref: https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py
#
# Upstream knobs:
#   seed=0, num_seqs=256
#   prompt len ~ U(100, 1024), max_tokens ~ U(100, 1024)
#   enforce_eager=False (CUDA graphs ON), max_model_len=4096
#   temperature=0.6, ignore_eos=True, one warmup generate
#
#   bash experiments/run_upstream_bench.sh
#   REPEATS=3 bash experiments/run_upstream_bench.sh
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

export NANOVLLM_TEST_MODEL="${NANOVLLM_TEST_MODEL:-$HOME/huggingface/Qwen3-0.6B}"
OUT_DIR="${COMPARE_OUT_DIR:-$HOME/engine_compare_results}"
mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

export UPSTREAM_BENCH_REPEATS="${REPEATS:-1}"
export UPSTREAM_BENCH_GPU_UTIL="${GPU_UTIL:-0.9}"
export UPSTREAM_BENCH_BASE_SEED="${BASE_SEED:-0}"
export UPSTREAM_BENCH_JSON_OUT="$OUT_DIR/${TS}_upstream_bench.json"

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

echo "=== upstream-style bench (GeeeekExplorer/nano-vllm bench.py) ==="
echo "model:     $NANOVLLM_TEST_MODEL"
echo "attn:      $NANOVLLM_ATTN_BACKEND  (matched FLASH_ATTN on vLLM)"
echo "workload:  num_seqs=256  in~U(100,1024)  out~U(100,1024)  seed=${UPSTREAM_BENCH_BASE_SEED}+"
echo "engine:    enforce_eager=False (graphs ON)  max_model_len=4096  util=$UPSTREAM_BENCH_GPU_UTIL"
echo "prefix:    OFF (fair; random prompts get no hits anyway)"
echo "repeats:   $UPSTREAM_BENCH_REPEATS"
echo

python experiments/run_upstream_bench.py

echo
echo "Done: $UPSTREAM_BENCH_JSON_OUT"
