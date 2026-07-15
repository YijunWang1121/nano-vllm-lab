#!/usr/bin/env bash
# Fair compare: lab vs vLLM with the same attention class + enforce_eager.
#
# Default: Flash Attention on both (vLLM already uses it; lab uses flash-attn when importable).
# Fallback: NANOVLLM_ATTN_BACKEND=torch → lab SDPA + vLLM TORCH_SDPA (V0).
#
#   cd /nano-vllm-lab
#   bash experiments/run_engine_compare.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-$ROOT}"

# Kill leftover GPU holders from a previous failed run (best effort).
python - <<'PY' || true
import gc
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"cuda free before run: {free/1024**3:.2f}/{total/1024**3:.2f} GiB")
except Exception as e:
    print("cuda pre-clean:", e)
gc.collect()
PY

# vLLM may need nvidia/*/lib on LD_LIBRARY_PATH (libcudart.so.13).
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

# Pick a fair attention mode: flash if importable, else torch/SDPA on both engines.
if [[ -z "${NANOVLLM_ATTN_BACKEND:-}" ]]; then
  if python -c "from flash_attn import flash_attn_varlen_func" 2>/dev/null; then
    export NANOVLLM_ATTN_BACKEND=flash
  else
    export NANOVLLM_ATTN_BACKEND=torch
  fi
fi

echo "lab root: $ROOT"
echo "model:    $MODEL"
echo "attn:     NANOVLLM_ATTN_BACKEND=$NANOVLLM_ATTN_BACKEND (matched on lab + vLLM)"

ENGINES="lab"
if python -c "import vllm" 2>/dev/null; then
  ENGINES="lab,vllm"
  echo "vLLM: available"
else
  echo "vLLM: not installed"
fi
echo "engines: $ENGINES"

python experiments/compare_engines.py \
  --model "$MODEL" \
  --engines "$ENGINES" \
  --num-seqs 64 \
  --max-input-len 512 \
  --max-output-len 128 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --warmup \
  --skip-missing \
  --json-out "$OUT_DIR/${TS}_compare.json"

echo
echo "Done: $OUT_DIR/${TS}_compare.json"
