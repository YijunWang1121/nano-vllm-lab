#!/usr/bin/env bash
# Compare nano-vLLM / PyTorch-serial / vLLM with the same workload knobs.
#
# Usage (on RunPod):
#   source /workspace/venv-nanovllm/bin/activate
#   export NANOVLLM_TEST_MODEL=/workspace/huggingface/Qwen3-0.6B
#   bash experiments/run_compare.sh
#   bash experiments/run_compare.sh --engines nanovllm,pytorch
#   ENGINES=nanovllm,pytorch,vllm NUM_SEQS=8 bash experiments/run_compare.sh
#
# Optional: VLLM_PYTHON=/workspace/venv-vllm/bin/python

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${NANOVLLM_TEST_MODEL:-${MODEL:-$HOME/huggingface/Qwen3-0.6B}}"
MODEL="$(python -c "import os; print(os.path.expanduser('$MODEL'))")"
ENGINES="${ENGINES:-nanovllm,pytorch}"
NUM_SEQS="${NUM_SEQS:-8}"
MIN_IN="${MIN_IN:-64}"
MAX_IN="${MAX_IN:-128}"
MIN_OUT="${MIN_OUT:-32}"
MAX_OUT="${MAX_OUT:-64}"
SEED="${SEED:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
VLLM_PYTHON="${VLLM_PYTHON:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engines) ENGINES="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --num-seqs) NUM_SEQS="$2"; shift 2 ;;
    --help|-h)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -d "$MODEL" ]]; then
  echo "ERROR: model dir not found: $MODEL" >&2
  echo "  export NANOVLLM_TEST_MODEL=/workspace/huggingface/Qwen3-0.6B" >&2
  exit 1
fi

EAGER_FLAG=(--enforce-eager)
if [[ "$ENFORCE_EAGER" == "0" ]]; then
  EAGER_FLAG=(--no-enforce-eager)
fi

COMMON=(
  --model "$MODEL"
  --num-seqs "$NUM_SEQS"
  --min-input-len "$MIN_IN"
  --max-input-len "$MAX_IN"
  --min-output-len "$MIN_OUT"
  --max-output-len "$MAX_OUT"
  --seed "$SEED"
)

echo "============================================================"
echo "compare model=$MODEL engines=$ENGINES num_seqs=$NUM_SEQS"
echo "  input=[$MIN_IN,$MAX_IN] output=[$MIN_OUT,$MAX_OUT] seed=$SEED"
echo "============================================================"

IFS=',' read -r -a engine_list <<< "$ENGINES"
for eng in "${engine_list[@]}"; do
  eng="$(echo "$eng" | tr -d '[:space:]')"
  echo
  echo "---------- $eng ----------"
  case "$eng" in
    nanovllm|lab|nano)
      python experiments/bench_nanovllm.py "${COMMON[@]}" "${EAGER_FLAG[@]}"
      ;;
    pytorch|torch|hf|serial)
      python experiments/bench_pytorch_baseline.py \
        "${COMMON[@]}" \
        --mode sequential \
        --attn-implementation sdpa
      ;;
    vllm)
      "$VLLM_PYTHON" experiments/bench_vllm.py "${COMMON[@]}" "${EAGER_FLAG[@]}"
      ;;
    *)
      echo "Unknown engine: $eng (use nanovllm,pytorch,vllm)" >&2
      exit 1
      ;;
  esac
done

echo
echo "Done."
