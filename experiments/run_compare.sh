#!/usr/bin/env bash
# Compare nano-vLLM / PyTorch-serial / vLLM with the same workload knobs.
#
# Usage (on RunPod):
#   source /workspace/venv-nanovllm/bin/activate
#   export NANOVLLM_TEST_MODEL=/workspace/huggingface/Qwen3-0.6B
#   bash experiments/run_compare.sh
#
#   # include vLLM (separate venv recommended)
#   ENGINES=nanovllm,pytorch,vllm VLLM_PYTHON=/workspace/venv-vllm/bin/python \
#     bash experiments/run_compare.sh
#
#   # multi-turn chat prompts (tokenizer chat template)
#   WORKLOAD=chat CHAT_TURNS=3 bash experiments/run_compare.sh --engines nanovllm,pytorch,vllm
#
#   ENFORCE_EAGER=0 bash experiments/run_compare.sh --engines nanovllm
#
#   # give the HF baseline FlashAttention too (needs flash-attn installed)
#   HF_ATTN=flash_attention_2 bash experiments/run_compare.sh --engines nanovllm,pytorch
#   # batched HF baseline instead of one-request-at-a-time
#   HF_MODE=batched HF_ATTN=flash_attention_2 bash experiments/run_compare.sh --engines pytorch

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${NANOVLLM_TEST_MODEL:-${MODEL:-$HOME/huggingface/Qwen3-0.6B}}"
MODEL="$(python -c "import os; print(os.path.expanduser('$MODEL'))")"
ENGINES="${ENGINES:-nanovllm,pytorch}"
WORKLOAD="${WORKLOAD:-random}"
NUM_SEQS="${NUM_SEQS:-8}"
MIN_IN="${MIN_IN:-64}"
MAX_IN="${MAX_IN:-128}"
MIN_OUT="${MIN_OUT:-32}"
MAX_OUT="${MAX_OUT:-64}"
CHAT_TURNS="${CHAT_TURNS:-3}"
SEED="${SEED:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
# HF baseline attention backend: sdpa | eager | flash_attention_2
HF_ATTN="${HF_ATTN:-sdpa}"
HF_MODE="${HF_MODE:-sequential}"
VLLM_PYTHON="${VLLM_PYTHON:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engines) ENGINES="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --num-seqs) NUM_SEQS="$2"; shift 2 ;;
    --workload) WORKLOAD="$2"; shift 2 ;;
    --chat-turns) CHAT_TURNS="$2"; shift 2 ;;
    --help|-h)
      sed -n '2,18p' "$0"
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
  --workload "$WORKLOAD"
  --num-seqs "$NUM_SEQS"
  --min-input-len "$MIN_IN"
  --max-input-len "$MAX_IN"
  --min-output-len "$MIN_OUT"
  --max-output-len "$MAX_OUT"
  --chat-turns "$CHAT_TURNS"
  --seed "$SEED"
)

echo "============================================================"
echo "compare model=$MODEL engines=$ENGINES workload=$WORKLOAD num_seqs=$NUM_SEQS"
if [[ "$WORKLOAD" == "chat" ]]; then
  echo "  chat_turns=$CHAT_TURNS output=[$MIN_OUT,$MAX_OUT] seed=$SEED"
else
  echo "  input=[$MIN_IN,$MAX_IN] output=[$MIN_OUT,$MAX_OUT] seed=$SEED"
fi
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
        --mode "$HF_MODE" \
        --attn-implementation "$HF_ATTN"
      ;;
    vllm)
      if [[ ! -x "$VLLM_PYTHON" ]]; then
        echo "ERROR: vLLM python not found: $VLLM_PYTHON" >&2
        echo "Create a separate venv (recommended; vLLM pins its own torch stack):" >&2
        echo "  python3 -m venv /workspace/venv-vllm" >&2
        echo "  source /workspace/venv-vllm/bin/activate && pip install vllm" >&2
        echo "Then rerun with:" >&2
        echo "  VLLM_PYTHON=/workspace/venv-vllm/bin/python bash experiments/run_compare.sh ..." >&2
        exit 1
      fi
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
