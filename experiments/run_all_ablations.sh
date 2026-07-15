#!/usr/bin/env bash
# Full ablation suite for nano-vLLM (GPU). Usage:
#   export PYTHONPATH=/nano-vllm-lab
#   bash experiments/run_all_ablations.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-$ROOT}"

MODEL="${NANOVLLM_TEST_MODEL:-$HOME/huggingface/Qwen3-0.6B}"
OUT_DIR="${ABLATION_OUT_DIR:-$HOME/ablation_results}"
mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

run() {
  local name="$1"; shift
  echo
  echo "############################################################"
  echo "# $name"
  echo "############################################################"
  python experiments/run_ablation.py "$@" --json-out "$OUT_DIR/${TS}_${name}.json"
}

echo "model=$MODEL"
echo "out_dir=$OUT_DIR"
python experiments/print_kv_capacity.py --model "$MODEL" | tee "$OUT_DIR/${TS}_kv_capacity.txt"

# 1) CUDA graph
run 01_cudagraph \
  --model "$MODEL" --preset baseline,no_cudagraph \
  --workload shared_prefix --shared-prefix-len 512 --num-seqs 64 --warmup

# 2) Prefix cache
run 02_prefix_shared \
  --model "$MODEL" --preset baseline,no_prefix \
  --workload shared_prefix --shared-prefix-len 512 --num-seqs 64 --warmup

run 03_prefix_multiturn \
  --model "$MODEL" --preset baseline,no_prefix \
  --workload multi_turn --num-sessions 16 --num-turns 4 \
  --shared-prefix-len 512 --warmup

# 3) Continuous batching
run 04_batching \
  --model "$MODEL" --preset baseline,single_seq,small_batch \
  --workload shared_prefix --shared-prefix-len 512 --num-seqs 64 --warmup

# 4) Chunked prefill (long prompt, tight token budget)
run 05_chunked_prefill \
  --model "$MODEL" --preset baseline,no_chunked_prefill \
  --workload shared_prefix --shared-prefix-len 1024 --suffix-len 256 \
  --max-num-batched-tokens 512 --num-seqs 32 --warmup

# 5) Preemption under low KV
run 06_preempt \
  --model "$MODEL" --preset baseline,low_kv,low_kv_no_preempt \
  --workload shared_prefix --shared-prefix-len 512 --suffix-len 256 \
  --num-seqs 128 --max-output-len 256 --warmup

echo
echo "All done. JSON logs in: $OUT_DIR"
ls -1 "$OUT_DIR/${TS}"* 2>/dev/null || true
