# Experiments

## Quick compare (nano / PyTorch-serial / vLLM)

Same seed and length ranges for all engines:

```bash
source /workspace/venv-nanovllm/bin/activate
cd /path/to/nano-vllm
export NANOVLLM_TEST_MODEL=/workspace/huggingface/Qwen3-0.6B

# random token-id workload (default)
bash experiments/run_compare.sh

# multi-turn chat prompts (tokenizer chat template, like example.py)
WORKLOAD=chat CHAT_TURNS=3 bash experiments/run_compare.sh --engines nanovllm,pytorch

# also vLLM (install in another env if needed)
ENGINES=nanovllm,pytorch,vllm \
  VLLM_PYTHON=/workspace/venv-vllm/bin/python \
  bash experiments/run_compare.sh

# chat + all three engines
WORKLOAD=chat CHAT_TURNS=3 \
  ENGINES=nanovllm,pytorch,vllm \
  VLLM_PYTHON=/workspace/venv-vllm/bin/python \
  bash experiments/run_compare.sh

# CUDA graphs for nano
ENFORCE_EAGER=0 bash experiments/run_compare.sh --engines nanovllm
```

Or call engines one by one:

```bash
python experiments/bench_nanovllm.py --model "$NANOVLLM_TEST_MODEL"
python experiments/bench_nanovllm.py --workload chat --chat-turns 3
python experiments/bench_pytorch_baseline.py --model "$NANOVLLM_TEST_MODEL" --mode sequential
python experiments/bench_pytorch_baseline.py --workload chat
python experiments/bench_vllm.py --model "$NANOVLLM_TEST_MODEL"
python experiments/bench_vllm.py --workload chat
```

Smoke generation (chat demo):

```bash
export NANOVLLM_TEST_MODEL=/workspace/huggingface/Qwen3-0.6B
python example.py
```

| Script | Engine |
|--------|--------|
| `bench_nanovllm.py` | this repo |
| `bench_pytorch_baseline.py` | HF `generate`, sequential by default |
| `bench_vllm.py` | production vLLM |
| `run_compare.sh` | runs a subset with matched knobs |

**Workloads**

| `--workload` | Description |
|--------------|-------------|
| `random` (default) | Random token IDs; good for offline throughput |
| `chat` | **Static** multi-turn history: assistant replies are pre-written text, then one generation per request. Good for benchmarking long chat-formatted prompts, not true dialogue. |
| *(interactive)* | See `chat_interactive_demo.py`: model output is appended each round before the next user message. |

True multi-turn loop (model output → next prompt):

```bash
python experiments/chat_interactive_demo.py
python experiments/chat_interactive_demo.py --max-tokens 128
python experiments/chat_interactive_demo.py \
  --user-turns "What is CUDA graph?" "When should I disable it?" "Show a tiny example."
```

**vLLM setup (RunPod example)**

```bash
python3 -m venv /workspace/venv-vllm
source /workspace/venv-vllm/bin/activate
pip install vllm
```

Then point `run_compare.sh` at that interpreter:

```bash
VLLM_PYTHON=/workspace/venv-vllm/bin/python \
  ENGINES=nanovllm,pytorch,vllm \
  bash experiments/run_compare.sh
```

**Note:** If `transformers` 5.x breaks on Torch 2.4 (`DTensor` import error):

```bash
pip install "transformers>=4.51.0,<5" "huggingface_hub<1"
```

For full lab-vs-vLLM sweeps / graph ablation, see branch `experiments/engine-compare`.
