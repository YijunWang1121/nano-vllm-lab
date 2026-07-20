# Experiments

## Quick compare (nano / PyTorch-serial / vLLM)

Same seed and length ranges for all engines:

```bash
source /workspace/venv-nanovllm/bin/activate   # or your venv
cd /path/to/nano-vllm
export NANOVLLM_TEST_MODEL=/workspace/huggingface/Qwen3-0.6B

# nano + pytorch serial (default)
bash experiments/run_compare.sh

# also vLLM (install in another env if needed)
ENGINES=nanovllm,pytorch,vllm \
  VLLM_PYTHON=/workspace/venv-vllm/bin/python \
  bash experiments/run_compare.sh

# larger batch
NUM_SEQS=64 MIN_IN=100 MAX_IN=512 MIN_OUT=64 MAX_OUT=128 \
  bash experiments/run_compare.sh --engines nanovllm,pytorch
```

Or call engines one by one:

```bash
python experiments/bench_nanovllm.py --model "$NANOVLLM_TEST_MODEL"
python experiments/bench_pytorch_baseline.py --model "$NANOVLLM_TEST_MODEL" --mode sequential
python experiments/bench_vllm.py --model "$NANOVLLM_TEST_MODEL"
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

**Note:** If `transformers` 5.x breaks on Torch 2.4 (`DTensor` import error):

```bash
pip install "transformers>=4.51.0,<5" "huggingface_hub<1"
```

For full lab-vs-vLLM sweeps / graph ablation, see branch `experiments/engine-compare`.
