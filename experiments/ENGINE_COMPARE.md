# Engine comparison: lab vs main vs vLLM

Branch: `experiments/engine-compare`

## What is being compared

| Engine | Source | Notes |
|--------|--------|--------|
| **lab** | this branch / `course/reference-solutions` | Course engine + ablation switches + GPU test hardening |
| **main** | git `main` via worktree | Upstream nano-vLLM (no course TODOs / ablation flags) |
| **vllm** | `pip install vllm` | Production engine; optional |

## Lab vs main (code)

Relative to `main`, the lab tree mainly adds:

- Ablation Config flags (`enable_prefix_caching`, `enable_preemption`, `enable_chunked_prefill`)
- Course scaffolding (TODOs on student branch, tests, docs)
- Robustness fixes used by the course GPU suite (engine exit / memory, CUDA-graph debug safety, Sampler without `@torch.compile`)
- Extracted `input_metadata` helpers

Default-on behavior should match upstream for a fair bench (`enforce_eager=False`, prefix/preempt/chunked on).

## How to run (GPU)

```bash
cd /nano-vllm-lab
git fetch origin
git switch experiments/engine-compare   # or pull this branch
source ~/venv-nanovllm/bin/activate
export PYTHONPATH=/nano-vllm-lab
export NANOVLLM_TEST_MODEL=/root/huggingface/Qwen3-0.6B

# optional vLLM
# pip install vllm

bash experiments/run_engine_compare.sh
```

Or manually:

```bash
git worktree add /tmp/nano-vllm-main main
python experiments/compare_engines.py --engines lab,main,vllm \
  --main-path /tmp/nano-vllm-main --warmup --num-seqs 64
```

## Metrics

Same as `bench.py`: **sum(max_tokens) / wall_clock** after optional warmup.  
Do not compare these numbers to `example.py` progress-bar Prefill/Decode rates.
