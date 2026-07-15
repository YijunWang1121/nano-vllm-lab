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

## Fair matching (attention + eager)

`run_engine_compare.sh` keeps **lab and vLLM on the same attention class**:

| Mode | Lab | vLLM |
|------|-----|------|
| **flash** (default if `flash_attn` imports) | `NANOVLLM_ATTN_BACKEND=flash` | Flash Attention (auto) |
| **torch** (fallback) | SDPA | `VLLM_ATTENTION_BACKEND=XFORMERS` + `VLLM_USE_V1=0` |

Install lab `flash-attn` (torch 2.6 / cu124 / py311) from a prebuilt wheel — do **not** compile from source unless you must:

```bash
pip install --no-cache-dir \
  https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu124torch2.6-cp311-cp311-linux_x86_64.whl
```

Both sides use `--enforce-eager` and `enable_prefix_caching=False` for this synthetic workload.

If a previous run forced `NANOVLLM_ATTN_BACKEND=torch`, **unset it** so flash can be used:

```bash
unset NANOVLLM_ATTN_BACKEND
python -c "from flash_attn import flash_attn_varlen_func; print('flash-attn ok')"
```

## How to run (GPU)

```bash
cd /nano-vllm-lab
git fetch origin
git switch experiments/engine-compare   # or pull this branch
source ~/venv-nanovllm/bin/activate
export PYTHONPATH=/nano-vllm-lab
export NANOVLLM_TEST_MODEL=/root/huggingface/Qwen3-0.6B
export LD_LIBRARY_PATH="$(find ~/venv-nanovllm/lib/python3.11/site-packages/nvidia -type d -name lib | paste -sd: -):${LD_LIBRARY_PATH:-}"
unset NANOVLLM_ATTN_BACKEND   # let script pick flash if available

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
