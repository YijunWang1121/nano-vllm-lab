# Engine comparison: lab vs main vs vLLM

Branch: `experiments/engine-compare`

**Report (results + interpretation):** [`ENGINE_COMPARE_REPORT.md`](./ENGINE_COMPARE_REPORT.md)  
**Numeric artifacts:** [`results/`](./results/)

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
| **vllm_flash** (default if available) | imports `vllm_flash_attn` (same .so) | `VLLM_ATTENTION_BACKEND=FLASH_ATTN` |
| **flash** | pip `flash_attn` (different binary) | `FLASH_ATTN` (still vLLM's bundle) |
| **torch** (fallback) | SDPA | `VLLM_ATTENTION_BACKEND=XFORMERS` |

vLLM ships its own FlashAttention copy (`vllm_flash_attn`). Pip `flash-attn` is a **different** shared library even when both say "Flash Attention". For same-kernel compare, use `NANOVLLM_ATTN_BACKEND=vllm_flash`.

Both sides use `--enforce-eager` and `enable_prefix_caching=False` for this synthetic workload.

```bash
unset NANOVLLM_ATTN_BACKEND
python -c "import vllm.vllm_flash_attn as m; print(m.__file__)"
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

## Reliable sweep (recommended)

Single-shot random lengths are noisy. Use the sweep for **fixed shapes × repeats**:

```bash
# default suite: 8 shapes × 3 seeds  (tens of minutes on 4090)
bash experiments/run_engine_sweep.sh

# smoke
SUITE=quick REPEATS=1 bash experiments/run_engine_sweep.sh

# heavier shapes
SUITE=stress REPEATS=2 bash experiments/run_engine_sweep.sh
```

Suites (`experiments/sweep_engine_compare.py`):

| suite | cases | intent |
|-------|------:|--------|
| `quick` | 2 | smoke |
| `default` | 8 | decode-heavy / prefill-heavy / short / long-ctx / batch sizes |
| `stress` | 7 | longer decode / wider batch (may OOM) |

Each case uses **fixed** `input_len` / `output_len` (same token ids across engines; different seed per repeat). Summary prints `mean±std` tok/s and `vs_lab`.
