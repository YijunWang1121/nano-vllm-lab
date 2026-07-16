# Engine comparison: lab vs main vs vLLM

Branch: `experiments/engine-compare`

**Report (results + interpretation):** [`ENGINE_COMPARE_REPORT.md`](./ENGINE_COMPARE_REPORT.md)  
**Numeric artifacts:** [`results/`](./results/)

## Upstream-style bench (`bench.py` parity)

Mirror [GeeeekExplorer/nano-vllm `bench.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py):

| Knob | Value |
|------|--------|
| `num_seqs` | 256 |
| prompt len | uniform 100–1024 |
| `max_tokens` | uniform 100–1024 |
| `enforce_eager` | **False** (CUDA graphs ON) |
| `max_model_len` | 4096 |
| seed | 0 (+ optional repeats) |
| Attention | matched `vllm_flash` / `FLASH_ATTN` |

```bash
bash experiments/run_upstream_bench.sh
REPEATS=3 bash experiments/run_upstream_bench.sh
```

Upstream README quotes ~1434 vs ~1361 tok/s on an RTX 4070 Laptop — different GPU; use this script for method-matched lab vs vLLM on your box.

## High-load / vLLM-favoring sweep

Push concurrency + decode length under conditions that historically favor vLLM
(same flash, **graphs OFF**, prefix OFF, high `gpu_memory_utilization`):

```bash
bash experiments/run_vllm_load_sweep.sh
# smoke:
REPEATS=1 bash experiments/run_vllm_load_sweep.sh
```

Suite `vllm_load`: wide batches, long decode, long context (see `sweep_engine_compare.py`).  
Note: with graphs **ON**, high offline load on 0.6B often still favors lab (Study E).

## Graph ablation (eager vs CUDA graphs only)

Eager sweep often favors vLLM; full-config SLO often favors lab on offline bs32.  
To test whether **CUDA graphs** alone cause that flip, keep **prefix OFF** and the same shapes/flash, and toggle only `enforce_eager`:

```bash
bash experiments/run_graph_ablation.sh
# faster smoke:
SUITE=graph_ablation REPEATS=1 bash experiments/run_graph_ablation.sh
# single flip-case:
CASES=bs32_in256_out128 REPEATS=3 bash experiments/run_graph_ablation.sh
```

Writes `*_graph_ablation_eager.json`, `*_graph_ablation_graphs.json`, and a merged table JSON.  
See `experiments/merge_graph_ablation.py`.

## Full-config SLO compare (graphs + prefix)

The earlier `run_engine_compare.sh` / sweep used **eager + prefix off** for a controlled same-kernel study.  
To compare **full optimizations** with finer metrics (TTFT / TPOT / RPS / batch / KV):

```bash
bash experiments/run_slo_compare.sh
# or:
WORKLOAD=independent bash experiments/run_slo_compare.sh
WORKLOAD=single_stream NUM_SEQS=16 bash experiments/run_slo_compare.sh
```

| Knob | Full-config default |
|------|---------------------|
| CUDA graphs | ON (`--no-enforce-eager`) |
| Prefix caching | ON (+ prime shared prefix when `shared_prefix`) |
| Attention | matched `vllm_flash` / `FLASH_ATTN` |
| Metrics | TTFT, TPOT, E2E, tok/s, req/s, SLO attainment, batch (lab), KV peak (lab) |

See `experiments/bench_slo_compare.py`.

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
