# Engine compare report: nano-vLLM lab vs vLLM 0.8.5

Branch: `experiments/engine-compare`  
Primary artifact: `experiments/results/20260715_223901_sweep_default.summary.json`  
Full pod JSON (when copied): `experiments/results/20260715_223901_sweep_default.json`

## 1. Executive summary

Under a **fair, same-kernel** setup (shared `vllm_flash_attn` binary, `enforce_eager=True`, prefix caching off, identical fixed-length token batches), production **vLLM 0.8.5** is typically **1.1–1.7×** faster than this course **lab** engine on Qwen3-0.6B / RTX 4090.

The gap is **workload-dependent**:

- Larger on small-batch / decode-heavy shapes (~1.5–1.7×).
- Smaller on larger batches and long-context / prefill-heavy shapes (~1.1–1.2×).
- On one prefill-leaning large-batch case (`bs64_in512_out32`), lab is slightly ahead (**0.95×** vLLM vs lab).

**Takeaway:** remaining differences are mostly engine scaffolding and non-attention fused ops, not “missing FlashAttention.” Course students wire FlashAttention APIs; they do not reimplement the kernel.

## 2. Hardware and software

| Item | Value |
|------|--------|
| GPU | NVIDIA RTX 4090 (~24 GB) |
| Model | Qwen3-0.6B (`NANOVLLM_TEST_MODEL`) |
| Lab | This branch (reference-level engine + experiment harness) |
| vLLM | 0.8.5, `VLLM_USE_V1=0` |
| Torch | 2.6.0+cu124 |
| Attention | Lab: `NANOVLLM_ATTN_BACKEND=vllm_flash` → `vllm.vllm_flash_attn`; vLLM: `VLLM_ATTENTION_BACKEND=FLASH_ATTN` |
| CUDA graphs | Disabled on both (`enforce_eager=True`) |
| Prefix caching | Disabled on both |
| `gpu_memory_utilization` | 0.85 |
| `max_model_len` | 4096 |
| Tensor parallel | 1 |

## 3. Method

### 3.1 Fairness controls

1. **Same tokens:** one synthetic batch of `prompt_token_ids` + per-sequence `max_tokens` is generated once and sent to every engine (fingerprint logged as `sha256`).
2. **Same attention library:** lab imports vLLM’s bundled FlashAttention (not a separate pip `flash_attn` `.so`).
3. **Same graph policy:** eager on both for this report.
4. **Fixed shapes:** each suite case uses constant `num_seqs`, `input_len`, `output_len` (not random lengths).
5. **Repeats:** `default` suite uses 3 seeds; report **mean ± std** tok/s.

### 3.2 Metric

\[
\mathrm{tok/s} = \frac{\sum_i \mathrm{max\_tokens}_i}{T_{\mathrm{timed\ generate}}}
\]

after a short warmup generate. This matches `bench.py`-style throughput and is **not** comparable to Prefill/Decode rates printed by `example.py` progress bars.

### 3.3 How to reproduce

```bash
cd /nano-vllm-lab
git switch experiments/engine-compare
source ~/venv-nanovllm/bin/activate
export PYTHONPATH=/nano-vllm-lab
export NANOVLLM_TEST_MODEL=/root/huggingface/Qwen3-0.6B   # or your path
unset NANOVLLM_ATTN_BACKEND   # auto → vllm_flash when available

# smoke
SUITE=quick REPEATS=1 bash experiments/run_engine_sweep.sh

# this report
SUITE=default REPEATS=3 bash experiments/run_engine_sweep.sh
```

Harness: `experiments/sweep_engine_compare.py`, wrapper `experiments/run_engine_sweep.sh`.

## 4. Results (`default` suite, 3 repeats)

Wall time for the full sweep on the pod: **~650 s**.

| Case | bs | in | out | Intent | Lab tok/s | vLLM tok/s | vs lab |
|------|---:|---:|----:|--------|----------:|-----------:|-------:|
| `bs1_in128_out256` | 1 | 128 | 256 | single-stream decode-heavy | 106.2 ± 6.4 | 159.8 ± 14.2 | **1.50×** |
| `bs8_in128_out256` | 8 | 128 | 256 | small-batch decode-heavy | 751.1 ± 50.6 | 1240.5 ± 138.7 | **1.65×** |
| `bs8_in1024_out32` | 8 | 1024 | 32 | prefill-heavy | 674.4 ± 4.9 | 989.4 ± 3.9 | **1.47×** |
| `bs32_in64_out64` | 32 | 64 | 64 | short prompts | 3092.0 ± 63.8 | 4498.7 ± 580.7 | **1.46×** |
| `bs32_in256_out128` | 32 | 256 | 128 | balanced mid | 2785.0 ± 209.2 | 4568.6 ± 48.1 | **1.64×** |
| `bs64_in256_out128` | 64 | 256 | 128 | larger batch | 5569.4 ± 153.0 | 6623.9 ± 247.8 | **1.19×** |
| `bs16_in2048_out64` | 16 | 2048 | 64 | long context | 1025.4 ± 76.6 | 1129.0 ± 18.5 | **1.10×** |
| `bs64_in512_out32` | 64 | 512 | 32 | large batch, short decode | 3454.0 ± 26.4 | 3279.0 ± 12.9 | **0.95×** |

Supporting smoke (`quick`, 1 repeat): bs8 ~1.85×, bs32 ~1.78× — same direction, higher noise.

### 4.1 Earlier single-shot (context only)

A prior one-off `run_engine_compare.sh` run used **random** lengths (`num_seqs=64`, prompt 32–512, max_tokens 16–128, seed=0, ~4931 output tokens) with the same `vllm_flash` matching. Lab was slightly ahead (~3473 vs ~3251 tok/s, **0.94×**). That is consistent with the sweep’s large-batch / compute-heavy regime, but is **not** the primary result of this report (no multi-seed aggregation).

### 4.2 Unfair baselines (do not cite as fair)

| Setup | Observed | Why unfair |
|-------|----------|------------|
| Lab SDPA/torch vs vLLM Flash | ~10× for vLLM | Different attention backends |
| Lab pip `flash_attn` vs vLLM `vllm_flash_attn` | Ambiguous | Different shared libraries |

## 5. Interpretation

1. **Small jobs amplify scaffolding.** With bs=1/8 and short wall clocks, per-step Python scheduling, metadata build, sampling, and non-fused LayerNorm/MLP dominate. vLLM invests heavily here; the teaching engine stays readable.
2. **When kernels dominate, gaps shrink.** Larger batches and longer prefills spend more time in FlashAttention / matmuls. Same `.so` → ratios approach ~1.0–1.2×; one case flips slightly for lab.
3. **Variance.** Some vLLM stds are large (e.g. `bs8_in128_out256`, `bs32_in64_out64`) on short timed windows; treat ratios as trends, not precise constants.
4. **Preemption note.** Lab preempt is **recompute** (free GPU KV, re-prefill), not CPU KV swap. This suite was not designed to stress KV exhaustion.

## 6. Course implications (student vs reference vs vLLM)

| Gap | What it is |
|-----|------------|
| Student → reference | Implement TODOs (scheduler, KV blocks, metadata, **call** flash-attn APIs, sampler, CUDA graphs, TP). Not “write FlashAttention from scratch.” |
| Reference/lab → vLLM (fair) | Fused non-attention ops, richer scheduler/serving, graph/compile stack, etc. |
| How to re-bench later | Same sweep harness; first match reference, then vs vLLM with `vllm_flash` + fixed shapes × repeats. |

## 7. Limitations

- Single model (0.6B) and single GPU SKU.
- Eager-only; CUDA graphs and compile paths not compared here.
- Offline batch throughput only; no OpenAI-server / streaming QPS.
- Lab on this branch includes experiment-only attention backend selection (`vllm_flash` / SDPA); teaching `course/*` branches stay flash-attn–centric for students.

## 8. Artifacts checklist

| Path | Role |
|------|------|
| `experiments/ENGINE_COMPARE_REPORT.md` | This report |
| `experiments/ENGINE_COMPARE.md` | How to run harness |
| `experiments/results/*.summary.json` | Committed numeric summary |
| `experiments/results/*_sweep_*.json` | Full pod dumps (copy then commit) |
| `experiments/compare_engines.py` / `sweep_engine_compare.py` | Measurement code |

## 9. Suggested figure for slides

Plot **vs_lab** (y) vs case ordered by increasing batch / prefill weight, or grouped bars of lab vs vLLM tok/s. Caption: “Same FlashAttention binary, eager, fixed shapes, n=3.”
