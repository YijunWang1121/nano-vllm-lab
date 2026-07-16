# Engine compare report: nano-vLLM lab vs vLLM 0.8.5

Branch: `experiments/engine-compare`  
Hardware: NVIDIA RTX 4090 (~24 GB)  
Model: Qwen3-0.6B  
Lab: this branch (reference-level engine + experiment harness)  
vLLM: 0.8.5 (`VLLM_USE_V1=0`)  
Torch: 2.6.0+cu124  

**Artifacts (pod paths; copy into `experiments/results/` when pushing):**

| File | Content |
|------|---------|
| `~/engine_compare_results/20260715_223801_sweep_quick.json` | Eager sweep smoke |
| `~/engine_compare_results/20260715_223901_sweep_default.json` | Eager multi-shape ×3 |
| `~/engine_compare_results/20260715_230715_slo_shared_prefix.json` | Full-config SLO |
| `~/engine_compare_results/20260715_230856_slo_independent.json` | Full-config SLO |
| `~/engine_compare_results/20260715_230954_slo_single_stream.json` | Full-config SLO |
| `experiments/results/20260715_232028_graph_ablation.json` | Graph ON/OFF merge table |
| `experiments/results/20260715_232028_graph_ablation_{eager,graphs}.json` | Per-mode sweeps |
| `experiments/results/20260715_234145_sweep_vllm_load.json` | High-load eager (vLLM-favoring knobs) |
| `experiments/results/20260716_000041_upstream_bench.json` | Upstream `bench.py` parity |
| `experiments/results/*.summary.json` | Committed numeric summary (partial) |

---

## 0. Executive summary

| Study | Config | Headline |
|-------|--------|----------|
| A. Unfair baseline | Lab SDPA vs vLLM Flash | vLLM ~**10×** — **not fair** |
| B. Same-kernel one-shot | Eager, random 64 seqs | Lab ≈ vLLM (**0.94×** vLLM/lab) |
| C. Eager shape sweep | Eager, fixed shapes ×3 | vLLM **1.1–1.7×** on small/mid; lab **0.95×** on one large prefill-ish case |
| D. Full-config SLO | Graphs + prefix + same flash | Offline bs32: lab **~1.35×** tok/s; single-stream: vLLM better **TTFT**, lab better **TPOT** |
| E. Graph ablation | Same shapes/flash; prefix OFF; only graphs ON/OFF | **Confirmed:** eager vLLM ahead; graphs → lab ahead on all 4 shapes |
| F. High-load eager | Eager, util=0.95, long decode / wide / long ctx | vLLM **1.1–1.7×** on decode-heavy mid; lab **0.79–0.81×** on widest / long-prefill |
| G. Upstream `bench.py` | 256 seqs, in/out U(100,1024), graphs ON, seed=0 | Lab **6044** vs vLLM **5190** tok/s (0.86×); same 133966 toks as upstream README |

**Takeaways**

1. Matching FlashAttention binaries removes most of the “10×” myth.  
2. Under **eager**, small/mid + **long decode** favor **vLLM**; pushing concurrency/prefill further can flip back to **lab**.  
3. **Study E:** turning on **CUDA graphs alone** (prefix still off) flips ranking — lab ahead on all tested shapes; lab’s graph speedup ≫ vLLM’s.  
4. Full-config SLO (graphs + prefix) matches that graphs-on story for offline bs32; **vLLM** still wins single-stream **TTFT**.  
5. Students wire Flash APIs; remaining gaps are engine path / fused ops / serving — not “missing FlashAttention.”  
6. **Why (code-level):** see §10 — lab’s thin CUDA-graph decode path vs vLLM’s heavier eager/serving scaffolding.  
7. Upstream-style `bench.py` (Study G) reproduces their token count and ranking direction on 4090.

---

## 1. Fairness controls (what “same” means)

| Control | How |
|---------|-----|
| Same tokens | One shared `prompt_token_ids` + `max_tokens` batch (fingerprint / fixed lens) |
| Same attention library | Lab `NANOVLLM_ATTN_BACKEND=vllm_flash` → `vllm.vllm_flash_attn`; vLLM `FLASH_ATTN` |
| Same graph policy | Explicit `enforce_eager` on/off per study |
| Prefix cache | Explicit on/off; `shared_prefix` primes then times |
| Metric (throughput) | `tok/s = sum(max_tokens) / timed_generate_wall` after warmup |
| SLO metrics | TTFT / TPOT / E2E / req/s / batch / KV (lab) |

**Not claimed identical:** RMSNorm/MLP/sampler fusion, serving APIs, CPU KV swap (lab preempt = recompute only).

---

## 2. Study A — Unfair baseline (historical)

**Config**

- Lab: `NANOVLLM_ATTN_BACKEND=torch` (SDPA), `enforce_eager=True`  
- vLLM: Flash Attention, `enforce_eager=True`  
- Workload: ~64 seqs, random lengths (from early `run_engine_compare.sh`)

**Result (approx.)**

- Lab ~460 tok/s, vLLM ~4900 tok/s → **~10×**  

**Interpretation:** Different attention backends. Do **not** cite as a fair engine comparison.

---

## 3. Study B — Same-kernel one-shot (eager)

**Config**

| Item | Value |
|------|--------|
| Attention | `vllm_flash` / `FLASH_ATTN` (same `.so`) |
| CUDA graphs | OFF (`enforce_eager=True`) |
| Prefix caching | OFF |
| `gpu_memory_utilization` | 0.85 |
| `max_model_len` | 4096 |
| Workload | `num_seqs=64`, random prompt len 32–512, random `max_tokens` 16–128, `seed=0` |
| Observed load | `total_tokens=4931` |

**Result**

| Engine | tok/s | wall | vs lab |
|--------|------:|-----:|-------:|
| lab | **3473** | 1.42 s | 1.00× |
| vLLM 0.8.5 | 3251 | 1.52 s | **0.94×** |

**Interpretation:** With matched Flash and a large random offline batch, lab can slightly beat vLLM in eager mode. Single-run noise; direction matches Study C’s large-batch cases.

---

## 4. Study C — Eager multi-shape sweep (primary controlled study)

**Harness:** `bash experiments/run_engine_sweep.sh` (`SUITE=default`, `REPEATS=3`)  
**Config:** same as Study B (eager, prefix off, `vllm_flash`, util=0.85)  
**Shapes:** fixed `bs / in / out` per case; 3 seeds → mean±std  
**Wall:** ~650 s  
**JSON:** `20260715_223901_sweep_default.json`

### 4.1 Results

| Case | bs | in | out | Lab tok/s | vLLM tok/s | vs lab |
|------|---:|---:|----:|----------:|-----------:|-------:|
| `bs1_in128_out256` | 1 | 128 | 256 | 106.2 ± 6.4 | 159.8 ± 14.2 | **1.50×** |
| `bs8_in128_out256` | 8 | 128 | 256 | 751.1 ± 50.6 | 1240.5 ± 138.7 | **1.65×** |
| `bs8_in1024_out32` | 8 | 1024 | 32 | 674.4 ± 4.9 | 989.4 ± 3.9 | **1.47×** |
| `bs32_in64_out64` | 32 | 64 | 64 | 3092.0 ± 63.8 | 4498.7 ± 580.7 | **1.46×** |
| `bs32_in256_out128` | 32 | 256 | 128 | 2785.0 ± 209.2 | 4568.6 ± 48.1 | **1.64×** |
| `bs64_in256_out128` | 64 | 256 | 128 | 5569.4 ± 153.0 | 6623.9 ± 247.8 | **1.19×** |
| `bs16_in2048_out64` | 16 | 2048 | 64 | 1025.4 ± 76.6 | 1129.0 ± 18.5 | **1.10×** |
| `bs64_in512_out32` | 64 | 512 | 32 | 3454.0 ± 26.4 | 3279.0 ± 12.9 | **0.95×** |

Smoke (`quick`, 1 repeat): bs8 ~1.85×, bs32 ~1.78× (same direction, higher noise).

### 4.2 Interpretation

- Small batch / decode-heavy → **vLLM** (per-step overhead).  
- Larger batch / longer prefill → gap shrinks; one case **lab ahead**.  
- Some vLLM stds large on short walls — use trends, not exact ratios.

---

## 5. Study D — Full-config SLO compare

**Harness:** `bash experiments/run_slo_compare.sh`  
**Config (both engines unless noted)**

| Knob | Value |
|------|--------|
| CUDA graphs | **ON** (`enforce_eager=False`) |
| Prefix caching | **ON** |
| Attention | `vllm_flash` / `FLASH_ATTN` |
| Lab preempt / chunked prefill | ON (defaults) |
| `gpu_memory_utilization` | 0.9 |
| `max_model_len` | 4096 |
| Sampling | `temperature=0.6`, `ignore_eos=True` |
| SLO thresholds | TTFT &lt; 500 ms, TPOT &lt; 50 ms |

**Metric definitions**

| Metric | Definition |
|--------|------------|
| TTFT | Admit → first output token |
| TPOT | `(finish − first) / (out_tokens − 1)` |
| E2E | Admit → finish |
| output tok/s | `total_output_tokens / wall` |
| req/s | `num_requests / wall` |
| batch (lab) | Scheduled batch size per step |
| KV (lab) | `num_blocks`, peak used blocks / frac |

Lab: step-loop instrumentation. vLLM: `RequestOutput.metrics` (slightly different clocks; direction OK).

### 5.1 Workload: `shared_prefix`

- 32 seqs, prefix 512 + suffix 128, out 128; prime shared prefix before timed run  
- JSON: `20260715_230715_slo_shared_prefix.json`

| | lab | vLLM |
|--|-----|------|
| wall | 0.550 s | 0.744 s |
| out tok/s | **7446** | 5502 |
| req/s | **58.2** | 43.0 |
| TTFT p50 (ms) | **37.1** | 49.2 |
| TPOT p50 (ms) | **4.04** | 5.44 |
| E2E p50 (ms) | **550** | 741 |
| SLO TTFT/TPOT | 100% / 100% | 100% / 100% |
| batch | mean/max 32, 128 steps | n/a |
| KV peak | 34 / 697 (~4.9%) | (init logs only) |

Lab throughput ≈ **1.35×** vLLM.

### 5.2 Workload: `independent`

- 32 seqs, in 256, out 128, **no** shared prefix (prefix flag still on but no hits expected)  
- JSON: `20260715_230856_slo_independent.json`

| | lab | vLLM |
|--|-----|------|
| wall | 0.533 s | 0.725 s |
| out tok/s | **7685** | 5651 |
| req/s | **60.0** | 44.1 |
| TTFT p50 (ms) | 64.5 | 64.5 |
| TTFT p90 (ms) | 64.5 | **80.8** (bimodal) |
| TPOT p50 (ms) | **3.69** | 5.17 |
| E2E p50 (ms) | **533** | 721 |
| KV peak | 64 / 697 (~9.2%) | — |

Lab still ~**1.36×** tok/s → advantage is **not** only from prefix reuse.

### 5.3 Workload: `single_stream`

- 16 requests, one at a time, in 256, out 128  
- JSON: `20260715_230954_slo_single_stream.json`

| | lab | vLLM |
|--|-----|------|
| wall (sum) | 4.73 s | 6.44 s |
| out tok/s | **433** | 318 |
| req/s | **3.39** | 2.49 |
| TTFT p50 (ms) | 12.1 | **7.9** |
| TPOT p50 (ms) | **2.18** | 2.98 |
| E2E p50 (ms) | **289** | 387 |
| batch | 1.0 | n/a |
| KV peak | 2 / 697 | — |

- **vLLM wins TTFT** (first token).  
- **lab wins TPOT / E2E / tok/s** on this sequential offline loop.  
- Throughput numbers are low because concurrency is 1 (not comparable to bs32 tok/s).

### 5.4 Full-config interpretation

| Scenario | Favors |
|----------|--------|
| Offline concurrent decode (bs32), graphs on | **lab** throughput / TPOT / E2E |
| Single-stream first token | **vLLM** TTFT |
| Eager small-batch (Study C) | **vLLM** |

Do not claim “lab beats vLLM everywhere.” Claim: **on this model/GPU, matched flash + graphs, offline batched decode often favors the teaching engine; latency-first TTFT still favors vLLM.**

---

## 6. Study E — Graph ablation (CUDA graphs only)

**Hypothesis:** Study C (eager) vs Study D (full) reverse because of **CUDA graphs**, not prefix or Flash.  
**Result:** **Confirmed.**

**Harness:** `bash experiments/run_graph_ablation.sh`  
**Controls (both modes):** same `vllm_flash` / `FLASH_ATTN`, **prefix OFF**, util=0.85, 4 shapes × 3 seeds  
**Only knob:** `enforce_eager=True` (eager) vs `False` (graphs)  
**JSON:** `experiments/results/20260715_232028_graph_ablation.json`

| Case | lab eager | lab graphs | lab× | vLLM eager | vLLM graphs | vLLM× | vs_lab eager | vs_lab graphs |
|------|----------:|-----------:|-----:|-----------:|------------:|------:|-------------:|--------------:|
| `bs1_in128_out256` | 102.7±7.4 | 446.8±2.4 | **4.35×** | 167.0±17.1 | 347.2±0.8 | 2.08× | **1.63×** | **0.78×** |
| `bs8_in128_out256` | 757.0±17.8 | 3026.3±9.4 | **4.00×** | 1283.3±199.5 | 2090.3±101.3 | 1.63× | **1.70×** | **0.69×** |
| `bs32_in256_out128` | 2972.8±1.1 | 7505.6±250.2 | **2.52×** | 4153.6±339.7 | 5747.6±52.7 | 1.38× | **1.40×** | **0.77×** |
| `bs64_in512_out32` | 3354.1±189.2 | 4228.0±35.0 | **1.26×** | 3276.2±37.6 | 3309.2±185.7 | 1.01× | **0.98×** | **0.78×** |

`lab×` / `vllm×` = tok/s(graphs) / tok/s(eager). `vs_lab` = vLLM / lab.

### Interpretation

- Eager column matches Study C (vLLM ahead on small/mid; near-parity on large prefill-ish).  
- Turning graphs **on** (prefix still off) flips ranking on every case: lab faster (`vs_lab` 0.69–0.78×).  
- Lab’s graph speedup is much larger (up to ~4×) than vLLM’s (~1–2×); on `bs32_in256_out128` lab ~2.5× vs vLLM ~1.4×.  
- Explains Study D’s offline bs32 lab win without needing prefix cache.

## 7. Study F — High-load eager (vLLM-favoring knobs)

**Goal:** Raise load as much as practical while keeping knobs that historically help vLLM.  
**Harness:** `bash experiments/run_vllm_load_sweep.sh`  
**Config:** `enforce_eager=True`, prefix OFF, `vllm_flash` / `FLASH_ATTN`, `gpu_memory_utilization=0.95`, ×3 seeds  
**JSON:** `experiments/results/20260715_234145_sweep_vllm_load.json`

| Case | bs | in | out | Lab tok/s | vLLM tok/s | vs_lab |
|------|---:|---:|----:|----------:|-----------:|-------:|
| `bs8_in128_out1024` | 8 | 128 | 1024 | 740.2 ± 8.5 | 1278.2 ± 86.3 | **1.73×** |
| `bs32_in128_out512` | 32 | 128 | 512 | 3006.6 ± 81.6 | 4506.4 ± 517.9 | **1.50×** |
| `bs64_in256_out256` | 64 | 256 | 256 | 5665.2 ± 290.9 | 6272.3 ± 662.8 | **1.11×** |
| `bs16_in2048_out256` | 16 | 2048 | 256 | 1342.1 ± 90.2 | 1532.1 ± 82.0 | **1.14×** |
| `bs64_in1024_out128` | 64 | 1024 | 128 | 3889.6 ± 117.1 | 3075.1 ± 274.1 | **0.79×** |
| `bs128_in256_out128` | 128 | 256 | 128 | 8971.7 ± 262.9 | 7314.0 ± 64.3 | **0.81×** |

### Interpretation

- **vLLM advantage peaks** on mid-width × **long decode** (many eager steps): up to **~1.7×**.  
- Pushing to **very wide** (`bs128`) or **long-prefill wide** (`bs64_in1024`) flips to **lab** even with graphs OFF.  
- So “加负载” alone不会让 vLLM 全面领先；在这个 0.6B offline 设定里，vLLM 赢在 **eager decode 步数多、batch 还没拉到极端**。

## 8. Study G — Upstream `bench.py` parity

**Ref:** [GeeeekExplorer/nano-vllm `bench.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py) / [README benchmark](https://github.com/GeeeekExplorer/nano-vllm)  
**Harness:** `bash experiments/run_upstream_bench.sh`  
**JSON:** `experiments/results/20260716_000041_upstream_bench.json`

| Knob | Upstream | Our run |
|------|----------|---------|
| `num_seqs` | 256 | 256 |
| prompt / out len | U(100, 1024) | same (`seed=0`) |
| `enforce_eager` | False | False (graphs ON) |
| `max_model_len` | 4096 | 4096 |
| `gpu_memory_utilization` | (default) | 0.9 |
| Attention | (their env) | matched `vllm_flash` / `FLASH_ATTN` |
| Prefix | default | OFF (no hits on random ids) |
| Hardware | RTX 4070 Laptop 8GB (README) | RTX 4090 ~24GB |

| | Output tokens | Time | tok/s | vs lab |
|--|--------------:|-----:|------:|-------:|
| **lab (this run)** | 133966 | 22.16 s | **6044** | 1.00× |
| **vLLM 0.8.5** | 133966 | 25.81 s | 5190 | **0.86×** |
| Upstream README nano | 133966 | 93.41 s | 1434 | — |
| Upstream README vLLM | 133966 | 98.37 s | 1362 | — |

`seed=0` reproduced the **same total output tokens (133966)** as the upstream README table. Absolute tok/s are higher on 4090; **direction matches** their claim (nano/lab slightly ahead of vLLM under graphs-on offline batch).

## 9. Cross-study comparison (vs_lab = vLLM / lab)

| Shape / study | Eager (C/E) vs_lab | Graphs only (E) vs_lab | Full-config (D) |
|---------------|-------------------:|-----------------------:|-----------------|
| Small bs / latency-like | 1.5–1.7× | **0.7–0.8×** (lab ahead) | Single-stream: vLLM better TTFT; lab better TPOT |
| Mid bs32 | ~1.4–1.6× | **~0.77×** | lab ~1.35× tok/s (prefix on) |
| Large / prefill-heavy | ~1.0× | **~0.78×** | Aligns with graphs-on lab win |

**CUDA graphs** are the knob that flips offline throughput ranking on this 0.6B / 4090 setup.

---

## 10. Why — implementation-level explanation

Fair compares use the **same FlashAttention binary** (`vllm_flash_attn`). Ranking flips are therefore about **everything around attention**: step loop, CUDA-graph capture shape, scheduler/metadata cost, and how much fixed CPU/launch overhead remains when the model is tiny (0.6B on 4090).

### 10.1 Shared baseline (what is *not* the difference)

| Piece | Lab | vLLM (these runs) |
|-------|-----|-------------------|
| Attention .so | `vllm.vllm_flash_attn` | `VLLM_ATTENTION_BACKEND=FLASH_ATTN` → same package |
| Workload | Shared `prompt_token_ids` + `max_tokens` | Same |
| API style | Offline `LLM.generate` batch | Offline `LLM.generate` batch |

So “lab faster” here does **not** mean “we wrote a better Flash kernel.”

### 10.2 Where lab is stronger — and why in code

#### A. CUDA graphs ON → lab wins almost everywhere (Studies D, E)

Lab’s graph path is intentionally minimal (`nanovllm/engine/model_runner.py`):

1. **Capture only decode forward:** `outputs[:bs] = self.model(input_ids[:bs], positions[:bs])` inside `torch.cuda.graph(...)`.
2. **Fixed padded buffers** (`input_ids`, `positions`, `slot_mapping`, `context_lens`, `block_tables`) written each step, then `graph.replay()`.
3. **Logits + sampling stay outside the graph:** `compute_logits(...)` then a tiny `Sampler` (Gumbel-max, no `@torch.compile`).
4. **Prefill never uses graphs** (`is_prefill` → eager). Decode with `bs > 512` also falls back to eager.

Why that helps lab more than vLLM on this setup:

- On 0.6B, each decode step’s GPU work is short. Cost is dominated by **kernel launches + Python/engine bookkeeping**.
- Lab’s captured region is basically **one thin model forward** over a teaching stack (Attention → Flash call, simple KV slot mapping). Replay collapses many launches into one graph launch.
- Study E numbers match this story: lab’s graphs/eager speedup is **~2.5–4.4×** on mid/small decode shapes; vLLM’s is only **~1.4–2.1×**. Same flash, different **graph ROI**.
- vLLM 0.8.5 V0 still carries a much thicker worker/model-runner path (more metadata, custom op boundaries, serving-oriented bookkeeping). Graphs help, but they do not erase as much relative overhead as they do for lab’s short path — so after both enable graphs, **lab’s lighter decode loop wins offline tok/s**.

#### B. Very wide batch / long-prefill-ish (Studies C, F) — lab can win even in eager

Examples: `bs64_in512_out32`, `bs64_in1024_out128`, `bs128_in256_out128`.

Implementation reading:

- **Prefill is always eager** on lab; wide/long prompts spend most wall time in FlashAttention + big matmuls. Per-step framework tax is a smaller fraction → vLLM’s eager scaffolding edge shrinks.
- Lab’s offline loop is simple: `Scheduler.schedule` → `prepare_prefill`/`prepare_decode` → `run_model` → `sampler`. For an all-at-once batch, there is little continuous-batching / arrival / policy work to do.
- vLLM’s production stack is built for **serving** (continuous batching, richer request state, more validation/stats paths). In a synthetic offline dump of N equal sequences, that extra machinery is mostly **fixed cost**, not a win — and at extreme width it shows up as lower tok/s than lab.

#### C. Steady decode latency (TPOT), not first token (Study D single-stream)

After the first token, lab’s decode is again the thin `prepare_decode` + (graph) forward + cheap sampler. Measured **TPOT / E2E / tok/s** favor lab; **TTFT** still favors vLLM (see §10.3).

### 10.3 Where vLLM is stronger — and why

#### A. Eager + mid batch + long decode (Studies C, F)

Many decode steps × modest batch (e.g. `bs8_out1024`, `bs32_out512`) → **1.5–1.7×** for vLLM.

Without graphs, every layer is a separate launch. vLLM’s years of work show up as **lower per-step overhead** even when attention .so matches: tighter worker loop, more fused / custom ops around the stack, less naive Python/tensor churn than a teaching engine. Lab still pays full eager tax on RMSNorm/MLP/etc. boundaries (even with some `@torch.compile` on Silu/RoPE/RMSNorm).

#### B. First-token latency (TTFT)

Single-stream Study D: vLLM TTFT better, lab TPOT better.

Plausible implementation reasons (not a full vLLM profile):

- Prefill→decode transition and first decode step are more optimized in vLLM.
- Lab always runs **logits outside** the CUDA graph and uses a Python `.tolist()` sampling path on the critical step boundary.
- Clocks differ slightly (lab step instrumentation vs vLLM `RequestOutput.metrics`), but direction is stable.

#### C. Unfair SDPA vs Flash (Study A)

Not an engine win — different attention backends. Listed only so it is not confused with §10.2.

### 10.4 One diagram (cost model)

```text
wall ≈ (prefill compute) + N_decode * (framework/step + launch/step + GPU/step)

0.6B decode:  GPU/step small
eager:        framework+launch dominate → vLLM (thicker but more optimized step) wins mid/long decode
graphs:       launch collapses; lab's thin captured forward + light scheduler wins offline batch
wide/prefill: GPU/prefill dominates → overhead gaps shrink; lab's simpler offline path can win
```

### 10.5 What this means for the course

| Claim | Supported by |
|-------|----------------|
| Students need not reimplement Flash to “catch vLLM” | Same .so fair runs |
| Remaining gap is engine path / fusion / serving | Eager mid-decode still favors vLLM |
| Enabling CUDA graphs is a first-class lab milestone | Study E flips ranking without prefix |
| “Lab beats vLLM” is **conditional**, not absolute | Eager vs graphs; TTFT vs TPOT; mid vs extreme width |

---

## 11. Lab features relevant to results

| Feature | Behavior in these studies |
|---------|---------------------------|
| FlashAttention | Piped to `vllm_flash_attn` or pip `flash_attn` |
| Prefix cache | OFF in C; ON (+ prime) in D `shared_prefix` |
| CUDA graphs | OFF in B/C/F; ON in D/E-graphs |
| Preemption | Recompute (free GPU KV), **not** CPU swap |
| Chunked prefill | ON by default in D lab |
| Graph capture | Decode-only `model()` into `torch.cuda.CUDAGraph`; logits/sampler outside |

`vllm_flash` + prefix-cache prefill needs `seqused_k` without `cu_seqlens_k` (fixed on this branch).

---

## 12. Course implications

| Gap | Meaning |
|-----|---------|
| Student → reference | TODOs: schedule, KV, metadata, **call** flash APIs, sampler, graphs, TP |
| Reference → vLLM (fair) | Fusion, serving, policies — not “implement Flash from scratch” |
| How to re-bench | Eager sweep for scaffolding; graph ablation for graph ROI; SLO for full-config |

---

## 13. Limitations

- Single model (0.6B), single GPU SKU.  
- Offline batch / sequential offline — not OpenAI-server QPS or Poisson arrivals.  
- vLLM internals not line-profiled here; §10 reasons are grounded in **lab code + measured ROI**, not a full vLLM audit.  
- vLLM batch/KV peak not fully exposed in offline API.  
- TTFT clocks differ slightly (lab step vs vLLM metrics).  
- Short walls → variance (esp. eager small shapes).  
- Experiment-branch attention backends (`vllm_flash` / SDPA) are for compare; teaching `course/*` stays flash-centric for students.

---

## 14. Reproduce

```bash
cd /nano-vllm-lab
git switch experiments/engine-compare && git pull --rebase origin experiments/engine-compare
source ~/venv-nanovllm/bin/activate
export PYTHONPATH=/nano-vllm-lab
export NANOVLLM_TEST_MODEL=/root/huggingface/Qwen3-0.6B
unset NANOVLLM_ATTN_BACKEND

# Study C (eager, shapes)
SUITE=default REPEATS=3 bash experiments/run_engine_sweep.sh

# Study E (graphs ON/OFF only; prefix OFF)
bash experiments/run_graph_ablation.sh

# Study F (high-load eager, vLLM-favoring knobs)
bash experiments/run_vllm_load_sweep.sh

# Study G (upstream bench.py parity)
bash experiments/run_upstream_bench.sh

# Study D (full config SLO)
bash experiments/run_slo_compare.sh
WORKLOAD=independent bash experiments/run_slo_compare.sh
WORKLOAD=single_stream NUM_SEQS=16 bash experiments/run_slo_compare.sh
```

Docs: `experiments/ENGINE_COMPARE.md`.

---

## 15. Suggested slide narrative

1. Unfair SDPA vs Flash → 10× (dismiss).  
2. Same Flash + eager → vLLM wins mid/long-decode; extreme wide/prefill can flip to lab.  
3. Graph ablation → graphs alone flip ranking to lab (thin capture vs thick serving path).  
4. Upstream `bench.py` parity → same 133966 toks; lab ahead under graphs (matches their README direction).  
5. Same Flash + graphs + prefix → offline bs32 lab faster; single-stream vLLM TTFT.  
6. Implementation takeaway (§10): same attention .so; ranking = step overhead × graph ROI.  
7. Course goal: understand *when* each engine wins, not a single number.
