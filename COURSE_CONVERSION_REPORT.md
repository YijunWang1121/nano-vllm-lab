# Course Conversion Report

## Baseline

| Item | Value |
|---|---|
| Baseline tag | `baseline-before-course-conversion` |
| Baseline commit SHA | `bb823b3e06983d71485a8e1f23715ebd87d98ef8` |
| Upstream remote | `git@github.com:GeeeekExplorer/nano-vllm.git` |
| Origin (fork) | `git@github.com:YijunWang1121/nano-vllm.git` |
| Upstream alignment | local `main` matched `upstream/main` at conversion time |

## Branches

| Branch | SHA (at conversion) | Role |
|---|---|---|
| `course/reference-solutions` | `e964f8a0bf11e2e6d67558ed445b320f01c48adb` | Complete implementation + course docs/tests/tools |
| `course/student` | `0d1eb842ec1418ebf9c8d1d270bebf345753c988` | Same architecture with excavated TODOs |

**Do not implement on the reference branch.** Work only on `course/student`.

Compare a file without overwriting your work:

```bash
git diff course/student..course/reference-solutions -- nanovllm/engine/scheduler.py
```

## Original architecture summary

nano-vLLM is a compact offline inference engine (~1.3k LOC) with:

- **API:** `nanovllm.LLM` → `LLMEngine.generate/add_request/step`
- **Scheduling:** waiting/running deques, prefill-first continuous batching, chunked prefill, preempt
- **Memory:** paged KV `BlockManager` with prefix caching (xxhash) and ref-counts
- **Execution:** `ModelRunner` prepares prefill/decode metadata, runs Qwen3, samples tokens
- **Attention:** FlashAttention varlen (prefill) / with_kvcache (decode) + Triton `store_kvcache`
- **Optimizations:** CUDA graphs (decode), tensor parallelism (NCCL + SharedMemory), torch compile historically on sampler
- **Model support:** Qwen3 only in-tree

See `docs/architecture.md` for Mermaid call graphs.

## TODO inventory (23 registered)

| ID | Level | Milestone | File | Symbol |
|---|---|---|---|---|
| TODO-L1-SEQ-01 | L1 | 1 | `sequence.py` | `Sequence.__init__` |
| TODO-L1-SEQ-02 | L1 | 1 | `sequence.py` | `Sequence.append_token` |
| TODO-L1-SEQ-03 | L1 | 1 | `sequence.py` | block helpers |
| TODO-L2-SCHED-01 | L2 | 2 | `scheduler.py` | `_schedule_prefill` |
| TODO-L2-SCHED-02 | L2 | 3 | `scheduler.py` | `_schedule_decode` |
| TODO-L2-SCHED-03 | L2 | 3 | `scheduler.py` | `preempt` |
| TODO-L2-SCHED-04 | L2 | 3 | `scheduler.py` | `postprocess` |
| TODO-L2-KVCACHE-01 | L2 | 4 | `block_manager.py` | `_allocate/_deallocate_block` |
| TODO-L3-KVCACHE-02 | L3 | 4 | `block_manager.py` | `can_allocate` |
| TODO-L3-KVCACHE-03 | L3 | 4 | `block_manager.py` | `allocate` |
| TODO-L2-KVCACHE-04 | L2 | 4 | `block_manager.py` | `deallocate` |
| TODO-L2-KVCACHE-05 | L2 | 4 | `block_manager.py` | `can_append/may_append` |
| TODO-L3-KVCACHE-06 | L3 | 4 | `block_manager.py` | `hash_blocks` |
| TODO-L2-RUNNER-01 | L2 | 5 | `input_metadata.py` | `build_block_tables` |
| TODO-L2-RUNNER-02 | L2 | 5 | `input_metadata.py` | `build_prefill_metadata` |
| TODO-L2-RUNNER-03 | L2 | 5 | `input_metadata.py` | `build_decode_metadata` |
| TODO-L2-ATTN-01 | L2 | 6 | `attention.py` | `Attention.forward` |
| TODO-L1-SAMPLE-01 | L1 | 7 | `sampler.py` | `Sampler.forward` |
| TODO-L2-ENGINE-01 | L2 | 8 | fake runner contract | integration understanding |
| TODO-L3-CUDAGRAPH-01 | L3 | 10 | `model_runner.py` | graph replay path |
| TODO-L3-CUDAGRAPH-02 | L3 | 10 | `model_runner.py` | `capture_cudagraph` |
| TODO-L2-TP-01 | L2 | 11 | `linear.py` | `ColumnParallelLinear.weight_loader` |
| TODO-L3-TP-02 | L3 | 11 | `linear.py` | `RowParallelLinear.forward` |

Registry: `course/todos.yaml`

## Milestone inventory

| # | Title | CPU/GPU | Est. hours |
|---|---|---|---|
| 1 | Sequence state | CPU | 1.5 |
| 2 | Scheduler basic / prefill | CPU | 2.5 |
| 3 | Prefill/decode + postprocess | CPU | 3 |
| 4 | KV-cache blocks + prefix hash | CPU | 6 |
| 5 | Model input metadata | CPU | 3 |
| 6 | Attention metadata integration | CPU* | 2 |
| 7 | Sampling | CPU | 1 |
| 8 | Fake-engine continuous batching | CPU | 2 |
| 9 | Eager GPU e2e | GPU | 2 |
| 10 | CUDA graphs | GPU | 5 |
| 11 | Tensor parallelism | CPU (unit) / GPU (full) | 4 |

\* Milestone 6 validates Context contracts on CPU; full Attention kernels need GPU + flash-attn.

Total estimated effort: **~40–60 hours**.

## Functions excavated (student)

- `Sequence.__init__`, `append_token`, `num_blocks`, `last_block_num_tokens`, `block`
- `Scheduler._schedule_prefill`, `_schedule_decode`, `preempt`, `postprocess`
- `BlockManager._allocate_block`, `_deallocate_block`, `can_allocate`, `allocate`, `deallocate`, `can_append`, `may_append`, `hash_blocks`
- `build_block_tables`, `build_prefill_metadata`, `build_decode_metadata`
- `Attention.forward` (path selection only)
- `Sampler.forward`
- `ModelRunner` CUDA graph replay + `capture_cudagraph` (eager path kept)
- `ColumnParallelLinear.weight_loader`, `RowParallelLinear.forward`

## Intentionally left implemented

- `LLMEngine` control loop / `generate`
- Config, SamplingParams, tokenizer glue
- Qwen3 model layers, RoPE, RMSNorm, activations
- Triton `store_kvcache` kernel and FlashAttention callsites (student wires them)
- Weight loading driver (`load_model`) except TP shard exercises in linear layers
- TP process group / SharedMemory orchestration in `ModelRunner`
- Packaging, examples, bench

Approximate excavation by logical importance: **~25–30%** of inference-engine decision logic.

## Files changed (course layer)

- `COURSE.md`, `COURSE_CONVERSION_REPORT.md`
- `course/todos.yaml`, `course/milestones.yaml`
- `docs/architecture.md`, `docs/tutorial/*`, `docs/reference/*`
- `nanovllm/course/*`, `nanovllm/utils/debug.py`, `nanovllm/engine/input_metadata.py`
- `tools/course_status.py`, `tools/run_milestone.py`
- `tests/**`, `pytest.ini`, `pyproject.toml` (optional course/gpu extras; flash-attn/triton Linux-gated)

## Test commands

```bash
# CPU (macOS / no CUDA)
PYTHONPATH=. pytest -m "not gpu"

# Single milestone
python tools/run_milestone.py 1
python tools/run_milestone.py 4

# Progress
python tools/course_status.py

# GPU (NVIDIA + model weights)
export NANOVLLM_TEST_MODEL=~/huggingface/Qwen3-0.6B/
pytest -m gpu
```

## Test results (conversion environment)

### Machine

- OS: macOS (darwin), no CUDA
- Python: conda `cse291pa3` with torch 2.12.0 CPU
- Model weights: **not present** (`~/huggingface/Qwen3-0.6B/` missing)

### `course/reference-solutions`

| Command | Result |
|---|---|
| `python -m compileall` (nanovllm/tools/tests) | **PASS** |
| `pytest -m "not gpu"` | **37 passed, 1 skipped, 4 deselected** |
| `pytest -m gpu` | **UNVERIFIED** (no CUDA) |
| Original `example.py` | **UNVERIFIED** (no GPU/model) |
| Deterministic greedy e2e | **UNVERIFIED** (no GPU/model) |
| Two-request continuous batching | **PASS** via `tests/milestones/test_08_engine_fake.py` (fake runner) |
| KV-cache cleanup | **PASS** via milestone 4/8 CPU tests |

### `course/student`

| Command | Result |
|---|---|
| `python -m compileall` | **PASS** |
| `pytest -m "not gpu"` | **3 passed, 35 failed** (expected: clear `CourseNotImplementedError` / TODO messages) |
| `pytest -m gpu` | **UNVERIFIED** (no CUDA) |
| `tools/course_status.py` | Reports remaining TODOs; next = `TODO-L1-SEQ-01` |

GPU tests are structurally present and skip cleanly when CUDA/model unavailable. Exact later commands:

```bash
pytest -m gpu
pytest -m gpu tests/milestones/test_09_end_to_end_eager.py
pytest -m gpu tests/milestones/test_10_cuda_graphs.py
python example.py   # requires Qwen3-0.6B locally
```

## Known limitations

1. No GPU verification in this conversion environment.
2. `flash-attn` / `triton` are Linux-oriented; CPU course work avoids importing them for milestones 1–8, 11.
3. Early scheduler milestones inject `tests/oracles/working_block_manager.py` (no prefix cache) so milestone 2–3 do not require completing milestone 4 first.
4. `SamplingParams` forbids true greedy `temperature=0`; low-temperature tests approximate greedy.
5. `TODO-L2-ENGINE-01` is a contract/integration milestone (fake runner provided), not a deleted engine function.
6. Student `Attention.forward` and CUDA graph paths need a real GPU stack to fully validate after implementation.

## Recommended first exercise

1. `git checkout course/student`
2. Read `COURSE.md` and `docs/tutorial/00_overview.md`
3. Implement `TODO-L1-SEQ-01` … `TODO-L1-SEQ-03`
4. Run: `python tools/run_milestone.py 1`


## Authoritative tip SHAs

Run `git rev-parse course/reference-solutions course/student` for the latest tips; the table above is accurate as of the conversion commits immediately preceding this note.
