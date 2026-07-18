# nano-vLLM Course

This course turns `nano-vLLM` into a guided implementation project. You will build the core execution path of a small vLLM-style inference engine without copying complete source solutions.

## What You Build

By the end, you should understand and implement the main components behind offline LLM generation:

- Public API: `nanovllm.LLM` and `nanovllm.SamplingParams`.
- Request state: `nanovllm/engine/sequence.py`.
- Continuous batching: `nanovllm/engine/scheduler.py`.
- Paged KV-cache allocation and prefix caching: `nanovllm/engine/block_manager.py`.
- CPU metadata builders: `nanovllm/engine/input_metadata.py`.
- Attention dispatch: `nanovllm/layers/attention.py`.
- Sampling: `nanovllm/layers/sampler.py`.
- CUDA graph replay and tensor parallelism: `nanovllm/engine/model_runner.py` and `nanovllm/layers/linear.py`.

The running example used throughout the docs has educational `Sequence.block_size=4`:

- Request A prompt: `[10, 11, 12, 13, 14]`, length 5, logical blocks `[0, 1]`.
- Request B prompt: `[20, 21, 22]`, length 3, logical blocks `[0]`.
- Example physical mapping: A `block_table=[7, 2]`, B `block_table=[5]`.
- A decode slot: `physical_block=2`, `offset=0`, so `slot=8`.

Production config defaults are different: `Config.kvcache_block_size` is `256`, and `Config.__post_init__` asserts that it is a multiple of 256.

## Compared With Production vLLM

`nano-vLLM` keeps the ideas and removes much of the infrastructure.

- It implements continuous batching, paged KV cache, prefix caching, FlashAttention prefill/decode paths, CUDA graphs, and tensor parallel linear layers.
- It does not implement vLLM's full serving stack, OpenAI-compatible server, distributed scheduler, speculative decoding, quantization matrix, multi-model support, LoRA stack, or production observability.
- The code is intentionally small enough to read end to end. That makes invariants visible, but it also means fewer guardrails than production vLLM.

Use `docs/tutorial/14_compare_with_vllm.md` for a grounded comparison.

## Prerequisites

You should be comfortable with:

- Python classes, dataclasses, and pytest.
- Basic PyTorch tensors and CUDA concepts.
- Transformer inference: prompt prefill, autoregressive decode, logits, sampling.
- Attention shapes: query tokens, key/value cache, heads, head dimension.
- Basic Git branching and `git diff`.

FlashAttention, Triton, NCCL, and CUDA graphs are introduced when needed. The early milestones are CPU-only.

## Expected Effort

Plan for about 40-60 focused hours:

- CPU milestones: about 20-30 hours.
- GPU eager path: about 4-8 hours, depending on local setup.
- CUDA graphs and tensor parallelism: about 10-18 hours.
- Review, debugging, and comparison with reference: about 6-10 hours.

## Branches

Work on:

```bash
git switch course/student
```

Do not work on `course/reference-solutions`. Treat that branch as read-only reference material. If you accidentally switch to it, switch back before editing:

```bash
git switch course/student
```

Compare one file against the reference branch with:

```bash
git diff course/student..course/reference-solutions -- nanovllm/engine/scheduler.py
```

Compare a docs or test file the same way:

```bash
git diff course/student..course/reference-solutions -- tests/milestones/test_04_kv_cache.py
```

## Setup

Create an environment using the repository's normal dependency flow, then install test dependencies as needed.

```bash
python -m pip install -e .
python -m pip install pytest pyyaml
```

For GPU milestones, install CUDA-compatible PyTorch, FlashAttention, and Triton in the way that matches your machine. The GPU tests skip when CUDA or a local model is unavailable.

Set the model path for GPU tests:

```bash
export NANOVLLM_TEST_MODEL="$HOME/huggingface/Qwen3-0.6B"
```

The README shows one way to download `Qwen/Qwen3-0.6B`.

## Ownership map

When unsure which component may mutate a field (especially `Sequence.block_table`, `num_cached_tokens`, or Block hash state), read [`docs/reference/ownership.md`](docs/reference/ownership.md). It lists managers vs readers for `Sequence`, `Block` / `BlockManager`, `Scheduler`, `Config`, and `Context`.

## Implementation Order

Follow the milestone order in `course/milestones.yaml`. Difficulty and time estimates live in that file (`difficulty`, `estimated_hours`) and are shown by `python tools/course_status.py` / `python tools/run_milestone.py N`.

| M | Topic | Difficulty | Est. hours |
|---|---|---|---|
| 1 | Sequence state and block indexing | easy | 1.5 |
| 2 | Scheduler prefill admission | medium | 2.5 |
| 3 | Decode, preemption, postprocess | medium | 3 |
| 4 | KV-cache block manager | hard | 6 |
| 5 | Model input metadata | medium | 3 |
| 6 | Attention metadata integration | medium | 2.5 |
| 7 | Sampling | easy | 1 |
| 8 | Fake end-to-end engine | medium | 2 |
| 9 | Eager GPU generation | medium | 2 |
| 10 | CUDA graphs | hard | 5 |
| 11 | Tensor parallelism | hard | 4 |
| 12–14 | Prefix-cache review, profiling, compare with vLLM | — | reading / synthesis |

Estimates are focused student hours (read + implement + tests). GPU milestones may take longer if the environment is not set up yet.

The TODO IDs are registered in `course/todos.yaml`. Keep the exact IDs in mind when reading test failures:

`TODO-L1-SEQ-01`, `TODO-L1-SEQ-02`, `TODO-L1-SEQ-03`, `TODO-L2-SCHED-01`, `TODO-L2-SCHED-02`, `TODO-L2-SCHED-03`, `TODO-L2-SCHED-04`, `TODO-L2-KVCACHE-01`, `TODO-L3-KVCACHE-02`, `TODO-L3-KVCACHE-03`, `TODO-L2-KVCACHE-04`, `TODO-L2-KVCACHE-05`, `TODO-L3-KVCACHE-06`, `TODO-L2-RUNNER-01`, `TODO-L2-RUNNER-02`, `TODO-L2-RUNNER-03`, `TODO-L2-ATTN-01`, `TODO-L1-SAMPLE-01`, `TODO-L2-ENGINE-01`, `TODO-L3-CUDAGRAPH-01`, `TODO-L3-CUDAGRAPH-02`, `TODO-L2-TP-01`, `TODO-L3-TP-02`.

## CPU vs GPU Milestones

Start with CPU tests:

```bash
python -m pytest -q -m "not gpu"
```

Run one milestone:

```bash
python tools/run_milestone.py 4
```

GPU milestones are marked with `pytest.mark.gpu`:

```bash
python -m pytest -q -m gpu
```

If GPU tests skip, continue through the CPU material first. The course is designed so `Sequence`, `Scheduler`, `BlockManager`, metadata, sampling, and many tensor-parallel checks can be understood without a GPU.

## Progress Tools

Show your next recommended TODO:

```bash
python tools/course_status.py
```

Run milestone tests while showing status:

```bash
python tools/course_status.py --run-tests
```

Run a specific milestone without executing tests:

```bash
python tools/run_milestone.py 5 --no-test
```

## Debug Flags

`nanovllm/utils/debug.py` provides opt-in logging:

```bash
export NANOVLLM_DEBUG_ENGINE=1
export NANOVLLM_DEBUG_SCHEDULER=1
export NANOVLLM_DEBUG_KVCACHE=1
export NANOVLLM_DEBUG_RUNNER=1
export NANOVLLM_DEBUG_ATTENTION=1
export NANOVLLM_DEBUG_SAMPLING=1
```

Enable one subsystem at a time when possible. Scheduler and KV-cache logs are especially useful while debugging `block_table`, `num_cached_tokens`, and preemption.

## Reset a Milestone

If a milestone gets tangled, inspect the reference diff before deleting work:

```bash
git diff -- nanovllm/engine/block_manager.py
git diff course/student..course/reference-solutions -- nanovllm/engine/block_manager.py
```

To reset your own edits for one file, ask your instructor or make a backup commit first. Avoid broad resets; they can erase unrelated work.

## Testing Rhythm

For each milestone:

1. Read the tutorial chapter.
2. Open the source file and TODO IDs.
3. Run the milestone test and observe the first failure.
4. Implement the smallest coherent piece.
5. Re-run the milestone test.
6. Run `python tools/course_status.py`.
7. Move on only when the invariants in `docs/reference/state_invariants.md` still hold.

Do not paste complete solutions from the reference branch. The goal is to build an executable mental model of the engine.
