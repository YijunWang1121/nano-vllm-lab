# 01. Repository Tour

## Learning Objectives

- Locate the implementation files, tests, course registry, and reference docs.
- Understand which files are CPU-only and which require CUDA.
- Know where public API code ends and engine internals begin.

## Why Exists

The repository is compact, so getting oriented pays off immediately. Most course confusion comes from opening a GPU file before the CPU invariants are clear.

## Call Path

The public call path starts in `nanovllm/__init__.py`, passes through `nanovllm/llm.py`, and lands in `nanovllm/engine/llm_engine.py`. From there the engine coordinates `Scheduler`, `ModelRunner`, `Sequence`, and `SamplingParams`.

## Files/functions

- `nanovllm/config.py`: `Config`, including `kvcache_block_size`.
- `nanovllm/engine/sequence.py`: `Sequence`, `SequenceStatus`.
- `nanovllm/engine/scheduler.py`: `Scheduler.schedule`, `preempt`, `postprocess`.
- `nanovllm/engine/block_manager.py`: `Block`, `BlockManager`.
- `nanovllm/engine/input_metadata.py`: CPU metadata builders.
- `nanovllm/engine/model_runner.py`: GPU execution, CUDA graphs, TP IPC.
- `nanovllm/layers/attention.py`: attention path selection.
- `nanovllm/layers/sampler.py`: temperature sampling.
- `nanovllm/layers/linear.py`: tensor parallel linear layers.

## Data Structures

Course data starts as Python lists in tests. It becomes tensors inside `ModelRunner.prepare_prefill()`, `prepare_decode()`, and `prepare_sample()`. `BlockManager` never stores K/V tensors; it stores IDs and ownership metadata.

## State Transitions

The repository separates state transitions by owner:

- `Sequence` owns token-level fields.
- `Scheduler` owns queue membership and request status.
- `BlockManager` owns physical block ownership and prefix-cache hashes.
- `ModelRunner` owns GPU context and tensor execution.

## Tensor Shapes

Look in `docs/reference/tensor_shapes.md` for the full reference, and `docs/reference/ownership.md` for who manages vs who reads each class field. For orientation, remember: `block_table` is per sequence, `slot_mapping` is per scheduled token, and KV cache is allocated as `[2, layers, blocks, block_size, kv_heads_per_rank, head_dim]`.

## Pseudocode

```text
read course/milestones.yaml
for milestone in order:
    read docs/tutorial/<chapter>.md
    read listed source files
    run listed tests
    implement only the TODOs for that milestone
```

## TODO IDs

This chapter has no implementation TODO. It points you to the exact registry containing `TODO-L1-SEQ-01` through `TODO-L3-TP-02`.

## Hints

Start with tests in `tests/milestones`. They use smaller block sizes and fake runners so you can reason without installing FlashAttention.

## Common Bugs

- Editing `tests/oracles` as if it were production code.
- Searching only `nanovllm/engine` and missing `nanovllm/layers`.
- Forgetting that `tools/course_status.py` determines progress by TODO marker state.

## Tests

Smoke-test the repository layout with:

```bash
python tools/course_status.py
python tools/run_milestone.py 1 --no-test
```

## Debugging

If an import fails, check whether you are running from repository root and whether the editable install is active. If a GPU dependency fails, use `-m "not gpu"` while working through CPU milestones.

## Expected Intermediate Behavior

You should be able to name the file that owns each major behavior: sequence state, scheduler policy, block allocation, metadata construction, attention execution, sampling, CUDA graphs, and tensor parallelism.

## Reflection

A small codebase can hide big ideas. Keep a local map of ownership boundaries; it will help you avoid adding fixes in the wrong layer.

## Connection to Production vLLM

Production vLLM has many more packages and configuration layers, but the same mental map still applies: request state, scheduler, KV cache manager, worker/model runner, attention backend, and sampling.
