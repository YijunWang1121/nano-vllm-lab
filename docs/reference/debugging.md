# Debugging Reference

## First Commands

Check course progress:

```bash
python tools/course_status.py
```

Run all CPU tests:

```bash
python -m pytest -q -m "not gpu"
```

Run one milestone:

```bash
python tools/run_milestone.py 4
```

Run GPU tests when CUDA and a local model are available:

```bash
export NANOVLLM_TEST_MODEL="$HOME/huggingface/Qwen3-0.6B"
python -m pytest -q -m gpu
```

## Debug Flags

Subsystem logs live in `nanovllm/utils/debug.py`.

```bash
export NANOVLLM_DEBUG_ENGINE=1
export NANOVLLM_DEBUG_SCHEDULER=1
export NANOVLLM_DEBUG_KVCACHE=1
export NANOVLLM_DEBUG_RUNNER=1
export NANOVLLM_DEBUG_ATTENTION=1
export NANOVLLM_DEBUG_SAMPLING=1
```

Prefer enabling one or two at a time. Scheduler plus KV-cache is the best starting pair for CPU milestones.

## Reading TODO Failures

If pytest reports `CourseNotImplementedError`, copy the TODO ID and search `course/todos.yaml`. The registry gives the source file, symbol, docs, and tests.

Exact TODO IDs used by the course:

`TODO-L1-SEQ-01`, `TODO-L1-SEQ-02`, `TODO-L1-SEQ-03`, `TODO-L2-SCHED-01`, `TODO-L2-SCHED-02`, `TODO-L2-SCHED-03`, `TODO-L2-SCHED-04`, `TODO-L2-KVCACHE-01`, `TODO-L3-KVCACHE-02`, `TODO-L3-KVCACHE-03`, `TODO-L2-KVCACHE-04`, `TODO-L2-KVCACHE-05`, `TODO-L3-KVCACHE-06`, `TODO-L2-RUNNER-01`, `TODO-L2-RUNNER-02`, `TODO-L2-RUNNER-03`, `TODO-L2-ATTN-01`, `TODO-L1-SAMPLE-01`, `TODO-L2-ENGINE-01`, `TODO-L3-CUDAGRAPH-01`, `TODO-L3-CUDAGRAPH-02`, `TODO-L2-TP-01`, `TODO-L3-TP-02`.

## Sequence Bugs

Symptoms:

- Wrong completion tokens.
- Wrong block count.
- Decode feeds the wrong last token.

Check:

- `seq.token_ids`.
- `seq.last_token`.
- `seq.num_tokens`.
- `seq.num_prompt_tokens`.
- `seq.num_blocks`.
- `seq.last_block_num_tokens`.

## Scheduler Bugs

Symptoms:

- Tests hang or loop forever.
- Requests remain in the wrong queue.
- Chunked prefill appends a sampled token too early.

Check:

- `waiting` and `running` contents.
- `seq.status`.
- `seq.is_prefill`.
- `seq.num_cached_tokens`.
- `seq.num_scheduled_tokens`.

## KV-Cache Bugs

Symptoms:

- Out-of-memory when blocks should be reusable.
- `block_table` has wrong length.
- Prefix cache reuses the wrong block.

Check:

- `free_block_ids`.
- `used_block_ids`.
- each `Block.ref_count`.
- `hash_to_block_id`.
- stored `Block.token_ids`.

For A `[10,11,12,13,14]` and B `[20,21,22]`, expected logical block counts are 2 and 1.

## Metadata Bugs

Symptoms:

- Slot mappings are off by one.
- Prefix-cache attention uses the wrong path.
- Decode context lengths are wrong.

Check pure Python builders directly:

```python
from nanovllm.engine.input_metadata import build_prefill_metadata, build_decode_metadata
```

For A `block_table=[7,2]`, decode slot should be `8`.

## Attention Bugs

Symptoms:

- GPU-only failures.
- FlashAttention argument errors.
- Incorrect output only with prefix cache.

Check:

- `context.is_prefill`.
- `context.slot_mapping`.
- `context.block_tables`.
- `context.context_lens`.
- whether `reset_context()` ran after the previous batch.

## Sampling Bugs

Symptoms:

- Output shape is `[batch, 1]` or `[batch, vocab]`.
- Low temperature does not prefer the largest logit.
- `temperature=0` silently works.

Check:

- `SamplingParams.__post_init__`.
- `logits.float()`.
- broadcasting of `temperatures.unsqueeze(1)`.
- output dtype is integer token IDs.

## CUDA Graph Bugs

Symptoms:

- Eager path works but graph path fails.
- Outputs depend on previous batch size.
- Decode graph replay uses stale block tables.

Check:

- `enforce_eager` first.
- `graph_bs` bucket chosen.
- unused `slot_mapping` entries are filled.
- `context_lens` and `block_tables` buffers are refreshed.
- output is sliced to actual batch size.

## Tensor Parallel Bugs

Symptoms:

- Local shard shapes are wrong.
- Single-rank works but multi-rank diverges.
- Row-parallel output is too small or too large.

Check:

- `tp_rank`, `tp_size`.
- shard dimension in `weight_loader`.
- `dist.all_reduce(y)` in `RowParallelLinear.forward`.
- bias is applied only on rank 0 for row-parallel layers.

## Comparing With Reference

Do not edit `course/reference-solutions`. Compare one file:

```bash
git diff course/student..course/reference-solutions -- nanovllm/engine/input_metadata.py
```

Use the diff to understand direction, then implement in your own words.
