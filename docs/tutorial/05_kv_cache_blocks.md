# 05. KV-Cache Blocks

## Learning Objectives

- Implement physical block allocation, deallocation, append checks, and prefix-cache reuse.
- Explain logical blocks versus physical blocks.
- Preserve `ref_count`, free-list, and hash-table invariants.

## Why Exists

Paged KV cache lets each sequence grow without requiring one contiguous cache tensor slice. `BlockManager` maps logical sequence blocks to physical KV-cache blocks.

## Call Path

`Scheduler.schedule()` asks `BlockManager.can_allocate()` and `allocate()` during prefill. During decode it asks `can_append()` and `may_append()`. `Scheduler.postprocess()` calls `hash_blocks()` and may call `deallocate()`.

## Files/functions

- `nanovllm/engine/block_manager.py`: `Block`, `BlockManager.compute_hash`, `_allocate_block`, `_deallocate_block`, `can_allocate`, `allocate`, `deallocate`, `can_append`, `may_append`, `hash_blocks`.
- `nanovllm/engine/sequence.py`: `Sequence.block`, `num_blocks`, `last_block_num_tokens`.
- `tests/milestones/test_04_kv_cache.py`.

## Data Structures

`BlockManager` owns:

- `blocks`: all `Block` objects.
- `hash_to_block_id`: prefix-cache lookup table.
- `free_block_ids`: available physical IDs.
- `used_block_ids`: physical IDs currently referenced.

Each `Block` stores `block_id`, `ref_count`, `hash`, and `token_ids`.

## State Transitions

A physical block starts free, becomes allocated with `ref_count=1`, may become shared when a prefix-cache hit increments `ref_count`, and returns to the free list when all referencing sequences deallocate it.

## Tensor Shapes

The manager tracks IDs, not tensors. With block size 4, A has 2 logical blocks and B has 1. An example mapping is A `block_table=[7,2]`, B `block_table=[5]`.

## Pseudocode

```text
can_allocate(seq):
    walk full prefix blocks
    compute chained xxhash values
    count reusable cached blocks
    ensure enough free blocks for remaining blocks
allocate(seq, cached):
    append reused block IDs first
    allocate new physical blocks for the rest
    set num_cached_tokens = cached * block_size
deallocate(seq):
    decrement ref counts
    free blocks whose ref count reaches zero
```

## TODO IDs

- `TODO-L2-KVCACHE-01`: physical block allocate and free.
- `TODO-L3-KVCACHE-02`: capacity check with prefix-cache reuse.
- `TODO-L3-KVCACHE-03`: allocate block table for a sequence.
- `TODO-L2-KVCACHE-04`: deallocate sequence blocks.
- `TODO-L2-KVCACHE-05`: append block on decode boundary.
- `TODO-L3-KVCACHE-06`: hash completed blocks for prefix caching.

## Hints

Only full blocks are prefix-cache candidates. `can_allocate()` loops through `seq.num_blocks - 1`, so the final partial block is not reused as a prefix hit.

## Common Bugs

- Returning cached blocks without checking `token_ids` for hash collision safety.
- Forgetting to remove a stale hash mapping when reusing a freed block.
- Allocating a new block for a cached prefix that is already in use.
- Clearing `block_table` before decrementing ref counts.

## Tests

Run:

```bash
python tools/run_milestone.py 4
```

Tests cover allocation, out-of-memory, decode boundary append, prefix-cache reuse, and A/B block table shape.

## Debugging

Enable `NANOVLLM_DEBUG_KVCACHE=1`. Watch `free_block_ids`, `used_block_ids`, `ref_count`, and each sequence's `block_table`.

## Expected Intermediate Behavior

For A with `[7,2]`, the decode slot for the current last token is `2 * 4 + 1 - 1 = 8`. B with `[5]` maps its current last token to `5 * 4 + 3 - 1 = 22`.

## Reflection

The block manager is a bookkeeping layer. If you are tempted to store actual key/value tensors here, stop: tensor storage belongs to `ModelRunner.allocate_kv_cache()`.

## Connection to Production vLLM

This is the miniature version of vLLM's paged attention memory manager. Production vLLM has more allocation policies and scheduling integration, but logical-to-physical block mapping is the same core idea.
