# 12. Prefix Caching

## Learning Objectives

- Explain how completed prompt blocks can be reused across requests.
- Connect `hash_blocks()` with `can_allocate()` reuse.
- Understand why prefix-cache prefill needs attention block tables.

## Why Exists

Many workloads repeat system prompts or long shared prefixes. Prefix caching avoids recomputing full KV blocks that are already present in the cache.

## Call Path

`Scheduler.postprocess()` calls `BlockManager.hash_blocks()` after scheduled tokens complete. Later, `Scheduler.schedule()` calls `BlockManager.can_allocate()`, which walks full prefix blocks and returns how many can be reused. `allocate()` then reuses those blocks and sets `seq.num_cached_tokens`.

## Files/functions

- `nanovllm/engine/block_manager.py`: `compute_hash`, `can_allocate`, `allocate`, `hash_blocks`.
- `nanovllm/engine/input_metadata.py`: `build_prefill_metadata`.
- `nanovllm/layers/attention.py`: prefix-cache prefill path.
- `tests/milestones/test_04_kv_cache.py`.
- `tests/milestones/test_05_model_inputs.py`.

## Data Structures

Prefix hashes are chained: each block hash includes the previous prefix hash and current block tokens. This distinguishes `[A][B]` from a block `B` that appears after a different prefix.

## State Transitions

A block becomes reusable only after `hash_blocks()` records its full token block and hash. A reused block increments `ref_count` if it is already used, or moves from free to used if it was cached but not currently referenced.

## Tensor Shapes

For A with first block `[10,11,12,13]`, a later request `[10,11,12,13,99]` can reuse one full block. Its prefill metadata schedules only token `[99]`, but `cu_seqlens_k` includes the cached prefix length.

## Pseudocode

```text
hash_blocks(seq):
    start = num_cached_tokens // block_size
    end = (num_cached_tokens + num_scheduled_tokens) // block_size
    for each newly completed full block:
        hash = compute_hash(tokens, previous_hash)
        block.update(hash, tokens)
        hash_to_block_id[hash] = block_id
```

## TODO IDs

- `TODO-L3-KVCACHE-06`: hash completed blocks for prefix caching.

Related earlier TODOs: `TODO-L3-KVCACHE-02` and `TODO-L3-KVCACHE-03`.

## Hints

`hash_blocks()` should not hash partial blocks. Use integer division boundaries based on cached and scheduled token counts.

## Common Bugs

- Hashing the final partial block.
- Not chaining the previous block hash.
- Ignoring token comparison after hash lookup.
- Forgetting to set `need_block_tables=True` when `cu_seqlens_k[-1] > cu_seqlens_q[-1]`.

## Tests

Run:

```bash
python tools/run_milestone.py 4
python -m pytest -q tests/milestones/test_05_model_inputs.py::test_prefill_with_prefix_cache_sets_need_block_tables
```

## Debugging

Inspect `hash_to_block_id`, each `Block.token_ids`, and `seq.num_cached_tokens`. Prefix-cache bugs usually appear as either no reuse or unsafe over-reuse.

## Expected Intermediate Behavior

After A's first full block is hashed, a request starting with `[10,11,12,13]` should report `num_cached_blocks=1`, reuse A's first physical block, and schedule only remaining prompt tokens.

## Reflection

Prefix caching is correct only when reuse is exact. A fast wrong cache hit is worse than no cache hit.

## Connection to Production vLLM

Production vLLM's prefix caching has more policy and eviction complexity, but it still depends on stable block hashes, token equality checks, block reference counts, and attention metadata that can attend to cached prefixes.
