# 07. Attention Metadata

## Learning Objectives

- Connect model-runner context fields to attention kernel arguments.
- Explain why prefill and decode use different FlashAttention calls.
- Understand when prefix-cache prefill needs a block table.

## Why Exists

Attention is where logical request metadata meets GPU kernels. The code must select the correct execution path and provide enough cache metadata for each token.

## Call Path

`ModelRunner.prepare_prefill()` or `prepare_decode()` calls `set_context()`. During model forward, `Attention.forward()` reads `get_context()`, stores K/V into cache with `store_kvcache()`, then dispatches to a prefill or decode attention function.

## Files/functions

- `nanovllm/layers/attention.py`: `store_kvcache_kernel`, `store_kvcache`, `Attention.forward`.
- `nanovllm/utils/context.py`: `Context`, `set_context`, `get_context`, `reset_context`.
- `nanovllm/engine/model_runner.py`: `prepare_prefill`, `prepare_decode`, `run`.
- `tests/milestones/test_06_attention_metadata.py`.

## Data Structures

The context object can carry `is_prefill`, cumulative sequence lengths, max sequence lengths, `slot_mapping`, `context_lens`, and `block_tables`. Not every field is used in every path.

## State Transitions

Context is set before model execution and reset after sampling. It should not leak from one batch to the next.

## Tensor Shapes

`store_kvcache()` expects `key` and `value` shaped like `[N, num_kv_heads, head_dim]`, where `N` is scheduled tokens. `slot_mapping` has length `N`. Cache tensors are indexed as flattened slots inside each layer's `[num_blocks, block_size, num_kv_heads, head_dim]` storage.

## Pseudocode

```text
Attention.forward(q, k, v):
    context = get_context()
    if cache exists:
        store k and v at context.slot_mapping
    if context.is_prefill:
        maybe replace k/v with cache tensors for prefix-cache attention
        call flash_attn_varlen_func(...)
    else:
        call flash_attn_with_kvcache(...)
```

## TODO IDs

- `TODO-L2-ATTN-01`: select attention execution path from context.

## Hints

Use `context.block_tables is not None` as the signal that prefill should attend through the cache, which happens for prefix-cache prefill. Decode always needs block tables.

## Common Bugs

- Forgetting to store K/V before calling attention.
- Passing block tables for normal prefill when not needed.
- Using decode `context_lens` in prefill.
- Failing to reset context after `ModelRunner.run()`.

## Tests

Run:

```bash
python tools/run_milestone.py 6
```

The CPU tests check the metadata contract. Full FlashAttention execution is exercised by GPU milestones when dependencies are available.

## Debugging

Enable:

```bash
export NANOVLLM_DEBUG_ATTENTION=1
export NANOVLLM_DEBUG_RUNNER=1
```

Check whether the path logs `prefill` or `decode`, and whether `use_block_table` matches prefix-cache expectations.

## Expected Intermediate Behavior

Normal full prefill for A uses `flash_attn_varlen_func` without a block table. Prefix-cache prefill for A's final token uses block tables because keys before the scheduled token are already in the KV cache.

## Reflection

Attention does not know about waiting or running queues. It only knows the batch metadata prepared for the current kernel call.

## Connection to Production vLLM

Production vLLM has multiple attention backends and metadata classes. The same conceptual fields remain: query lengths, cache lengths, cache block tables, and physical slot mappings.
