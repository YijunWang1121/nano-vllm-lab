# State Invariants Reference

Use this page when debugging tests. Most failures are violations of one of these ownership rules.

## Config

- `nanovllm/config.py` sets `Config.kvcache_block_size=256` by default.
- `Config.__post_init__` asserts `kvcache_block_size % 256 == 0`.
- Course tests use helpers that set `Sequence.block_size=4` for small examples; do not confuse that with production config.

## Sequence

Path: `nanovllm/engine/sequence.py`.

- `status` is one of `WAITING`, `RUNNING`, `FINISHED`.
- `num_prompt_tokens` never changes after construction.
- `num_completion_tokens == num_tokens - num_prompt_tokens`.
- `last_token == token_ids[-1]` whenever `token_ids` is available.
- `append_token()` updates `token_ids`, `last_token`, and `num_tokens`.
- `block_table` stores physical block IDs, not logical block indexes.
- `num_blocks == ceil(num_tokens / Sequence.block_size)`.

## Scheduler

Path: `nanovllm/engine/scheduler.py`.

- `waiting` contains sequences that still need prefill or recomputation.
- `running` contains sequences eligible for decode.
- `schedule()` prefers prefill if any waiting sequence can be admitted.
- Prefill may schedule multiple tokens per sequence.
- Decode schedules one token per sequence.
- Incomplete chunked prefill must not append the sampled token.
- `preempt()` frees KV blocks, resets the sequence to prefill mode, and places it back in `waiting`.
- Finished sequences are removed from `running` and deallocated.

## BlockManager

Path: `nanovllm/engine/block_manager.py`.

- `free_block_ids` and `used_block_ids` must agree with block `ref_count`.
- A block in `used_block_ids` has `ref_count > 0`.
- A block in `free_block_ids` has `ref_count == 0`.
- Reused prefix-cache blocks increment `ref_count` when already in use.
- Only full blocks are hashed for prefix caching.
- `hash_to_block_id` maps chained xxhash values to physical blocks.
- Token IDs must be checked after a hash lookup to guard against collisions.
- `deallocate(seq)` clears `seq.block_table` and resets `seq.num_cached_tokens`.

## Metadata Builders

Path: `nanovllm/engine/input_metadata.py`.

- Builders should not mutate `Sequence`.
- `build_block_tables()` pads with `-1`.
- `slot_mapping` length equals the number of scheduled tokens.
- Prefix-cache prefill has `cu_seqlens_k[-1] > cu_seqlens_q[-1]`.
- Decode slot mapping points to the current last token's cache slot.

## ModelRunner

Path: `nanovllm/engine/model_runner.py`.

- `ModelRunner.run()` resets context before returning.
- Prefill uses `prepare_prefill()` and decode uses `prepare_decode()`.
- Rank 0 samples token IDs; worker ranks return `None`.
- `run_model()` uses eager execution for prefill, `enforce_eager=True`, or decode batch size above 512.
- CUDA graph replay copies current metadata into static buffers and slices outputs back to the real batch size.

## Attention

Path: `nanovllm/layers/attention.py`.

- K/V cache is updated through `store_kvcache()` when cache tensors are allocated.
- Prefill calls `flash_attn_varlen_func`.
- Decode calls `flash_attn_with_kvcache`.
- Prefix-cache prefill passes block tables because some K/V entries already live in cache.

## Sampling

Path: `nanovllm/layers/sampler.py`.

- `SamplingParams.temperature` must be strictly positive.
- Sampler output shape is `[batch]`.
- The implementation uses temperature plus Gumbel-max via exponential noise and `argmax`.
- Greedy `temperature=0` is intentionally forbidden.

## Tensor Parallelism

Path: `nanovllm/layers/linear.py`.

- `divide()` asserts exact divisibility.
- `ColumnParallelLinear` shards output rows.
- `QKVParallelLinear` shards packed Q, K, and V projections.
- `RowParallelLinear` shards input columns and all-reduces partial outputs.
- Row-parallel bias is applied on rank 0 only before all-reduce.
