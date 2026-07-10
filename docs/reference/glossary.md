# Glossary

## Block Table

`Sequence.block_table` maps a sequence's logical block index to a physical KV-cache block ID. Example: A `block_table=[7,2]` means logical block 0 uses physical block 7 and logical block 1 uses physical block 2.

## Chunked Prefill

Prefill that processes only part of a prompt because `max_num_batched_tokens` is smaller than the remaining prompt length. Incomplete chunked prefill must not append the sampled token.

## Decode

The autoregressive phase after prefill. Each scheduled sequence contributes one current `last_token`, and the model samples one next token.

## Gumbel-Max Trick

A sampling method where categorical sampling is implemented by adding equivalent random noise and taking `argmax`. This repository uses exponential noise as `probs / exponential_noise` followed by `argmax`.

## KV Cache

GPU memory that stores attention keys and values from previous tokens. In this repository, tensor storage is allocated in `ModelRunner.allocate_kv_cache()`, while block ownership is tracked by `BlockManager`.

## Logical Block

A block index within one sequence, computed from token positions and `Sequence.block_size`. Logical block IDs are local to a sequence.

## Physical Block

A block ID in the global KV-cache pool managed by `BlockManager`. Physical IDs appear in `Sequence.block_table`.

## Prefill

The phase that processes prompt tokens. Prefill can schedule multiple tokens per sequence and uses `flash_attn_varlen_func`.

## Prefix Caching

Reusing previously computed full KV-cache blocks for a later request with the same token prefix. Implemented through `BlockManager.hash_blocks()`, `hash_to_block_id`, and `can_allocate()`.

## Ref Count

`Block.ref_count` records how many sequences reference a physical block. Prefix-cache reuse can make the count greater than one.

## Running Example

The tutorials use A `[10,11,12,13,14]` and B `[20,21,22]` with block size 4. Example physical mapping: A `[7,2]`, B `[5]`.

## Sequence

The request state object in `nanovllm/engine/sequence.py`. It stores prompt and completion tokens, status, cache counters, scheduled counters, and block table.

## Slot Mapping

A list or tensor mapping each scheduled token to a flattened physical KV-cache slot. Formula: `physical_block_id * block_size + offset`.

## Tensor Parallelism

Splitting model weights across ranks. `ColumnParallelLinear` shards output rows, `RowParallelLinear` shards input columns and all-reduces outputs, and `QKVParallelLinear` shards packed Q/K/V projection weights.

## WAITING/RUNNING/FINISHED

The three `SequenceStatus` values. `WAITING` sequences need prefill or recomputation, `RUNNING` sequences can decode, and `FINISHED` sequences are complete and should have freed blocks.

## xxhash

The hash function used by `BlockManager.compute_hash()` to identify reusable full prefix blocks. The implementation also compares stored `token_ids` to protect against hash collisions.
