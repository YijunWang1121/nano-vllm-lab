# Tensor Shapes Reference

This page summarizes the shapes used by the course implementation. CPU milestone tests often use Python lists; `ModelRunner` converts them to tensors.

## Running Example

Educational tests use `Sequence.block_size=4`.

- Request A: `[10, 11, 12, 13, 14]`, length 5, logical blocks `[0, 1]`.
- Request B: `[20, 21, 22]`, length 3, logical blocks `[0]`.
- Example physical blocks: A `block_table=[7, 2]`, B `block_table=[5]`.

## Sequence-Level Values

| Value | A | B |
|---|---:|---:|
| `num_tokens` | 5 | 3 |
| `num_blocks` | 2 | 1 |
| `last_block_num_tokens` | 1 | 3 |
| current decode slot | `2 * 4 + 0 = 8` | `5 * 4 + 2 = 22` |

## `build_block_tables(seqs)`

Path: `nanovllm/engine/input_metadata.py`.

Input:

- A `block_table=[7,2]`
- B `block_table=[5]`

Output:

```python
[[7, 2], [5, -1]]
```

Shape after tensor conversion: `[num_seqs, max_num_blocks]`.

## Prefill Metadata

Full prefill for A and B:

- `input_ids`: `[10,11,12,13,14,20,21,22]`, shape `[8]`.
- `positions`: `[0,1,2,3,4,0,1,2]`, shape `[8]`.
- `cu_seqlens_q`: `[0,5,8]`, shape `[num_seqs + 1]`.
- `cu_seqlens_k`: `[0,5,8]`, shape `[num_seqs + 1]`.
- `max_seqlen_q`: `5`.
- `max_seqlen_k`: `5`.
- `slot_mapping`: `[28,29,30,31,8,20,21,22]`.
- `need_block_tables`: `False`.

Prefix-cache prefill for A with first block cached:

- `num_cached_tokens=4`.
- `num_scheduled_tokens=1`.
- `input_ids=[14]`.
- `positions=[4]`.
- `cu_seqlens_q=[0,1]`.
- `cu_seqlens_k=[0,5]`.
- `slot_mapping=[8]`.
- `need_block_tables=True`.

## Decode Metadata

Decode for A and B:

- `input_ids=[14,22]`, shape `[2]`.
- `positions=[4,2]`, shape `[2]`.
- `context_lens=[5,3]`, shape `[2]`.
- `slot_mapping=[8,22]`, shape `[2]`.
- `block_tables=[[7,2],[5,-1]]`, shape `[2,2]`.

## ModelRunner Tensors

`prepare_prefill()` creates CUDA tensors:

- `input_ids`: `torch.int64`, `[num_scheduled_tokens_total]`.
- `positions`: `torch.int64`, `[num_scheduled_tokens_total]`.
- `cu_seqlens_q`: `torch.int32`, `[num_seqs + 1]`.
- `cu_seqlens_k`: `torch.int32`, `[num_seqs + 1]`.
- `slot_mapping`: `torch.int32`, `[num_scheduled_tokens_total]`.
- `block_tables`: `torch.int32`, `[num_seqs, max_num_blocks]` or `None`.

`prepare_decode()` creates:

- `input_ids`: `torch.int64`, `[num_seqs]`.
- `positions`: `torch.int64`, `[num_seqs]`.
- `slot_mapping`: `torch.int32`, `[num_seqs]`.
- `context_lens`: `torch.int32`, `[num_seqs]`.
- `block_tables`: `torch.int32`, `[num_seqs, max_num_blocks]`.

## KV Cache Tensor

Allocated in `ModelRunner.allocate_kv_cache()`:

```text
[2, num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads_per_rank, head_dim]
```

The first dimension is key/value. Each attention layer receives a layer slice as `k_cache` and `v_cache`.

## Attention Tensors

`store_kvcache()` expects:

- `key`: `[N, num_kv_heads, head_dim]`.
- `value`: `[N, num_kv_heads, head_dim]`.
- `slot_mapping`: `[N]`.

Prefill uses `flash_attn_varlen_func`. Decode uses `flash_attn_with_kvcache` with `q.unsqueeze(1)`.

## Sampler Tensors

`Sampler.forward(logits, temperatures)`:

- `logits`: `[batch, vocab_size]`.
- `temperatures`: `[batch]`.
- output token IDs: `[batch]`.

## CUDA Graph Buffers

`ModelRunner.capture_cudagraph()` allocates static buffers up to `max_bs`:

- `input_ids`: `[max_bs]`.
- `positions`: `[max_bs]`.
- `slot_mapping`: `[max_bs]`.
- `context_lens`: `[max_bs]`.
- `block_tables`: `[max_bs, max_num_blocks]`.
- `outputs`: `[max_bs, hidden_size]`.
