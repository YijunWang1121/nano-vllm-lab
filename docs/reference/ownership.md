# Ownership: Who Manages / Who Accesses

Use this page when deciding which component is allowed to mutate a field.
**Manager** = who writes or owns the invariant. **Readers** = who consume it.

Rule of thumb:

- `BlockManager` owns physical pages and `Sequence.block_table`.
- `Scheduler` owns queues, `status`, and scheduling counters.
- Metadata builders / Attention **read** sequence + context; they should not mutate `Sequence`.

---

## Sequence (`nanovllm/engine/sequence.py`)

### Class attributes

| Member | Manager (writes) | Readers | Meaning |
|---|---|---|---|
| `block_size` | `LLMEngine.__init__` ← `Config.kvcache_block_size` (tests may override to `4`) | `num_blocks` / `block()` / `last_block_num_tokens`; often compared with `BlockManager.block_size` | Tokens per KV page |
| `counter` | Advanced only in `Sequence.__init__` via `next(...)` | Indirectly yields `seq_id` | Global ID generator |

### Instance fields

| Member | Manager (writes) | Readers | Meaning |
|---|---|---|---|
| `seq_id` | `Sequence.__init__` | Engine outputs, debug logs | Unique request id |
| `status` | `__init__` → `WAITING`; Scheduler prefill → `RUNNING`; `preempt` → `WAITING`; `postprocess` → `FINISHED` | `is_finished`; engine filters finished outputs | Lifecycle |
| `token_ids` | `__init__`; `append_token`; pickle `__setstate__` | `block()`; prompt/completion slices; prefill metadata | Full token list |
| `last_token` | `__init__`; `append_token`; `__setstate__` | Decode metadata; pickle | Latest token |
| `num_tokens` | `__init__`; `append_token` | `__len__`; scheduler; block helpers | Current length |
| `num_prompt_tokens` | `__init__` only (immutable after) | `num_completion_tokens`; slicing | Fixed prompt length |
| `num_cached_tokens` | **`BlockManager.allocate`** (initial); **`Scheduler.postprocess`** (`+= scheduled`); **`BlockManager.deallocate`** → `0` | Scheduler (chunked prefill); `hash_blocks`; prefill metadata | Tokens already in KV / reusable prefix |
| `num_scheduled_tokens` | Scheduler `_schedule_prefill` / `_schedule_decode`; cleared in `postprocess` | Engine throughput; `hash_blocks`; prefill metadata | Tokens for this step |
| `is_prefill` | `__init__` / decode schedule / `preempt` | Pickle path (`__getstate__`) | Prefill vs decode serialization mode |
| `block_table` | **Only `BlockManager`**: `allocate`, `may_append`, `deallocate` | Scheduler (`if not block_table`); `hash_blocks`; metadata builders | Logical → physical block ids |
| `temperature` | Copied from `SamplingParams` at construct | `ModelRunner` → sampler | Sampling temperature |
| `max_tokens` | From `SamplingParams` | `Scheduler.postprocess` finish check | Max new tokens |
| `ignore_eos` | From `SamplingParams` | `Scheduler.postprocess` | Whether EOS stops |

Derived (read-only): `is_finished`, `num_completion_tokens`, `prompt_token_ids`, `completion_token_ids`, `num_blocks`, `last_block_num_tokens`, `block(i)`.

---

## Block (`nanovllm/engine/block_manager.py`)

| Member | Manager (writes) | Readers | Meaning |
|---|---|---|---|
| `block_id` | `Block.__init__` | Identity / tables | Physical index |
| `ref_count` | `reset()`; `allocate` share (`+=1`); `deallocate` (`-=1`) | `_allocate_block` / `_deallocate_block` asserts | How many seqs hold this page |
| `hash` | `hash_blocks` → `update`; `reset` → `-1`; stale cleared on re-allocate | `can_allocate` / chain in `hash_blocks` | Prefix-cache fingerprint (`-1` = none) |
| `token_ids` | `update` / `reset` | `can_allocate` collision check | Tokens stored for verification |

---

## BlockManager

| Member | Manager (writes) | Readers | Meaning |
|---|---|---|---|
| `block_size` | `__init__` ← Config / Scheduler | `allocate`, `can_append`, `may_append`, `hash_blocks` | Page size |
| `blocks` | Pool created at init; fields mutated by BM methods | All BM ops | Physical `Block` objects |
| `free_block_ids` | `_allocate_block` / `_deallocate_block` / reclaim on prefix reuse | `can_allocate`, `can_append` | Free physical ids (`deque`) |
| `used_block_ids` | Same allocate/free paths | `can_allocate` (shared vs free hit) | In-use physical ids (`set`) |
| `hash_to_block_id` | `hash_blocks` insert; `_allocate_block` delete stale | `can_allocate`, `allocate` | Prefix hash → physical id |

**API side effects on `Sequence`:**

| Method | Writes on `Sequence` | Reads on `Sequence` |
|---|---|---|
| `can_allocate` | — | `num_blocks`, `block(i)` |
| `allocate` | `block_table`, `num_cached_tokens` | tokens / blocks for reuse |
| `deallocate` | clear `block_table`, `num_cached_tokens=0` | `block_table` |
| `can_append` | — | `len(seq)` |
| `may_append` | maybe `block_table.append` | `len(seq)` |
| `hash_blocks` | — (updates `Block` + hash map) | `block_table`, cached/scheduled, `block(i)` |

---

## Scheduler (`nanovllm/engine/scheduler.py`)

| Member | Manager (writes) | Readers | Meaning |
|---|---|---|---|
| `max_num_seqs` | `__init__` ← Config | Prefill / decode loops | Max seqs per batch |
| `max_num_batched_tokens` | `__init__` ← Config | Prefill token budget | Max tokens per prefill batch |
| `eos` | `__init__` ← Config (engine sets from tokenizer) | `postprocess` | EOS id |
| `block_size` | `__init__` ← Config | Kept in sync with seq/BM | Page size |
| `block_manager` | `__init__` (or test inject) | schedule / preempt / postprocess | KV manager |
| `waiting` | `add`; prefill pops; `preempt` appendleft | `is_finished`; prefill FIFO | Prefill / recompute queue |
| `running` | Prefill completion; decode restore; finish remove; preempt source | Decode loop | Decode-eligible queue |

Scheduler also manages these **on `Sequence`**: `status`, `num_scheduled_tokens`, `is_prefill`, and advances `num_cached_tokens` in `postprocess` (after `hash_blocks`). It calls `append_token` when a sample should be kept.

---

## Config (`nanovllm/config.py`)

| Member | Manager | Readers | Meaning |
|---|---|---|---|
| `model` | Caller | HF load / tokenizer | Model directory |
| `max_num_batched_tokens` / `max_num_seqs` | Caller / defaults | Scheduler; warmup | Batch limits |
| `max_model_len` | Default; capped vs HF in `__post_init__` | Runner warmup | Context cap |
| `gpu_memory_utilization` | Caller / default | `allocate_kv_cache` | GPU fraction for KV |
| `tensor_parallel_size` | Caller / default | Engine spawn; runner | TP size |
| `enforce_eager` | Caller / default | Runner (graphs off) | Disable CUDA graphs |
| `hf_config` | `__post_init__` | Model / KV shapes | HF config |
| `eos` | Engine overwrites from tokenizer | Scheduler | EOS id |
| `kvcache_block_size` | Caller / default (`% 256 == 0`) | `Sequence.block_size`; BM; runner | Page size |
| `num_kvcache_blocks` | Often `-1` then set by runner after mem estimate | BM construction; KV tensor | Pool size |

---

## SamplingParams (`nanovllm/sampling_params.py`)

| Member | Manager | Readers | Meaning |
|---|---|---|---|
| `temperature` / `max_tokens` / `ignore_eos` | User at construct (immutable afterward) | Copied onto `Sequence` at request add | Sampling / stop config |

---

## LLMEngine (key fields)

| Member | Manager | Readers | Meaning |
|---|---|---|---|
| `scheduler` | `__init__` | `add_request` / `step` / `generate` | Queues + policy |
| `model_runner` | `__init__` | `step` run; `exit` | Execute model |
| `tokenizer` | `__init__` | encode/decode; sets `config.eos` | HF tokenizer |
| `ps` / `events` | `__init__` TP spawn | `exit` / IPC | Worker procs |

---

## Context (`nanovllm/utils/context.py`)

Filled each step by `ModelRunner.prepare_prefill` / `prepare_decode` via `set_context`, cleared by `reset_context`.

| Member | Manager | Readers | Meaning |
|---|---|---|---|
| `is_prefill` | ModelRunner | Attention; LM head | Kernel path |
| `cu_seqlens_q` / `cu_seqlens_k` | From prefill metadata | Attention varlen; LM head | Cumulative lengths |
| `max_seqlen_q` / `max_seqlen_k` | Prefill metadata | Attention | Max lengths |
| `slot_mapping` | Prefill/decode metadata | `store_kvcache`; graphs | Token → KV slot |
| `context_lens` | Decode metadata | `flash_attn_with_kvcache` | Per-seq KV length |
| `block_tables` | `build_block_tables` (+ prefill may omit) | Attention; graphs | Batched physical tables |

---

## Cross-component summary

```text
SamplingParams ──copy──► Sequence.{temperature, max_tokens, ignore_eos}
Config ──► Scheduler limits / eos
       ──► Sequence.block_size (via LLMEngine)
       ──► BlockManager(block_size, num_blocks)
       ──► ModelRunner (KV tensor, TP, eager)

BlockManager  WRITE Sequence.block_table (+ initial num_cached_tokens)
Scheduler     WRITE status, num_scheduled_tokens, is_prefill;
              ADVANCE num_cached_tokens in postprocess;
              CALL append_token / BlockManager APIs
input_metadata READ Sequence (+ block_table) ──► Context (via ModelRunner)
Attention     READ Context
```

See also: [`state_invariants.md`](state_invariants.md), [`call_graph.md`](call_graph.md).
