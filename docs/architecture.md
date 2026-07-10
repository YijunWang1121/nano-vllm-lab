# nano-vLLM Architecture

This page maps the course implementation to the actual source tree. It is intentionally high level: use it to understand where data moves, then open the referenced files while implementing each TODO.

## Public Surface

The user-facing API is `nanovllm.LLM` and `nanovllm.SamplingParams`.

- `nanovllm/__init__.py` exports `LLM` and `SamplingParams`.
- `nanovllm/llm.py` defines `LLM` as a thin subclass of `nanovllm.engine.llm_engine.LLMEngine`.
- `nanovllm/sampling_params.py` defines `SamplingParams`; `temperature=0` is forbidden by assertion.
- `nanovllm/config.py` defines `Config`, with `kvcache_block_size=256` by default and an assertion that the configured block size is a multiple of 256. The course tests override `Sequence.block_size=4` for small examples.

## Engine Construction

```mermaid
flowchart TD
    User["User imports nanovllm.LLM"] --> LLM["nanovllm/llm.py: LLM"]
    LLM --> Engine["nanovllm/engine/llm_engine.py: LLMEngine.__init__"]
    Engine --> Config["nanovllm/config.py: Config"]
    Config --> HFConfig["transformers.AutoConfig"]
    Engine --> SeqBlock["Sequence.block_size = config.kvcache_block_size"]
    Engine --> Workers{"tensor_parallel_size > 1?"}
    Workers -->|yes| Spawn["spawn ModelRunner workers"]
    Workers -->|no| Rank0["rank 0 ModelRunner only"]
    Spawn --> Runner["nanovllm/engine/model_runner.py: ModelRunner"]
    Rank0 --> Runner
    Runner --> Model["nanovllm/models/qwen3.py: Qwen3ForCausalLM"]
    Runner --> Sampler["nanovllm/layers/sampler.py: Sampler"]
    Engine --> Tokenizer["transformers.AutoTokenizer"]
    Engine --> Scheduler["nanovllm/engine/scheduler.py: Scheduler"]
    Scheduler --> Blocks["nanovllm/engine/block_manager.py: BlockManager"]
```

`LLMEngine.__init__` owns process setup, tokenizer loading, the rank-0 `ModelRunner`, and the `Scheduler`. The `ModelRunner` owns the model, sampler, KV-cache tensor, CUDA graph capture, and tensor-parallel worker IPC.

## Request Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Created: LLMEngine.add_request()
    Created --> WAITING: Sequence(prompt, SamplingParams)
    WAITING --> RUNNING: Scheduler.schedule() prefill completes
    WAITING --> WAITING: chunked prefill incomplete
    RUNNING --> RUNNING: decode step appends token
    RUNNING --> WAITING: Scheduler.preempt()
    RUNNING --> FINISHED: eos or max_tokens
    FINISHED --> [*]: BlockManager.deallocate()
```

The request state lives in `nanovllm/engine/sequence.py`. `SequenceStatus` has exactly `WAITING`, `RUNNING`, and `FINISHED`. Important fields are `token_ids`, `last_token`, `num_prompt_tokens`, `num_cached_tokens`, `num_scheduled_tokens`, `is_prefill`, and `block_table`.

## One Inference Iteration

```mermaid
sequenceDiagram
    participant Engine as LLMEngine.step
    participant Sched as Scheduler.schedule
    participant BM as BlockManager
    participant Runner as ModelRunner.run
    participant Meta as input_metadata.py
    participant Model as Qwen3ForCausalLM
    participant Sample as Sampler

    Engine->>Sched: schedule()
    Sched->>BM: can_allocate/allocate or can_append/may_append
    Sched-->>Engine: seqs, is_prefill
    Engine->>Runner: call("run", seqs, is_prefill)
    Runner->>Meta: build_prefill_metadata or build_decode_metadata
    Runner->>Model: model(input_ids, positions)
    Model-->>Runner: hidden states/logits
    Runner->>Sample: Sampler(logits, temperatures)
    Sample-->>Runner: token_ids
    Runner-->>Engine: token_ids
    Engine->>Sched: postprocess(seqs, token_ids, is_prefill)
    Sched->>BM: hash_blocks/deallocate as needed
```

`LLMEngine.step()` returns completed outputs plus a signed token count: positive for prefill throughput and negative for decode throughput.

## Prefill vs Decode

```mermaid
flowchart LR
    Schedule["Scheduler.schedule()"] --> Waiting{"waiting non-empty?"}
    Waiting -->|yes| Prefill["Prefill batch"]
    Waiting -->|no| Decode["Decode batch"]

    Prefill --> PrefillMeta["build_prefill_metadata"]
    PrefillMeta --> PrefillAttention["flash_attn_varlen_func"]
    PrefillAttention --> PrefillPost["postprocess: maybe append sampled token"]

    Decode --> DecodeMeta["build_decode_metadata"]
    DecodeMeta --> DecodeAttention["flash_attn_with_kvcache"]
    DecodeAttention --> DecodePost["postprocess: append one token"]
```

Prefill can schedule many prompt tokens per sequence and supports chunking. Decode schedules one token per running sequence. The scheduler always tries prefill first; decode only runs when no prefill batch was admitted.

## KV-Cache Block Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Free: Block(ref_count=0)
    Free --> Allocated: _allocate_block()
    Allocated --> Referenced: allocate(seq)
    Referenced --> Shared: prefix cache reuse increments ref_count
    Shared --> Referenced: deallocate one sequence
    Referenced --> Hashed: hash_blocks() records xxhash
    Hashed --> Free: deallocate() ref_count reaches 0
    Free --> Allocated: reused for new sequence
```

`BlockManager` implements paged attention metadata, not tensor storage. It tracks physical block IDs, `free_block_ids`, `used_block_ids`, prefix hashes, and each `Block.ref_count`. Model tensor storage is allocated later in `ModelRunner.allocate_kv_cache()`.

Running example with educational `block_size=4`:

- Request A prompt `[10, 11, 12, 13, 14]` has logical blocks `[0, 1]`.
- Request B prompt `[20, 21, 22]` has logical blocks `[0]`.
- Example physical mapping: A `block_table=[7, 2]`, B `block_table=[5]`.
- A decode slot for the current last token is `physical_block=2`, `offset=0`, so `slot=2 * 4 + 0 = 8`.

## Model-Runner Execution Path

```mermaid
flowchart TD
    Run["ModelRunner.run(seqs, is_prefill)"] --> Branch{"is_prefill?"}
    Branch -->|yes| PrepPrefill["prepare_prefill"]
    Branch -->|no| PrepDecode["prepare_decode"]
    PrepPrefill --> ContextPrefill["set_context(is_prefill=True, cu_seqlens, slot_mapping, block_tables?)"]
    PrepDecode --> ContextDecode["set_context(is_prefill=False, context_lens, block_tables, slot_mapping)"]
    ContextPrefill --> RunModel["run_model"]
    ContextDecode --> RunModel
    RunModel --> Path{"prefill/eager/bs>512?"}
    Path -->|yes| Eager["model(input_ids, positions)"]
    Path -->|no| Graph["CUDA graph replay"]
    Eager --> Logits["compute_logits"]
    Graph --> Logits
    Logits --> Sample["Sampler.forward"]
    Sample --> Reset["reset_context()"]
```

Attention reads `nanovllm.utils.context.get_context()` inside `nanovllm/layers/attention.py`. Prefill stores K/V into the cache with the Triton `store_kvcache` kernel and calls `flash_attn_varlen_func`. Decode calls `flash_attn_with_kvcache` using `context_lens` and `block_tables`.

## Tensor Parallel Overview

`ModelRunner` initializes NCCL with `dist.init_process_group("nccl", ...)`. Rank 0 communicates method calls to worker ranks through Python `SharedMemory` plus per-rank events. Layer sharding lives in `nanovllm/layers/linear.py`:

- `ColumnParallelLinear` loads an output-dimension shard and returns a partial output.
- `QKVParallelLinear` separately loads Q, K, and V shards into one packed projection.
- `RowParallelLinear` loads an input-dimension shard and uses `dist.all_reduce()` to combine partial outputs.
