# 14. Compare With vLLM

## Learning Objectives

- Relate nano-vLLM components to production vLLM concepts.
- Identify which features are intentionally omitted from this course project.
- Compare behavior without assuming identical APIs or performance claims.

## Why Exists

nano-vLLM is educational. It borrows key vLLM ideas, but it is not a drop-in replacement for the production serving stack.

## Call Path

nano-vLLM's user path is `nanovllm.LLM.generate()`. Production vLLM exposes richer offline and serving APIs. The comparable internal path is still request admission, scheduling, model execution, sampling, and output assembly.

## Files/functions

- `nanovllm/engine/llm_engine.py`: local engine loop.
- `nanovllm/engine/scheduler.py`: simplified scheduler.
- `nanovllm/engine/block_manager.py`: paged blocks and prefix cache.
- `nanovllm/engine/model_runner.py`: one-process or tensor-parallel runner.
- `nanovllm/layers/attention.py`: FlashAttention integration.

## Data Structures

nano-vLLM uses `Sequence` for one request. Production vLLM has richer request and sequence group structures for serving, streaming, scheduling policies, and output metadata.

## State Transitions

The simplified states are `WAITING`, `RUNNING`, and `FINISHED`. Production vLLM has more lifecycle detail because it handles concurrent serving, cancellation, streaming, preemption modes, and scheduler policy.

## Tensor Shapes

The core tensor ideas match: flattened token batches, positions, KV-cache block tables, cache slots, and decode context lengths. Production vLLM supports more backends and model families.

## Pseudocode

```text
nano-vLLM:
    schedule local batch
    run local/tensor-parallel model
    sample token ids
    postprocess sequences
production vLLM:
    schedule serving/offline work
    dispatch to workers/backends
    sample with rich parameters
    stream or return structured outputs
```

## TODO IDs

This comparison has no new TODO. It synthesizes all milestones from `TODO-L1-SEQ-01` through `TODO-L3-TP-02`.

## Hints

Compare concepts, not line-for-line implementation. nano-vLLM is valuable because it removes layers that would distract from the core engine.

## Common Bugs

- Assuming nano-vLLM supports all vLLM `SamplingParams`.
- Expecting production serving APIs.
- Treating benchmark numbers as portable across models or hardware.
- Porting vLLM code directly instead of preserving this repository's simpler invariants.

## Tests

Use the nano-vLLM test suite for this repository:

```bash
python -m pytest -q -m "not gpu"
python -m pytest -q -m gpu
```

Do not use production vLLM tests as direct compatibility tests.

## Debugging

When comparing behavior, first reduce the case to one prompt, a small `max_tokens`, and positive temperature. Then compare scheduling and cache metadata before comparing generated text.

## Expected Intermediate Behavior

After completing the course, you should be able to point to the nano-vLLM file that corresponds to each major vLLM concept: engine, scheduler, cache manager, attention backend integration, sampler, model runner, and tensor parallel layers.

Grounded comparison:

| Area | nano-vLLM in this repo | Production vLLM concept |
|---|---|---|
| Public API | `nanovllm.LLM`, `SamplingParams` | Offline `LLM` plus serving APIs |
| Request state | `Sequence` with `WAITING/RUNNING/FINISHED` | Rich request, sequence, and sequence-group state |
| Scheduler | `Scheduler` with waiting/running deques, prefill priority, decode, preempt | Policy-rich scheduler with serving features |
| KV cache | `BlockManager`, physical block IDs, `ref_count`, prefix hashes | Paged KV cache with production allocation policies |
| Prefix caching | `hash_blocks()` plus `can_allocate()` reuse | Prefix cache integrated with production scheduler/cache manager |
| Attention | `flash_attn_varlen_func`, `flash_attn_with_kvcache`, Triton cache store | Multiple attention backends and metadata types |
| Sampling | Temperature plus Gumbel-max; no greedy zero temperature | Many sampling parameters and output options |
| CUDA graphs | Decode graph buckets in `ModelRunner` | More advanced graph/compile execution strategy |
| Tensor parallelism | `ColumnParallelLinear`, `QKVParallelLinear`, `RowParallelLinear` | Distributed executor and broader parallel features |
| Serving | Not implemented | OpenAI-compatible server and streaming |

## Reflection

The best outcome is not memorizing nano-vLLM. It is learning the stable abstractions that make production vLLM understandable.

## Connection to Production vLLM

Every chapter connects to production vLLM through one idea: high-throughput generation is a coordination problem between scheduling policy, KV-cache memory, attention metadata, and model execution.
