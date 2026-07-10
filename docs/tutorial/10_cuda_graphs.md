# 10. CUDA Graphs

## Learning Objectives

- Explain why decode can replay CUDA graphs while prefill stays eager.
- Understand graph batch-size buckets and static input buffers.
- Implement graph capture and replay without changing scheduler semantics.

## Why Exists

Decode executes many small, repeated GPU workloads. CUDA graphs reduce launch overhead by capturing a static graph and replaying it with new inputs.

## Call Path

`ModelRunner.__init__()` calls `capture_cudagraph()` unless `Config.enforce_eager` is true. During decode, `run_model()` chooses a graph bucket, copies current inputs and context tensors into static buffers, replays the graph, and computes logits.

## Files/functions

- `nanovllm/engine/model_runner.py`: `run_model`, `capture_cudagraph`.
- `nanovllm/utils/context.py`: graph capture context.
- `nanovllm/layers/attention.py`: decode attention during capture/replay.
- `tests/milestones/test_10_cuda_graphs.py`.

## Data Structures

`ModelRunner.graph_vars` contains static tensors: `input_ids`, `positions`, `slot_mapping`, `context_lens`, `block_tables`, and `outputs`. `graph_bs` stores supported batch sizes such as `1, 2, 4, 8, 16, ...`.

## State Transitions

CUDA graph capture happens once after warmup and KV-cache allocation. Runtime decode copies current batch data into static buffers; it does not change sequence state.

## Tensor Shapes

For a graph batch size `bs`, capture uses:

- `input_ids`: `[bs]`.
- `positions`: `[bs]`.
- `slot_mapping`: `[bs]`.
- `context_lens`: `[bs]`.
- `block_tables`: `[bs, max_num_blocks]`.
- `outputs`: `[bs, hidden_size]`.

## Pseudocode

```text
if is_prefill or enforce_eager or batch_size > 512:
    run model eagerly
else:
    graph_bs = first bucket >= batch_size
    copy current inputs into graph_vars
    clear unused slots
    graph.replay()
    return logits for first batch_size rows
```

## TODO IDs

- `TODO-L3-CUDAGRAPH-01`: CUDA graph replay path in `run_model`.
- `TODO-L3-CUDAGRAPH-02`: capture decode CUDA graphs.

## Hints

Capture decode only. Prefill has variable token counts and sequence lengths, which makes it a poor fit for these fixed graph buckets.

## Common Bugs

- Replaying a graph with stale `slot_mapping` or `context_lens`.
- Forgetting to zero or fill unused static buffer regions.
- Capturing before `set_context()` points attention at static graph tensors.
- Returning padded rows instead of slicing to the real batch size.

## Tests

Run GPU graph tests:

```bash
python tools/run_milestone.py 10
```

These tests may skip without CUDA and required dependencies.

## Debugging

Use `enforce_eager=True` to establish correctness first. Then enable `NANOVLLM_DEBUG_RUNNER=1` and look for `cudagraph_replay` logs with expected `bs` and `graph_bs`.

## Expected Intermediate Behavior

With eager disabled, prefill still logs the eager path. Decode batches with size at most 512 should replay a graph bucket greater than or equal to the actual batch size.

## Reflection

CUDA graphs optimize the execution path, not the scheduling policy. If graph replay changes outputs, the static metadata buffers are probably stale or incorrectly sliced.

## Connection to Production vLLM

Production vLLM uses more advanced graph and compilation strategies, but the same constraints apply: static shapes, reusable buffers, and careful separation between prefill variability and decode regularity.
