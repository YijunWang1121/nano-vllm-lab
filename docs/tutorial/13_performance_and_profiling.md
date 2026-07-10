# 13. Performance and Profiling

## Learning Objectives

- Identify which subsystem limits throughput in prefill versus decode.
- Use debug logs and tests to isolate performance regressions.
- Understand the interaction between batching, KV-cache capacity, CUDA graphs, and tensor parallelism.

## Why Exists

Correctness comes first, but inference engines are built for throughput. This chapter helps you reason about performance without turning every bug into a GPU mystery.

## Call Path

`LLMEngine.generate()` measures prefill and decode throughput around `LLMEngine.step()`. `step()` includes scheduling, model runner execution, sampling, and postprocess.

## Files/functions

- `nanovllm/engine/llm_engine.py`: throughput counters in `generate`.
- `nanovllm/engine/scheduler.py`: batch construction.
- `nanovllm/engine/model_runner.py`: eager path, graph replay, KV-cache allocation.
- `bench.py`: benchmark entry point.
- `nanovllm/utils/debug.py`: subsystem logs.

## Data Structures

Performance-sensitive state includes `max_num_batched_tokens`, `max_num_seqs`, `num_kvcache_blocks`, `block_table` length, and graph batch-size buckets.

## State Transitions

Throughput changes as requests move from prefill to decode. Prefill is token-heavy and variable. Decode is repeated and shape-regular, making it the CUDA graph target.

## Tensor Shapes

Prefill cost scales with total scheduled prompt tokens and attention length. Decode cost scales with number of running sequences, context length, and cache block table size.

## Pseudocode

```text
time one engine step
if num_tokens > 0:
    prefill_toks_per_sec = num_tokens / elapsed
else:
    decode_toks_per_sec = -num_tokens / elapsed
```

## TODO IDs

This chapter introduces no new implementation TODO. It helps validate work from `TODO-L2-SCHED-01` through `TODO-L3-TP-02`.

## Hints

Profile only after CPU milestone tests pass. A wrong `slot_mapping` can look like bad performance because it causes recomputation, graph fallback, or incorrect cache use.

## Common Bugs

- Benchmarking with `enforce_eager=True` and expecting graph replay speed.
- Using too small `max_num_batched_tokens`, causing excessive chunked prefill.
- Running with debug logs enabled during timing.
- Comparing GPU results when model path, dtype, or CUDA dependencies differ.

## Tests

Run correctness first:

```bash
python -m pytest -q -m "not gpu"
python -m pytest -q -m gpu
```

Then use `bench.py` with a known local model if your environment supports it.

## Debugging

Start with scheduler logs to see batch sizes. Then use runner logs to see eager versus CUDA graph paths. Use attention logs to confirm prefill/decode routing.

## Expected Intermediate Behavior

With CUDA graphs enabled, decode batches at size 1, 2, 4, 8, or larger buckets should log graph replay. Prefill should continue to use the eager model path.

## Reflection

Performance work is meaningful only after invariants are stable. Otherwise, speed changes are often artifacts of doing less correct work.

## Connection to Production vLLM

Production vLLM adds scheduling heuristics, memory profiling, async serving, metrics, and backend selection. The same major levers remain: batch shape, KV-cache utilization, kernel choice, graph capture, and parallelism.
