# 04. Prefill and Decode

## Learning Objectives

- Distinguish prompt prefill from autoregressive decode.
- Implement decode scheduling, preemption, and postprocess behavior.
- Avoid sampling side effects during incomplete chunked prefill.

## Why Exists

LLM inference has two very different phases. Prefill processes many prompt tokens at once; decode processes one new token per running request. The scheduler must expose that difference to the runner.

## Call Path

`Scheduler.schedule()` returns `(seqs, True)` for prefill and `(seqs, False)` for decode. `ModelRunner.run()` chooses `prepare_prefill()` or `prepare_decode()`. `Scheduler.postprocess()` then hashes completed blocks, advances cache counters, and appends sampled tokens when appropriate.

## Files/functions

- `nanovllm/engine/scheduler.py`: `schedule` decode branch, `preempt`, `postprocess`.
- `nanovllm/engine/block_manager.py`: `can_append`, `may_append`, `deallocate`, `hash_blocks`.
- `nanovllm/engine/input_metadata.py`: `build_prefill_metadata`, `build_decode_metadata`.
- `tests/milestones/test_03_prefill_decode.py`.

## Data Structures

`is_prefill` is both a batch-level return value and a per-sequence flag. During decode, `seq.num_scheduled_tokens=1` and `seq.is_prefill=False`.

## State Transitions

Decode keeps sequences in `RUNNING` until EOS or `max_tokens`. Preemption sets `status=WAITING`, `is_prefill=True`, frees blocks, and places the sequence at the front of `waiting`.

## Tensor Shapes

Prefill for A and B uses 8 tokens total. Decode for A and B uses 2 tokens total, one last token per sequence. Decode metadata contains `input_ids=[14,22]`, `positions=[4,2]`, and `context_lens=[5,3]`.

## Pseudocode

```text
decode:
    pop running sequences until batch full
    ensure cache can append if a new block is needed
    preempt lower-priority work if necessary
    mark each scheduled sequence for one token
postprocess:
    hash completed full blocks
    advance num_cached_tokens
    if incomplete chunked prefill: do not append sampled token
    otherwise append sampled token
    finish on eos or max_tokens
```

## TODO IDs

- `TODO-L2-SCHED-02`: decode scheduling loop.
- `TODO-L2-SCHED-03`: preempt a running sequence.
- `TODO-L2-SCHED-04`: postprocess sampled tokens and finish sequences.

## Hints

`postprocess()` receives sampled token IDs even for prefill batches. For chunked prefill, ignore that sampled token until the prompt is fully cached.

## Common Bugs

- Appending during incomplete chunked prefill.
- Forgetting to remove finished sequences from `running`.
- Failing to deallocate finished or preempted sequences.
- Not restoring preempted sequences to prefill mode.

## Tests

Run:

```bash
python tools/run_milestone.py 3
```

The tests cover one-token decode, EOS finish, `max_tokens` finish, chunked prefill, and explicit preemption.

## Debugging

Print `seq.status`, `seq.is_prefill`, `seq.num_cached_tokens`, `seq.num_scheduled_tokens`, and queue contents after each scheduler call. Most bugs show up as one field changing too early.

## Expected Intermediate Behavior

After full prefill, A and B are in `running`. The next scheduler call returns decode work with one scheduled token for each sequence and `is_prefill=False`.

## Reflection

The sampled token belongs to the logical sequence only after the KV state that produced it is valid. That is the key reason `postprocess()` handles chunked prefill specially.

## Connection to Production vLLM

Production vLLM has more elaborate preemption and recomputation policies, but the same tradeoff exists: when KV memory is scarce, the engine may discard cache for some requests and recompute later.
