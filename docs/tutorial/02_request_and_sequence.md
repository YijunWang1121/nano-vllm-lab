# 02. Request and Sequence

## Learning Objectives

- Implement the request state container used by the scheduler.
- Compute logical KV-cache block layout from token length.
- Understand why `last_token`, `num_cached_tokens`, and `num_scheduled_tokens` are separate fields.

## Why Exists

`Sequence` is the smallest unit of scheduling. It represents one prompt plus generated completion tokens, independent of batching and GPU execution.

## Call Path

`LLMEngine.add_request()` tokenizes a prompt if needed, constructs `Sequence(prompt, sampling_params)`, and passes it to `Scheduler.add()`. The scheduler later reads `len(seq)`, `seq.block(i)`, `seq.last_token`, and `seq.block_table`.

## Files/functions

- `nanovllm/engine/sequence.py`: `SequenceStatus`, `Sequence.__init__`, `append_token`, `num_completion_tokens`, `prompt_token_ids`, `completion_token_ids`, `num_blocks`, `last_block_num_tokens`, `block`.
- `nanovllm/sampling_params.py`: `SamplingParams`.
- `tests/milestones/test_01_sequence.py`: expected behavior.
- `tests/helpers.py`: `make_sequence`, `EXAMPLE_A_PROMPT`, `EXAMPLE_B_PROMPT`.

## Data Structures

Important fields:

- `token_ids`: prompt tokens plus generated tokens.
- `last_token`: the token fed during decode.
- `num_tokens`: current total length.
- `num_prompt_tokens`: original prompt length.
- `num_cached_tokens`: prompt/decode tokens already represented in KV cache.
- `num_scheduled_tokens`: tokens selected for the current model run.
- `block_table`: physical KV block IDs assigned later by `BlockManager`.

## State Transitions

New sequences start as `WAITING` and `is_prefill=True`. `append_token()` extends `token_ids`, updates `last_token`, and increments `num_tokens`; it does not change queue membership or allocate blocks.

## Tensor Shapes

`Sequence` itself stores Python lists. With test `block_size=4`, Request A length 5 has `num_blocks=2`, `last_block_num_tokens=1`, `block(0)=[10, 11, 12, 13]`, and `block(1)=[14]`.

## Pseudocode

```text
initialize:
    status = WAITING
    token_ids = copy(prompt)
    last_token = prompt[-1]
    counters reflect prompt length
    cache counters start at 0
append_token(token):
    push token
    update last_token and num_tokens
num_blocks:
    ceil(num_tokens / block_size)
```

## TODO IDs

- `TODO-L1-SEQ-01`: initialize tracking fields.
- `TODO-L1-SEQ-02`: append a generated token.
- `TODO-L1-SEQ-03`: compute block indexing helpers.

## Hints

Keep `num_prompt_tokens` fixed after construction. Completion length should be derived from `num_tokens - num_prompt_tokens`, not tracked independently.

## Common Bugs

- Mutating the caller's prompt list instead of copying it.
- Forgetting to update `last_token` in `append_token()`.
- Returning zero blocks for a non-empty prompt.
- Computing `last_block_num_tokens` as `num_tokens % block_size` without handling full blocks.

## Tests

Run:

```bash
python tools/run_milestone.py 1
```

The tests verify initialization, append, block slicing, Request B's single block, and `is_finished`.

## Debugging

Use a tiny script or pytest failure output to print `seq.token_ids`, `seq.num_tokens`, `seq.num_blocks`, and `seq.block_table`. No CUDA is involved.

## Expected Intermediate Behavior

For A `[10, 11, 12, 13, 14]` with block size 4, appending `99` should make completion tokens `[99]`, total length 6, and logical blocks `[10,11,12,13]`, `[14,99]`.

## Reflection

`Sequence` is deliberately not a scheduler. If a change feels like queue policy or memory policy, it probably belongs in `Scheduler` or `BlockManager`.

## Connection to Production vLLM

Production vLLM has richer sequence groups, request IDs, and output objects, but the same invariant remains: scheduling decisions depend on current token length, prompt/completion split, and KV-cache block mapping.
