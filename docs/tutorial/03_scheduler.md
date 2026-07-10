# 03. Scheduler

## Learning Objectives

- Implement prefill admission from the waiting queue.
- Respect `max_num_seqs`, `max_num_batched_tokens`, and KV-cache capacity.
- Understand why prefill is tried before decode in this codebase.

## Why Exists

The scheduler turns many independent `Sequence` objects into a batch for one model execution. It is the policy layer between request state and GPU work.

## Call Path

`LLMEngine.step()` calls `Scheduler.schedule()`. If any waiting sequence can be admitted, `schedule()` returns a prefill batch and `is_prefill=True`. Otherwise it builds a decode batch from `running`.

## Files/functions

- `nanovllm/engine/scheduler.py`: `Scheduler.__init__`, `add`, `schedule`, `preempt`, `postprocess`.
- `nanovllm/engine/block_manager.py`: `can_allocate`, `allocate`, `can_append`, `may_append`.
- `tests/milestones/test_02_scheduler_basic.py`.
- `tests/milestones/test_03_prefill_decode.py`.

## Data Structures

The scheduler owns two deques:

- `waiting`: requests not fully prefetched or preempted requests that must be recomputed.
- `running`: requests with completed prefill that can decode.

It also owns one `BlockManager`.

## State Transitions

During prefill scheduling, a sequence may move from `WAITING` to `RUNNING` only when `num_cached_tokens + num_scheduled_tokens == num_tokens`. Chunked prefill leaves it in `WAITING`.

## Tensor Shapes

The scheduler does not create tensors. It sets counters that metadata builders later flatten. For A and B, a full prefill batch has `num_scheduled_tokens` 5 and 3, so metadata sees 8 input tokens.

## Pseudocode

```text
scheduled = []
while waiting and batch has room:
    seq = waiting[0]
    ask BlockManager whether blocks can be allocated or reused
    compute remaining prompt tokens
    choose scheduled token count under token budget
    allocate blocks on first admission
    if prompt complete: move WAITING -> RUNNING
if scheduled:
    return scheduled, True
otherwise:
    build decode batch from running
```

## TODO IDs

- `TODO-L2-SCHED-01`: prefill scheduling loop.

Decode, preemption, and postprocess continue in the next chapter: `TODO-L2-SCHED-02`, `TODO-L2-SCHED-03`, and `TODO-L2-SCHED-04`.

## Hints

Only the first sequence in a prefill batch may be chunked when the token budget is too small. If one sequence is already scheduled and the next does not fit, stop the batch.

## Common Bugs

- Moving a chunked-prefill sequence into `running` too early.
- Forgetting to allocate blocks before setting `block_table`.
- Counting cached prefix tokens as newly batched tokens.
- Ignoring `max_num_seqs`.

## Tests

Run:

```bash
python tools/run_milestone.py 2
```

The two-request test expects A and B to schedule together when token and sequence budgets allow it.

## Debugging

Enable:

```bash
export NANOVLLM_DEBUG_SCHEDULER=1
export NANOVLLM_DEBUG_KVCACHE=1
```

Then inspect scheduled sequence IDs, waiting/running lengths, and block tables.

## Expected Intermediate Behavior

With block size 4 and enough cache blocks, A `[10,11,12,13,14]` and B `[20,21,22]` should both leave `waiting`, enter `running`, and have non-empty `block_table` values.

## Reflection

Scheduling is not just batching; it is also deciding what memory must exist before the model can run. That is why `Scheduler` and `BlockManager` interact closely.

## Connection to Production vLLM

Production vLLM has more sophisticated policies, priorities, request groups, and fairness controls. The core shape is the same: admit prefill work, interleave decode work, and stay within token and KV-cache budgets.
