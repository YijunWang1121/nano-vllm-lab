# 09. End-to-End Eager

## Learning Objectives

- Run the engine loop with a fake runner, then with a real eager GPU runner.
- Trace how sampled tokens become completions.
- Verify cleanup after requests finish.

## Why Exists

Individual components are easier to test in isolation, but the engine only works when scheduling, cache allocation, model execution, sampling, and postprocess agree on one contract.

## Call Path

`LLMEngine.generate()` adds requests and loops over `LLMEngine.step()`. One step calls `Scheduler.schedule()`, `ModelRunner.call("run", seqs, is_prefill)`, and `Scheduler.postprocess()`.

## Files/functions

- `nanovllm/engine/llm_engine.py`: `LLMEngine.add_request`, `step`, `generate`.
- `nanovllm/engine/model_runner.py`: `ModelRunner.run`, `run_model`.
- `tests/oracles/fake_model_runner.py`: fake runner contract.
- `tests/milestones/test_08_engine_fake.py`.
- `tests/milestones/test_09_end_to_end_eager.py`.

## Data Structures

The end-to-end loop passes `list[Sequence]` and `is_prefill` from scheduler to runner. Runner returns `list[int]` token IDs in the same order. Postprocess zips sequences with token IDs.

## State Transitions

The loop ends when both `waiting` and `running` are empty. Finished sequences should have completion tokens and empty `block_table` after deallocation.

## Tensor Shapes

Fake runner tests avoid tensors. Real eager GPU generation uses prefill shapes from chapter 6 and decode shapes of one token per sequence.

## Pseudocode

```text
finished = {}
while not scheduler.is_finished():
    seqs, is_prefill = scheduler.schedule()
    token_ids = runner.run(seqs, is_prefill)
    scheduler.postprocess(seqs, token_ids, is_prefill)
    collect finished completion_token_ids
```

## TODO IDs

- `TODO-L2-ENGINE-01`: wire fake end-to-end engine step and understand the runner contract.

Milestone 9 has no new TODO; it validates real eager GPU generation.

## Hints

The fake runner is not the production engine. Use it to prove the scheduler and postprocess loop before debugging CUDA, FlashAttention, or model weights.

## Common Bugs

- Returning token IDs in a different order than scheduled sequences.
- Forgetting to deallocate blocks at finish.
- Treating prefill throughput count as decode count.
- Running GPU tests without setting `NANOVLLM_TEST_MODEL`.

## Tests

CPU fake loop:

```bash
python tools/run_milestone.py 8
```

GPU eager:

```bash
export NANOVLLM_TEST_MODEL="$HOME/huggingface/Qwen3-0.6B"
python tools/run_milestone.py 9
```

## Debugging

Use fake-runner tests first. For real GPU, enable `NANOVLLM_DEBUG_ENGINE=1`, `NANOVLLM_DEBUG_RUNNER=1`, and one other subsystem at a time.

## Expected Intermediate Behavior

In the fake two-request test, A and B should each finish with two generated tokens, all block tables should be empty, and `Scheduler.is_finished()` should return true.

## Reflection

End-to-end failures are often ownership failures. Identify which component was responsible for the wrong state before changing code.

## Connection to Production vLLM

Production vLLM's engine loop has async serving, streaming outputs, and richer request objects. The core iteration remains schedule, execute, sample, postprocess, and return newly finished outputs.
