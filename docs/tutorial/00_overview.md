# 00. Overview

## Learning Objectives

- Explain the end-to-end path from `nanovllm.LLM.generate()` to sampled token IDs.
- Distinguish prefill, decode, scheduling, KV-cache allocation, metadata building, attention, and sampling.
- Use the course TODO registry without copying full source solutions.

## Why Exists

The project is small enough to read, but the concepts are layered. This overview gives you the map before you implement individual pieces.

## Call Path

`nanovllm.LLM` is a subclass of `nanovllm.engine.llm_engine.LLMEngine`. `LLMEngine.generate()` adds requests, loops over `LLMEngine.step()`, calls `Scheduler.schedule()`, runs `ModelRunner.run()`, and then calls `Scheduler.postprocess()`.

## Files/functions

- `nanovllm/__init__.py`: public exports.
- `nanovllm/llm.py`: `LLM`.
- `nanovllm/engine/llm_engine.py`: `LLMEngine.__init__`, `add_request`, `step`, `generate`.
- `course/todos.yaml`: TODO registry.
- `course/milestones.yaml`: milestone order.
- `tools/course_status.py`: progress report.
- `tools/run_milestone.py`: one-milestone runner.

## Data Structures

The central object is `Sequence`. It carries token IDs, prompt length, completion length, status, scheduled-token counters, and a `block_table` mapping logical blocks to physical KV-cache blocks.

Running example: Request A is `[10, 11, 12, 13, 14]`; Request B is `[20, 21, 22]`; educational block size is 4.

## State Transitions

Requests begin as `WAITING`, become `RUNNING` after prefill completes, may temporarily return to `WAITING` on preemption, and end as `FINISHED` when EOS or `max_tokens` is reached.

## Tensor Shapes

At the course level, most CPU tests use lists. GPU execution converts those lists to tensors:

- `input_ids`: `[num_tokens]` for prefill, `[num_seqs]` for decode.
- `positions`: same length as `input_ids`.
- `block_tables`: `[num_seqs, max_num_blocks]`.
- `slot_mapping`: one physical cache slot per scheduled token.

## Pseudocode

```text
for prompt in prompts:
    engine.add_request(prompt, sampling_params)
while not scheduler.is_finished():
    seqs, is_prefill = scheduler.schedule()
    token_ids = model_runner.run(seqs, is_prefill)
    scheduler.postprocess(seqs, token_ids, is_prefill)
```

## TODO IDs

All course TODO IDs appear in `course/todos.yaml`. Start with `TODO-L1-SEQ-01`, `TODO-L1-SEQ-02`, and `TODO-L1-SEQ-03`; end with `TODO-L2-TP-01` and `TODO-L3-TP-02`.

## Hints

Read tests before code. Each milestone test encodes the expected behavior with small numbers, especially the A/B running example.

## Common Bugs

- Mixing up logical block indexes with physical block IDs.
- Appending sampled tokens during incomplete chunked prefill.
- Forgetting that `SamplingParams.temperature=0` is forbidden.
- Treating production `kvcache_block_size=256` as the test block size.

## Tests

Run all CPU milestones:

```bash
python -m pytest -q -m "not gpu"
```

Run one milestone:

```bash
python tools/run_milestone.py 1
```

## Debugging

Use `NANOVLLM_DEBUG_SCHEDULER=1` and `NANOVLLM_DEBUG_KVCACHE=1` first. They show which sequences were admitted and how blocks were allocated.

## Expected Intermediate Behavior

After the first milestones, you should be able to create a `Sequence`, append tokens, compute block counts, and schedule A and B into a prefill batch without touching CUDA.

## Reflection

The course is about invariants more than syntax. At every step, ask which object owns the state and when that state is allowed to change.

## Connection to Production vLLM

Production vLLM has more policies, APIs, and distributed machinery, but the same core split appears: scheduler, sequence/request state, paged KV cache, attention metadata, model runner, and sampler.
