# 08. Sampling

## Learning Objectives

- Implement temperature sampling using the Gumbel-max trick.
- Explain why this project forbids exact greedy `temperature=0`.
- Check sampler output shape and vocabulary bounds.

## Why Exists

The model returns logits. The engine needs token IDs. `Sampler` is the small layer that turns a batch of logits into one sampled token per sequence.

## Call Path

`ModelRunner.run()` builds per-sequence temperatures with `prepare_sample()`, computes logits with `run_model()`, and calls `self.sampler(logits, temperatures)`. `Scheduler.postprocess()` receives those sampled token IDs.

## Files/functions

- `nanovllm/layers/sampler.py`: `Sampler.forward`.
- `nanovllm/sampling_params.py`: `SamplingParams.__post_init__`.
- `nanovllm/engine/model_runner.py`: `prepare_sample`, `run`.
- `tests/milestones/test_07_sampling.py`.
- `tests/oracles/sampling_oracle.py`.

## Data Structures

Inputs:

- `logits`: tensor shaped `[batch, vocab_size]`.
- `temperatures`: tensor shaped `[batch]`.

Output:

- token IDs shaped `[batch]`.

## State Transitions

Sampling does not mutate sequences. It returns token IDs. The scheduler appends or discards them depending on prefill/decode state.

## Tensor Shapes

For two running sequences and a vocabulary of 50, logits are `[2,50]` and temperatures are `[2]`. The output is `[2]`.

## Pseudocode

```text
scaled = logits / temperatures[:, None]
probs = softmax(scaled)
noise = exponential(1) with same shape as probs
tokens = argmax(probs / clamp(noise), dim=-1)
```

## TODO IDs

- `TODO-L1-SAMPLE-01`: temperature sampling with Gumbel-max trick.

## Hints

Use floating-point logits before division. Very low but positive temperature approximates greedy behavior; exact zero is not allowed by `SamplingParams`.

## Common Bugs

- Implementing `argmax(logits)` and skipping randomness.
- Accepting `temperature=0` despite the assertion.
- Dividing temperatures across the wrong dimension.
- Returning probabilities instead of token IDs.

## Tests

Run:

```bash
python tools/run_milestone.py 7
```

The tests verify shape, dtype, low-temperature argmax-like behavior, and bounds.

## Debugging

Set `torch.manual_seed(...)` in local experiments. Enable `NANOVLLM_DEBUG_SAMPLING=1` to inspect sampled tokens and temperatures.

## Expected Intermediate Behavior

With logits `[[0, 10, 0]]` and temperature `1e-4`, the sampled token should almost always be `1` because the softmax peak overwhelms noise.

## Reflection

The sampler is intentionally simple. The important design point is where sampling belongs: after model logits and before scheduler postprocess.

## Connection to Production vLLM

Production vLLM supports many sampling controls, including top-k, top-p, penalties, logprobs, and greedy decoding. This course isolates the core idea: per-request temperature plus stochastic token selection.
