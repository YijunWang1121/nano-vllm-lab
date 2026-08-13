"""Shared workloads for engine compare scripts."""

from __future__ import annotations

import os
import random
from random import choice, randint, seed


def default_model_path() -> str:
    return os.path.expanduser(
        os.environ.get("NANOVLLM_TEST_MODEL", "~/huggingface/Qwen3-0.6B/")
    )


def make_workload(
    num_seqs: int,
    min_input_len: int,
    max_input_len: int,
    min_output_len: int,
    max_output_len: int,
    seed_value: int = 0,
) -> tuple[list[list[int]], list[int]]:
    """Random token-id prompts (offline throughput style)."""
    seed(seed_value)
    lo_i, hi_i = max(1, min_input_len), max(min_input_len, max_input_len)
    lo_o, hi_o = max(1, min_output_len), max(min_output_len, max_output_len)
    prompts = [[randint(0, 10000) for _ in range(randint(lo_i, hi_i))] for _ in range(num_seqs)]
    max_tokens = [randint(lo_o, hi_o) for _ in range(num_seqs)]
    return prompts, max_tokens


def make_poisson_serving_workload(
    num_seqs: int,
    mean_input_len: int = 512,
    mean_output_len: int = 256,
    request_rate: float = 4.0,
    seed_value: int = 0,
    min_len: int = 4,
) -> list[tuple[list[int], int, float]]:
    """Online-serving-style workload: (prompt_token_ids, max_tokens, arrival_time_s).

    Matches vLLM's own benchmark_serving.py conventions (see
    https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py):
    lengths sampled from a Poisson distribution around ShareGPT-like means
    (~512 input / ~256 output tokens for real conversational traces) rather
    than uniform-random, and *arrival* staggered as a Poisson process at
    `request_rate` requests/sec (burstiness=1) rather than every request
    landing at t=0. `request_rate=float("inf")` reproduces the old
    submit-everything-immediately "offline batch" behavior as a special case.

    Arrival matters here specifically because it's the only way to exercise
    the actual scenario Config.step_mode="combined"/"unified" target: new
    prefill work showing up *while* other requests are already mid-decode.
    A single upfront burst has every request start prefilling together, so
    it can't distinguish "separate" from "combined"/"unified" scheduling.
    """
    rng = random.Random(seed_value)
    requests = []
    t = 0.0
    for _ in range(num_seqs):
        input_len = _poisson(rng, mean_input_len, min_len)
        output_len = _poisson(rng, mean_output_len, 1)
        prompt = [rng.randint(0, 10000) for _ in range(input_len)]
        requests.append((prompt, output_len, t))
        if request_rate != float("inf"):
            t += rng.expovariate(request_rate)  # Poisson process: exponential inter-arrival times
    return requests


def _poisson(rng: random.Random, mean: int, min_len: int) -> int:
    # Normal approximation to Poisson(mean) (valid for mean well above ~20,
    # true here at 256/512) -- exact Knuth sampling needs O(mean) draws per
    # call, needlessly slow for the few hundred-token means used here.
    if mean <= 0:
        return min_len
    return max(min_len, round(rng.gauss(mean, mean ** 0.5)))


# Multi-turn chat topics: (opening user question, short assistant reply, follow-up user question)
CHAT_SCENARIOS: list[tuple[str, str, str]] = [
    (
        "What is paged KV cache in LLM serving?",
        "Paged KV cache stores attention keys/values in fixed-size blocks so memory can be allocated non-contiguously.",
        "Why does that help continuous batching?",
    ),
    (
        "Explain prefill vs decode in inference.",
        "Prefill processes many prompt tokens at once; decode generates one new token per step per sequence.",
        "Which phase is usually memory-bandwidth bound?",
    ),
    (
        "What does a CUDA graph do for decode?",
        "It records a fixed GPU kernel launch sequence once, then replays it with new input buffers.",
        "When would you not use CUDA graphs?",
    ),
    (
        "How does prefix caching work?",
        "If two prompts share an identical token prefix, their KV blocks can be reused instead of recomputed.",
        "What metadata does attention need for prefix-cache prefill?",
    ),
    (
        "Compare tensor parallelism and pipeline parallelism.",
        "Tensor parallelism splits individual layers across GPUs; pipeline parallelism splits layers across stages.",
        "Which one adds all-reduce style communication in attention projections?",
    ),
    (
        "What is continuous batching?",
        "New requests join a running batch and finished ones leave without waiting for the whole batch to finish.",
        "How is that different from static batching in HF generate?",
    ),
    (
        "Why use FlashAttention instead of materializing full attention matrices?",
        "It fuses softmax with matmul and avoids storing the full N x N attention map in HBM.",
        "Does decode still benefit from FlashAttention?",
    ),
    (
        "What is slot_mapping used for?",
        "It maps each scheduled token to a physical slot inside the paged KV cache before attention runs.",
        "Is slot_mapping the same as block_tables?",
    ),
]


def make_chat_workload(
    num_seqs: int,
    min_output_len: int,
    max_output_len: int,
    seed_value: int = 0,
    chat_turns: int = 3,
) -> tuple[list[list[dict[str, str]]], list[int]]:
    """Build multi-turn chat message lists (before template).

    ``chat_turns`` is the number of completed user/assistant rounds before the
    final user message that triggers generation.
    """
    seed(seed_value)
    lo_o, hi_o = max(1, min_output_len), max(min_output_len, max_output_len)
    messages_list: list[list[dict[str, str]]] = []
    max_tokens: list[int] = []
    follow_ups = [
        "Can you give a concrete example?",
        "What should I read next?",
        "Summarize in one sentence.",
        "Can you elaborate more?",
    ]
    follow_replies = [
        "Sure. Imagine two requests sharing the first 512 prompt tokens.",
        "Start with the scheduler and KV block manager docs in this repo.",
        "Serving is mostly about batching policy plus KV memory layout.",
        "Happy to expand on any part of that.",
    ]

    rounds = max(1, chat_turns)
    for i in range(num_seqs):
        topic = CHAT_SCENARIOS[(i + seed_value) % len(CHAT_SCENARIOS)]
        user1, asst1, user_final = topic
        messages: list[dict[str, str]] = [
            {"role": "user", "content": user1},
            {"role": "assistant", "content": asst1},
        ]
        for r in range(1, rounds):
            messages.append({"role": "user", "content": follow_ups[(i + r) % len(follow_ups)]})
            messages.append({"role": "assistant", "content": follow_replies[(i + r) % len(follow_replies)]})
        messages.append(
            {
                "role": "user",
                "content": user_final if i % 2 == 0 else choice([user_final, follow_ups[i % len(follow_ups)]]),
            }
        )
        messages_list.append(messages)
        max_tokens.append(randint(lo_o, hi_o))

    return messages_list, max_tokens


def apply_chat_template(
    tokenizer,
    messages_list: list[list[dict[str, str]]],
) -> list[str]:
    prompts: list[str] = []
    for messages in messages_list:
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts
