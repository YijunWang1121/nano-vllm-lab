"""Shared synthetic workload for engine compare scripts."""

from __future__ import annotations

import os
from random import randint, seed


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
    seed(seed_value)
    lo_i, hi_i = max(1, min_input_len), max(min_input_len, max_input_len)
    lo_o, hi_o = max(1, min_output_len), max(min_output_len, max_output_len)
    prompts = [[randint(0, 10000) for _ in range(randint(lo_i, hi_i))] for _ in range(num_seqs)]
    max_tokens = [randint(lo_o, hi_o) for _ in range(num_seqs)]
    return prompts, max_tokens
