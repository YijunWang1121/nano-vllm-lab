"""Milestone 9: eager GPU end-to-end (skipped without CUDA / model)."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.gpu


def _model_path() -> str | None:
    path = os.path.expanduser(os.environ.get("NANOVLLM_TEST_MODEL", "~/huggingface/Qwen3-0.6B/"))
    return path if os.path.isdir(path) else None


@pytest.fixture
def model_path(require_cuda):
    path = _model_path()
    if path is None:
        pytest.skip("Set NANOVLLM_TEST_MODEL to a local HF model directory")
    return path


def test_greedyish_short_generation(model_path):
    from nanovllm import LLM, SamplingParams

    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
    sp = SamplingParams(temperature=0.01, max_tokens=8)
    outs = llm.generate(["Hello"], sp, use_tqdm=False)
    assert "text" in outs[0]
    assert len(outs[0]["token_ids"]) <= 8


def test_two_request_batch(model_path):
    from nanovllm import LLM, SamplingParams

    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
    sp = SamplingParams(temperature=0.8, max_tokens=4)
    outs = llm.generate(["Hi", "Hey"], sp, use_tqdm=False)
    assert len(outs) == 2
