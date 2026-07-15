"""Milestone 10: CUDA graph path (GPU)."""

from __future__ import annotations

import gc
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


def test_cudagraph_generation(model_path):
    from nanovllm import LLM, SamplingParams
    import torch

    llm = LLM(model_path, enforce_eager=False, tensor_parallel_size=1)
    try:
        assert hasattr(llm.model_runner, "graphs")
        assert llm.model_runner.graphs
        sp = SamplingParams(temperature=0.8, max_tokens=4)
        outs = llm.generate(["Hello"], sp, use_tqdm=False)
        assert len(outs[0]["token_ids"]) <= 4
    finally:
        llm.exit()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
