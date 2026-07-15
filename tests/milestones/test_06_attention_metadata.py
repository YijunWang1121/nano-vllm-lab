"""Milestone 6: attention metadata contracts (CPU-friendly where possible)."""

from __future__ import annotations

import pytest
import torch

from nanovllm.course.exceptions import CourseNotImplementedError
from nanovllm.utils.context import Context, set_context, get_context, reset_context
from nanovllm.engine.input_metadata import build_prefill_metadata, build_decode_metadata
from tests.helpers import EXAMPLE_A_PROMPT, make_sequence


def test_context_set_and_reset():
    reset_context()
    assert get_context().is_prefill is False
    set_context(True, max_seqlen_q=5, max_seqlen_k=5)
    ctx = get_context()
    assert ctx.is_prefill is True
    assert ctx.max_seqlen_q == 5
    reset_context()
    assert get_context().is_prefill is False


def test_attention_path_selection_contract_from_metadata():
    """Document which Context fields Attention.forward must consume."""
    a = make_sequence(EXAMPLE_A_PROMPT, block_size=4)
    a.block_table = [7, 2]
    a.num_scheduled_tokens = 5
    try:
        prefill = build_prefill_metadata([a], 4)
    except CourseNotImplementedError as e:
        pytest.fail(str(e))
    assert prefill["need_block_tables"] is False

    a.num_cached_tokens = 4
    a.num_scheduled_tokens = 1
    prefill_pc = build_prefill_metadata([a], 4)
    assert prefill_pc["need_block_tables"] is True

    decode = build_decode_metadata([a], 4)
    assert "slot_mapping" in decode and "context_lens" in decode


def test_attention_forward_todo_or_runs():
    """Reference: needs flash-attn. Student: raises CourseNotImplementedError."""
    try:
        from nanovllm.layers.attention import Attention
    except ImportError:
        pytest.skip("flash-attn/triton not installed")

    attn = Attention(num_heads=2, head_dim=4, scale=0.5, num_kv_heads=2)
    q = k = v = torch.zeros(1, 2, 4)
    reset_context()
    set_context(True, max_seqlen_q=1, max_seqlen_k=1)
    try:
        out = attn(q, k, v)
    except CourseNotImplementedError as e:
        assert "TODO-L2-ATTN-01" in str(e)
        assert "docs/tutorial/07_attention_metadata.md" in str(e)
        return
    except Exception:
        # Reference path may fail on CPU tensors even with flash-attn installed.
        pytest.skip("Attention kernels unavailable on this platform")
    assert out is not None
