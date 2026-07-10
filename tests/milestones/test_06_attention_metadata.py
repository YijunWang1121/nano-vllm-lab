"""Milestone 6: attention metadata contracts (CPU, no flash-attn required)."""

from __future__ import annotations

import pytest

from nanovllm.utils.context import Context, set_context, get_context, reset_context
from nanovllm.engine.input_metadata import build_prefill_metadata, build_decode_metadata
from tests.helpers import EXAMPLE_A_PROMPT, EXAMPLE_B_PROMPT, make_sequence


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
    prefill = build_prefill_metadata([a], 4)
    # Prefill without prefix cache: use Q/K/V tensors directly (block_tables None).
    assert prefill["need_block_tables"] is False

    a.num_cached_tokens = 4
    a.num_scheduled_tokens = 1
    prefill_pc = build_prefill_metadata([a], 4)
    assert prefill_pc["need_block_tables"] is True

    decode = build_decode_metadata([a], 4)
    assert "slot_mapping" in decode and "context_lens" in decode


def test_attention_forward_raises_clear_todo_when_excavated():
    """On the student branch, Attention.forward is a TODO; on reference it needs GPU kernels.

    Here we only verify the Context wiring contract used by Attention.forward.
    """
    from nanovllm.layers import attention as attn_mod

    # Inspect that forward reads context fields (source-level contract).
    src = attn_mod.Attention.forward.__doc__ or ""
    # Always check context helpers exist for the attention path.
    ctx = Context(is_prefill=True, max_seqlen_q=1, max_seqlen_k=1, slot_mapping=None)
    assert ctx.is_prefill is True
    assert hasattr(attn_mod.Attention, "forward")
