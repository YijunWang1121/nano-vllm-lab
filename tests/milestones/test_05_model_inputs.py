"""Milestone 5: model input metadata builders (CPU)."""

from __future__ import annotations

import pytest

from nanovllm.course.exceptions import CourseNotImplementedError
from nanovllm.engine.input_metadata import (
    build_block_tables,
    build_prefill_metadata,
    build_decode_metadata,
)
from tests.helpers import EXAMPLE_A_PROMPT, EXAMPLE_B_PROMPT, make_sequence


def test_build_block_tables_padding():
    a = make_sequence(EXAMPLE_A_PROMPT, block_size=4)
    b = make_sequence(EXAMPLE_B_PROMPT, block_size=4)
    a.block_table = [7, 2]
    b.block_table = [5]
    try:
        tables = build_block_tables([a, b])
    except CourseNotImplementedError as e:
        pytest.fail(str(e))
    assert tables == [[7, 2], [5, -1]]


def test_prefill_metadata_two_requests():
    # A: 5 tokens, B: 3 tokens, block_size=4
    a = make_sequence(EXAMPLE_A_PROMPT, block_size=4)
    b = make_sequence(EXAMPLE_B_PROMPT, block_size=4)
    a.block_table = [7, 2]
    b.block_table = [5]
    a.num_scheduled_tokens = 5
    b.num_scheduled_tokens = 3
    a.num_cached_tokens = 0
    b.num_cached_tokens = 0
    try:
        meta = build_prefill_metadata([a, b], block_size=4)
    except CourseNotImplementedError as e:
        pytest.fail(str(e))
    assert meta["input_ids"] == EXAMPLE_A_PROMPT + EXAMPLE_B_PROMPT
    assert meta["positions"] == [0, 1, 2, 3, 4, 0, 1, 2]
    assert meta["cu_seqlens_q"] == [0, 5, 8]
    assert meta["cu_seqlens_k"] == [0, 5, 8]
    assert meta["max_seqlen_q"] == 5
    assert meta["max_seqlen_k"] == 5
    # A slots: block7 offsets 0..3, block2 offset 0
    # B slots: block5 offsets 0..2
    assert meta["slot_mapping"] == [
        7 * 4 + 0, 7 * 4 + 1, 7 * 4 + 2, 7 * 4 + 3, 2 * 4 + 0,
        5 * 4 + 0, 5 * 4 + 1, 5 * 4 + 2,
    ]
    assert meta["need_block_tables"] is False


def test_prefill_with_prefix_cache_sets_need_block_tables():
    a = make_sequence(EXAMPLE_A_PROMPT, block_size=4)
    a.block_table = [7, 2]
    a.num_cached_tokens = 4  # first block cached
    a.num_scheduled_tokens = 1  # only last prompt token
    meta = build_prefill_metadata([a], block_size=4)
    assert meta["input_ids"] == [14]
    assert meta["positions"] == [4]
    assert meta["cu_seqlens_q"] == [0, 1]
    assert meta["cu_seqlens_k"] == [0, 5]
    assert meta["need_block_tables"] is True
    assert meta["slot_mapping"] == [2 * 4 + 0]


def test_decode_metadata_slot_mapping():
    a = make_sequence(EXAMPLE_A_PROMPT, block_size=4)  # len=5
    b = make_sequence(EXAMPLE_B_PROMPT, block_size=4)  # len=3
    a.block_table = [7, 2]
    b.block_table = [5]
    # Decode writes into current last slot (last token position).
    try:
        meta = build_decode_metadata([a, b], block_size=4)
    except CourseNotImplementedError as e:
        pytest.fail(str(e))
    assert meta["input_ids"] == [14, 22]
    assert meta["positions"] == [4, 2]
    assert meta["context_lens"] == [5, 3]
    # A: last_block_num_tokens=1 -> slot = 2*4 + 1 - 1 = 8
    # B: last_block_num_tokens=3 -> slot = 5*4 + 3 - 1 = 22
    assert meta["slot_mapping"] == [8, 22]
