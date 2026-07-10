"""Milestone 1: Sequence state."""

from __future__ import annotations

import pytest

from nanovllm.course.exceptions import CourseNotImplementedError
from nanovllm.engine.sequence import Sequence, SequenceStatus
from tests.helpers import EXAMPLE_A_PROMPT, EXAMPLE_B_PROMPT, make_sequence


def _fail_todo(exc: CourseNotImplementedError):
    pytest.fail(str(exc))


def test_sequence_initialization():
    try:
        seq = make_sequence(EXAMPLE_A_PROMPT, block_size=4, max_tokens=8)
    except CourseNotImplementedError as e:
        _fail_todo(e)
    assert seq.status == SequenceStatus.WAITING
    assert seq.num_tokens == 5
    assert seq.num_prompt_tokens == 5
    assert seq.num_completion_tokens == 0
    assert seq.last_token == 14
    assert seq.num_cached_tokens == 0
    assert seq.block_table == []
    assert seq.is_prefill is True


def test_append_token_updates_state():
    seq = make_sequence(EXAMPLE_A_PROMPT, block_size=4, max_tokens=8)
    try:
        seq.append_token(99)
    except CourseNotImplementedError as e:
        _fail_todo(e)
    assert seq.token_ids[-1] == 99
    assert seq.last_token == 99
    assert seq.num_tokens == 6
    assert seq.num_completion_tokens == 1
    assert seq.completion_token_ids == [99]


def test_block_indexing_with_block_size_4():
    # A has 5 tokens, block_size=4 -> 2 logical blocks: [10,11,12,13] and [14]
    seq = make_sequence(EXAMPLE_A_PROMPT, block_size=4)
    try:
        n_blocks = seq.num_blocks
        last = seq.last_block_num_tokens
        b0 = seq.block(0)
        b1 = seq.block(1)
    except CourseNotImplementedError as e:
        _fail_todo(e)
    assert n_blocks == 2
    assert last == 1
    assert b0 == [10, 11, 12, 13]
    assert b1 == [14]


def test_request_b_single_block():
    seq = make_sequence(EXAMPLE_B_PROMPT, block_size=4)
    assert seq.num_blocks == 1
    assert seq.last_block_num_tokens == 3
    assert seq.block(0) == [20, 21, 22]


def test_is_finished_property():
    seq = make_sequence(EXAMPLE_A_PROMPT)
    assert seq.is_finished is False
    seq.status = SequenceStatus.FINISHED
    assert seq.is_finished is True
