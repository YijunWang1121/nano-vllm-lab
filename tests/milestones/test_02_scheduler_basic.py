"""Milestone 2: basic scheduler queues and prefill admission."""

from __future__ import annotations

import pytest

from nanovllm.course.exceptions import CourseNotImplementedError
from nanovllm.engine.sequence import SequenceStatus
from tests.helpers import EXAMPLE_A_PROMPT, EXAMPLE_B_PROMPT, make_scheduler, make_sequence


def test_add_places_sequence_in_waiting():
    sched = make_scheduler()
    seq = make_sequence(EXAMPLE_A_PROMPT)
    sched.add(seq)
    assert list(sched.waiting) == [seq]
    assert list(sched.running) == []
    assert not sched.is_finished()


def test_prefill_admits_single_request():
    sched = make_scheduler(block_size=4, num_kvcache_blocks=16)
    seq = make_sequence(EXAMPLE_A_PROMPT, block_size=4)
    sched.add(seq)
    try:
        seqs, is_prefill = sched.schedule()
    except CourseNotImplementedError as e:
        pytest.fail(str(e))
    assert is_prefill is True
    assert seqs == [seq]
    assert seq.status == SequenceStatus.RUNNING
    assert seq.num_scheduled_tokens == 5
    assert seq.block_table  # allocated
    assert not sched.waiting
    assert list(sched.running) == [seq]


def test_prefill_two_concurrent_requests():
    # Tutorial running example: A len=5, B len=3, block_size=4
    sched = make_scheduler(block_size=4, num_kvcache_blocks=16, max_num_batched_tokens=64)
    a = make_sequence(EXAMPLE_A_PROMPT, block_size=4)
    b = make_sequence(EXAMPLE_B_PROMPT, block_size=4)
    sched.add(a)
    sched.add(b)
    try:
        seqs, is_prefill = sched.schedule()
    except CourseNotImplementedError as e:
        pytest.fail(str(e))
    assert is_prefill is True
    assert seqs == [a, b]
    assert a.num_scheduled_tokens == 5
    assert b.num_scheduled_tokens == 3
    assert a.status == SequenceStatus.RUNNING
    assert b.status == SequenceStatus.RUNNING


def test_token_budget_limits_prefill_batch():
    sched = make_scheduler(max_num_batched_tokens=5, block_size=4, num_kvcache_blocks=16)
    a = make_sequence(EXAMPLE_A_PROMPT, block_size=4)  # 5 tokens
    b = make_sequence(EXAMPLE_B_PROMPT, block_size=4)  # 3 tokens
    sched.add(a)
    sched.add(b)
    seqs, is_prefill = sched.schedule()
    assert is_prefill
    # First seq can take the full budget; second must wait.
    assert seqs == [a]
    assert a.num_scheduled_tokens == 5
    assert list(sched.waiting) == [b]


def test_max_num_seqs_limits_admission():
    sched = make_scheduler(max_num_seqs=1, block_size=4, num_kvcache_blocks=16)
    a = make_sequence(EXAMPLE_A_PROMPT, block_size=4)
    b = make_sequence(EXAMPLE_B_PROMPT, block_size=4)
    sched.add(a)
    sched.add(b)
    seqs, _ = sched.schedule()
    assert seqs == [a]
    assert list(sched.waiting) == [b]
