"""Milestone 3: prefill/decode scheduling, preempt, postprocess."""

from __future__ import annotations

import pytest

from nanovllm.course.exceptions import CourseNotImplementedError
from nanovllm.engine.sequence import SequenceStatus
from tests.helpers import EXAMPLE_A_PROMPT, EXAMPLE_B_PROMPT, make_scheduler, make_sequence


def _prefill_all(sched, seqs):
    for s in seqs:
        sched.add(s)
    scheduled, is_prefill = sched.schedule()
    assert is_prefill
    # Mark prefill complete for postprocess path.
    for s in scheduled:
        s.num_cached_tokens = 0  # will be updated in postprocess
    return scheduled


def test_decode_schedules_one_token_per_seq():
    sched = make_scheduler(block_size=4, num_kvcache_blocks=16)
    a = make_sequence(EXAMPLE_A_PROMPT, block_size=4, max_tokens=4)
    b = make_sequence(EXAMPLE_B_PROMPT, block_size=4, max_tokens=4)
    scheduled = _prefill_all(sched, [a, b])
    # Simulate completed prefill without sampling yet.
    for s in scheduled:
        s.num_cached_tokens = s.num_tokens
        s.num_scheduled_tokens = 0
    try:
        seqs, is_prefill = sched.schedule()
    except CourseNotImplementedError as e:
        pytest.fail(str(e))
    assert is_prefill is False
    assert set(seqs) == {a, b}
    assert all(s.num_scheduled_tokens == 1 for s in seqs)
    assert all(s.is_prefill is False for s in seqs)


def test_postprocess_appends_and_finishes_on_max_tokens():
    sched = make_scheduler(block_size=4, num_kvcache_blocks=16, eos=2)
    a = make_sequence(EXAMPLE_A_PROMPT, block_size=4, max_tokens=1)
    sched.add(a)
    seqs, is_prefill = sched.schedule()
    assert is_prefill
    try:
        sched.postprocess(seqs, [77], is_prefill=True)
    except CourseNotImplementedError as e:
        pytest.fail(str(e))
    assert a.completion_token_ids == [77]
    assert a.is_finished
    assert a.status == SequenceStatus.FINISHED
    assert a.block_table == []
    assert sched.is_finished()


def test_postprocess_finishes_on_eos():
    sched = make_scheduler(block_size=4, num_kvcache_blocks=16, eos=2)
    a = make_sequence(EXAMPLE_A_PROMPT, block_size=4, max_tokens=10)
    sched.add(a)
    seqs, is_prefill = sched.schedule()
    sched.postprocess(seqs, [2], is_prefill=True)
    assert a.is_finished
    assert a.completion_token_ids == [2]


def test_chunked_prefill_does_not_sample_until_prompt_done():
    sched = make_scheduler(max_num_batched_tokens=3, block_size=4, num_kvcache_blocks=16)
    a = make_sequence(EXAMPLE_A_PROMPT, block_size=4, max_tokens=4)  # 5 tokens
    sched.add(a)
    seqs, is_prefill = sched.schedule()
    assert is_prefill
    assert a.num_scheduled_tokens == 3
    assert a.status == SequenceStatus.WAITING  # still waiting for remaining prompt
    before = list(a.token_ids)
    sched.postprocess(seqs, [999], is_prefill=True)
    # Chunked prefill incomplete: do not append sampled token yet.
    assert a.token_ids == before
    assert a.num_cached_tokens == 3


def test_preempt_returns_seq_to_waiting_and_frees_blocks():
    sched = make_scheduler(block_size=4, num_kvcache_blocks=2)  # tiny pool
    # Fill memory with a long-ish sequence occupying both blocks.
    a = make_sequence(list(range(5)), block_size=4, max_tokens=8)  # needs 2 blocks
    sched.add(a)
    seqs, _ = sched.schedule()
    assert a.block_table
    try:
        sched.preempt(a)
    except CourseNotImplementedError as e:
        pytest.fail(str(e))
    assert a.status == SequenceStatus.WAITING
    assert a.is_prefill is True
    assert a.block_table == []
    assert a in sched.waiting
