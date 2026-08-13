"""Regression tests for Scheduler.add()'s admission checks (Gap 1 of the
GPU-memory-exhaustion remediation work): a request that can never be served
by this engine's configured limits must be rejected synchronously at
add()-time, not silently admitted into `waiting` where it would sit forever
(BlockManager.can_allocate returns -1 forever, Scheduler.is_finished() never
becomes True).

Two independent, separately-triggerable checks:
  1. seq.num_prompt_tokens > Config.max_model_len (a static config bound).
  2. seq.num_blocks > the KV-cache pool's total block count (this run's
     actual, GPU-measured capacity -- can be smaller than max_model_len
     allows on a small GPU / aggressive concurrency settings).
"""

from __future__ import annotations

import pytest

from nanovllm.exceptions import RequestTooLargeError
from tests.helpers import make_scheduler, make_sequence


def test_add_rejects_prompt_longer_than_max_model_len():
    sched = make_scheduler(block_size=4, num_kvcache_blocks=32, max_model_len=8)
    seq = make_sequence(list(range(10)), block_size=4)  # 10 > max_model_len=8
    with pytest.raises(RequestTooLargeError):
        sched.add(seq)
    assert seq not in sched.waiting


def test_add_rejects_prompt_larger_than_kvcache_pool():
    # block_size=4, num_kvcache_blocks=2 -> pool can hold at most 8 tokens
    # total, regardless of max_model_len being generous.
    sched = make_scheduler(block_size=4, num_kvcache_blocks=2, max_model_len=1000)
    seq = make_sequence(list(range(10)), block_size=4)  # needs 3 blocks, pool has 2
    with pytest.raises(RequestTooLargeError):
        sched.add(seq)
    assert seq not in sched.waiting


def test_add_accepts_normal_sized_request():
    sched = make_scheduler(block_size=4, num_kvcache_blocks=32, max_model_len=1000)
    seq = make_sequence(list(range(5)), block_size=4)
    sched.add(seq)  # must not raise
    assert seq in sched.waiting


def test_add_rejects_using_real_block_manager_too():
    # Same checks must hold against the real BlockManager, not just the
    # WorkingBlockManager test oracle (make_scheduler's default).
    sched = make_scheduler(
        block_size=4, num_kvcache_blocks=2, max_model_len=1000, use_real_block_manager=True,
    )
    seq = make_sequence(list(range(10)), block_size=4)
    with pytest.raises(RequestTooLargeError):
        sched.add(seq)
