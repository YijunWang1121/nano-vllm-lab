"""CPU unit tests for Config ablation switches."""

from __future__ import annotations

import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import SequenceStatus
from tests.helpers import EXAMPLE_A_PROMPT, EXAMPLE_B_PROMPT, make_scheduler_config, make_sequence


def test_prefix_caching_off_never_hits():
    bm = BlockManager(num_blocks=32, block_size=4, enable_prefix_caching=True)
    a = make_sequence(EXAMPLE_A_PROMPT, block_size=4)
    n = bm.can_allocate(a)
    assert n >= 0
    bm.allocate(a, n)
    a.num_scheduled_tokens = a.num_tokens
    bm.hash_blocks(a)

    b = make_sequence(EXAMPLE_A_PROMPT + EXAMPLE_B_PROMPT, block_size=4)
    hits_on = bm.can_allocate(b)
    assert hits_on > 0

    bm_off = BlockManager(num_blocks=32, block_size=4, enable_prefix_caching=False)
    c = make_sequence(EXAMPLE_A_PROMPT, block_size=4)
    assert bm_off.can_allocate(c) == 0
    bm_off.allocate(c, 0)
    c.num_scheduled_tokens = c.num_tokens
    bm_off.hash_blocks(c)  # no-op when disabled
    d = make_sequence(EXAMPLE_A_PROMPT + EXAMPLE_B_PROMPT, block_size=4)
    assert bm_off.can_allocate(d) == 0


def test_preemption_disabled_raises_when_kv_full():
    config = make_scheduler_config(num_kvcache_blocks=2, block_size=4, enable_preemption=False)
    sched = Scheduler(config)
    seq = make_sequence([1, 2, 3, 4], block_size=4)
    n = sched.block_manager.can_allocate(seq)
    sched.block_manager.allocate(seq, n)
    seq.status = SequenceStatus.RUNNING
    seq.num_cached_tokens = seq.num_tokens
    seq.append_token(99)  # len % block_size == 1 → needs a new block
    sched.running.append(seq)
    sched.block_manager.free_block_ids.clear()
    with pytest.raises(RuntimeError, match="enable_preemption=False"):
        sched.schedule()


def test_chunked_prefill_disabled_refuses_partial():
    config_off = make_scheduler_config(
        num_kvcache_blocks=64,
        block_size=4,
        max_num_batched_tokens=4,
        enable_chunked_prefill=False,
    )
    sched_off = Scheduler(config_off)
    long = make_sequence(list(range(20)), block_size=4)
    sched_off.add(long)
    scheduled_off, _ = sched_off._schedule_prefill()
    assert scheduled_off == []

    config_on = make_scheduler_config(
        num_kvcache_blocks=64,
        block_size=4,
        max_num_batched_tokens=4,
        enable_chunked_prefill=True,
    )
    sched_on = Scheduler(config_on)
    long2 = make_sequence(list(range(20)), block_size=4)
    sched_on.add(long2)
    scheduled_on, is_prefill = sched_on._schedule_prefill()
    assert is_prefill and scheduled_on
    assert scheduled_on[0].num_scheduled_tokens == 4
