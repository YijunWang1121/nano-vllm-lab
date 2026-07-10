"""Milestone 8: end-to-end continuous batching with a fake model runner."""

from __future__ import annotations

import pytest

from nanovllm.engine.sequence import SequenceStatus
from tests.helpers import EXAMPLE_A_PROMPT, EXAMPLE_B_PROMPT, make_scheduler, make_sequence
from tests.oracles.fake_model_runner import FakeModelRunner


def test_two_request_continuous_batching_fake_engine():
    sched = make_scheduler(block_size=4, num_kvcache_blocks=32, eos=2)
    a = make_sequence(EXAMPLE_A_PROMPT, block_size=4, max_tokens=2)
    b = make_sequence(EXAMPLE_B_PROMPT, block_size=4, max_tokens=2)
    sched.add(a)
    sched.add(b)
    runner = FakeModelRunner(next_token_fn=lambda seq, is_prefill: 100 + seq.num_completion_tokens)

    finished = {}
    steps = 0
    while not sched.is_finished() and steps < 20:
        seqs, is_prefill = sched.schedule()
        token_ids = runner.run(seqs, is_prefill)
        sched.postprocess(seqs, token_ids, is_prefill)
        for seq in seqs:
            if seq.is_finished:
                finished[seq.seq_id] = list(seq.completion_token_ids)
        steps += 1

    assert a.is_finished and b.is_finished
    assert len(a.completion_token_ids) == 2
    assert len(b.completion_token_ids) == 2
    assert a.block_table == [] and b.block_table == []
    assert sched.is_finished()


def test_kv_cache_cleanup_after_finish():
    sched = make_scheduler(block_size=4, num_kvcache_blocks=8, eos=2)
    a = make_sequence(EXAMPLE_A_PROMPT, block_size=4, max_tokens=1)
    sched.add(a)
    free_before = len(sched.block_manager.free_block_ids)
    seqs, is_prefill = sched.schedule()
    free_mid = len(sched.block_manager.free_block_ids)
    assert free_mid < free_before
    runner = FakeModelRunner(next_token_fn=lambda seq, is_prefill: 42)
    token_ids = runner.run(seqs, is_prefill)
    sched.postprocess(seqs, token_ids, is_prefill)
    assert a.is_finished
    assert len(sched.block_manager.free_block_ids) == free_before
