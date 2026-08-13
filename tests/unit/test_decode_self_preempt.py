"""Regression tests for Scheduler._schedule_decode's self-preemption branch.

Two pre-existing bugs (found while stress-testing KV-cache swap, but present
regardless of Config.kv_swap_enabled -- these are general scheduler
correctness bugs, unrelated to swap):

1. `return scheduled,` was a 1-tuple instead of `(scheduled, False)`, so
   `seqs, is_prefill = scheduler.schedule()` raised ValueError whenever the
   last running sequence had to preempt itself (no other sequence left to
   evict, but it still needs a block the pool doesn't have).
2. That early return skipped `self.running.extendleft(reversed(scheduled))`
   entirely, so any sequences already popped into `scheduled` earlier in
   the same call were silently dropped from self.running forever --
   correctly returned and processed for this one step, but invisible to
   every scheduling decision after that (still SequenceStatus.RUNNING, but
   tracked in neither self.running nor self.waiting).

Reproducing those two needs enough concurrent growing sequences to exhaust a
modest pool such that, within a single _schedule_decode call, an earlier
sequence succeeds before a later one is forced to preempt itself.

A third, separate case: a *single* sequence alone in a pool it already fills
entirely used to just get stuck preempting-and-retrying itself forever (no
other sequence exists to ever free room). Scheduler.abort() (see
_schedule_decode/_schedule_unified's `elif len(seq.block_table) >=
self._pool_capacity()` branch) now detects this as provably unrecoverable
and aborts the sequence instead of looping.
"""

from __future__ import annotations

from nanovllm.engine.sequence import SequenceStatus
from tests.helpers import make_scheduler, make_sequence
from tests.oracles.fake_model_runner import FakeModelRunner


def _make_four_growing_sequences():
    return [make_sequence(list(range(i, i + 4)), block_size=4, max_tokens=10) for i in range(0, 16, 4)]


def test_self_preempt_end_to_end_does_not_orphan_or_crash():
    # 4 sequences competing for an 8-block pool: each needs up to 3 blocks
    # to reach max_tokens=10, so growth genuinely forces preemption
    # (confirmed empirically: 2 preemption events over the run), including
    # at least one self-preemption within a single _schedule_decode call.
    sched = make_scheduler(block_size=4, num_kvcache_blocks=8, max_num_batched_tokens=100, eos=999)
    seqs = _make_four_growing_sequences()
    for s in seqs:
        sched.add(s)
    runner = FakeModelRunner(next_token_fn=lambda seq, is_prefill: 100 + seq.num_completion_tokens)

    steps = 0
    while not sched.is_finished() and steps < 100:
        scheduled, is_prefill = sched.schedule()  # must not raise ValueError unpacking a 1-tuple
        token_ids = runner.run(scheduled, is_prefill)
        sched.postprocess(scheduled, token_ids, is_prefill)
        steps += 1
        # No RUNNING sequence may go untracked by both queues (the orphan bug).
        tracked = {s.seq_id for s in sched.running} | {s.seq_id for s in sched.waiting}
        for s in seqs:
            if s.status == SequenceStatus.RUNNING:
                assert s.seq_id in tracked, f"seq {s.seq_id} is RUNNING but orphaned at step {steps}"

    assert steps < 100, "did not converge"
    assert sched.num_preemptions > 0, "test scenario didn't actually exercise preemption"
    assert all(s.is_finished for s in seqs)
    assert all(len(s.completion_token_ids) == 10 for s in seqs)


def _lone_sequence_filling_the_pool():
    # block_size=4, num_kvcache_blocks=2 -> pool holds exactly 8 tokens; an
    # 8-token prompt fills it completely, so the very next decode-phase
    # token (crossing into block 3) can never be satisfied by anyone.
    sched = make_scheduler(
        block_size=4, num_kvcache_blocks=2, max_num_batched_tokens=100,
        max_model_len=1000, eos=999,
    )
    seq = make_sequence(list(range(8)), block_size=4, max_tokens=20)
    sched.add(seq)
    return sched, seq


def test_self_preempt_lone_sequence_aborts_instead_of_hanging_forever():
    sched, seq = _lone_sequence_filling_the_pool()
    runner = FakeModelRunner(next_token_fn=lambda seq, is_prefill: 1)  # never matches eos=999

    steps = 0
    while not sched.is_finished() and steps < 20:
        scheduled, is_prefill = sched.schedule()
        if scheduled:
            token_ids = runner.run(scheduled, is_prefill)
            sched.postprocess(scheduled, token_ids, is_prefill)
        steps += 1

    assert steps < 20, "did not converge -- looks like the old infinite self-preempt loop"
    assert seq.status == SequenceStatus.FINISHED
    assert seq.aborted is True
    assert seq in sched.drain_pending_aborts()
    assert len(sched.block_manager.free_block_ids) == 2, "aborted seq's blocks must be returned to the pool"
    assert sched.is_finished()


def test_unified_self_preempt_lone_sequence_aborts_instead_of_hanging_forever():
    # _schedule_unified's decode-admission loop is a separate implementation
    # of the same self-preempt pattern and needs the identical guard.
    sched, seq = _lone_sequence_filling_the_pool()
    runner = FakeModelRunner(next_token_fn=lambda seq, is_prefill: 1)

    steps = 0
    while not sched.is_finished() and steps < 20:
        scheduled = sched._schedule_unified()
        if scheduled:
            token_ids = runner.run(scheduled, True)
            sched.postprocess(scheduled, token_ids, True)
        steps += 1

    assert steps < 20, "did not converge -- looks like the old infinite self-preempt loop"
    assert seq.status == SequenceStatus.FINISHED
    assert seq.aborted is True
    assert seq in sched.drain_pending_aborts()
    assert len(sched.block_manager.free_block_ids) == 2
    assert sched.is_finished()


def test_abort_marks_finished_frees_blocks_and_untracks_from_both_queues():
    sched, seq = _lone_sequence_filling_the_pool()
    scheduled, is_prefill = sched.schedule()  # admits + fully prefills seq -> RUNNING
    assert scheduled == [seq] and is_prefill
    sched.postprocess(scheduled, [1], is_prefill)
    assert seq in sched.running

    sched.abort(seq, reason="test")

    assert seq.status == SequenceStatus.FINISHED
    assert seq.aborted is True
    assert seq not in sched.running
    assert seq not in sched.waiting
    assert len(sched.block_manager.free_block_ids) == 2
    assert sched.drain_pending_aborts() == [seq]
    assert sched.drain_pending_aborts() == []  # drained, not re-returned


def test_abort_batch_aborts_every_sequence_in_the_batch():
    sched = make_scheduler(block_size=4, num_kvcache_blocks=8, max_model_len=1000, eos=999)
    seqs = [make_sequence(list(range(i, i + 4)), block_size=4, max_tokens=20) for i in range(0, 8, 4)]
    for s in seqs:
        sched.add(s)
    scheduled, is_prefill = sched.schedule()
    assert set(scheduled) == set(seqs)
    sched.postprocess(scheduled, [1, 2], is_prefill)

    sched.abort_batch(seqs, reason="test_batch")

    assert all(s.status == SequenceStatus.FINISHED and s.aborted for s in seqs)
    assert len(sched.running) == 0
    assert sched.drain_pending_aborts() == seqs
