"""Unit tests for optional CPU KV-cache swap (Config.kv_swap_enabled).

Disabled by default: Scheduler.preempt() behaves exactly like the classic
recompute-only path (unchanged) -- see tests/milestones/test_03_prefill_decode.py.

Enabled: on preemption, a sequence's private (non-shared) GPU blocks are
copied to a CPU-side pool instead of being discarded, and restored on
re-admission -- skipping the prefill recompute that would otherwise be
needed to regenerate that KV data.

These tests exercise the scheduler-side bookkeeping only (block ids move
between GPU/CPU free lists; no real tensor data exists at this level).
Real D2H/H2D copy correctness needs a GPU test -- see tests/gpu/.
"""

from __future__ import annotations

from nanovllm.engine.sequence import SequenceStatus
from tests.helpers import make_scheduler, make_sequence
from tests.oracles.fake_model_runner import FakeModelRunner

LONG_PROMPT = list(range(4))
OTHER_PROMPT = list(range(4, 8))


def _running_pair(block_size=4, num_kvcache_blocks=4, **kwargs):
    sched = make_scheduler(
        max_num_batched_tokens=100, block_size=block_size, num_kvcache_blocks=num_kvcache_blocks,
        use_real_block_manager=True, **kwargs,
    )
    a = make_sequence(LONG_PROMPT, block_size=block_size)
    b = make_sequence(OTHER_PROMPT, block_size=block_size)
    sched.add(a)
    sched.add(b)
    seqs, is_prefill = sched.schedule()
    sched.postprocess(seqs, [900, 901], is_prefill=True)
    return sched, a, b


def _preempt(sched, seq):
    # Scheduler.preempt() assumes the caller already popped `seq` out of
    # self.running (every real call site does: `preempt(self.running.pop())`
    # / `preempt(seq)` right after `seq = self.running.popleft()`). Mirror
    # that here so direct unit-test calls don't leave seq in both queues.
    if seq in sched.running:
        sched.running.remove(seq)
    sched.preempt(seq)


def test_disabled_by_default_preempt_is_unchanged():
    sched, a, b = _running_pair(num_kvcache_blocks=2)  # exactly enough for a+b's first block each
    _preempt(sched, b)
    assert b.status == SequenceStatus.WAITING
    assert b.is_prefill is True
    assert b.block_table == []
    assert b.swap_state == "none"
    assert sched.num_swaps == 0


def test_preempt_swaps_out_private_blocks_when_enabled():
    sched, a, b = _running_pair(num_kvcache_blocks=2, kv_swap_enabled=True, num_cpu_kvcache_blocks=8)
    _preempt(sched, b)

    assert b.status == SequenceStatus.WAITING
    assert b.swap_state == "swapped"
    assert b.block_table == []  # GPU freed
    assert b.cpu_block_table == [0]  # moved to CPU pool
    assert b.num_swapped_tokens == 4  # num_cached_tokens at the moment of preemption
    assert sched.num_swaps == 1
    assert sched.pending_swap_outs == [([1], [0])]  # (gpu_block_ids, cpu_block_ids), captured before dealloc


def test_should_swap_declines_when_blocks_are_shared():
    # Give a and b an identical prefix long enough to hash-share a block,
    # then preempt the one still holding a live reference to it.
    block_size = 4
    sched = make_scheduler(
        max_num_batched_tokens=100, block_size=block_size, num_kvcache_blocks=4,
        kv_swap_enabled=True, num_cpu_kvcache_blocks=8, use_real_block_manager=True,
    )
    shared_prefix = [1, 2, 3, 4]
    a = make_sequence(shared_prefix + [10], block_size=block_size)
    b = make_sequence(shared_prefix + [20], block_size=block_size)
    sched.add(a)
    seqs, _ = sched.schedule()
    sched.postprocess(seqs, [900], is_prefill=True)  # a's first block gets hashed
    sched.add(b)
    seqs, _ = sched.schedule()
    sched.postprocess(seqs, [901], is_prefill=True)  # b cache-hits a's shared first block

    assert sched.block_manager.blocks[a.block_table[0]].ref_count == 2  # confirms sharing actually happened

    _preempt(sched, b)

    assert b.swap_state == "none"  # fell back to recompute: can't swap a shared block
    assert b.block_table == []
    assert sched.num_swaps == 0
    assert sched.num_preemptions == 1


def test_should_swap_declines_below_min_tokens_threshold():
    sched, a, b = _running_pair(
        num_kvcache_blocks=2, kv_swap_enabled=True, num_cpu_kvcache_blocks=8, kv_swap_min_tokens=100,
    )
    _preempt(sched, b)
    assert b.swap_state == "none"
    assert sched.num_swaps == 0


def test_should_swap_declines_when_cpu_pool_exhausted():
    sched, a, b = _running_pair(num_kvcache_blocks=2, kv_swap_enabled=True, num_cpu_kvcache_blocks=0)
    _preempt(sched, b)
    assert b.swap_state == "none"  # nowhere to put it, falls back to recompute
    assert sched.num_swaps == 0
    assert sched.num_preemptions == 1


def test_should_swap_declines_when_sequence_needs_the_whole_gpu_pool():
    # b alone occupies every GPU block: swapping it out just to immediately
    # need it all back (and then some) on its very next decode turn can't
    # ever make progress. Refuse up front rather than swap-cycle forever.
    sched = make_scheduler(
        max_num_batched_tokens=100, block_size=4, num_kvcache_blocks=1,
        kv_swap_enabled=True, num_cpu_kvcache_blocks=8, use_real_block_manager=True,
    )
    b = make_sequence(LONG_PROMPT, block_size=4)
    sched.add(b)
    seqs, _ = sched.schedule()
    sched.postprocess(seqs, [900], is_prefill=True)

    _preempt(sched, b)

    assert b.swap_state == "none"
    assert sched.num_swaps == 0


def test_swap_in_restores_state_and_skips_straight_to_running():
    sched, a, b = _running_pair(num_kvcache_blocks=2, kv_swap_enabled=True, num_cpu_kvcache_blocks=8)
    _preempt(sched, b)
    sched.drain_pending_swap_outs()
    # Free a's GPU block directly (standing in for it finishing normally)
    # so b's swap-in has room -- isolates swap-in behavior from decode.
    sched.block_manager.deallocate(a)
    sched.running.remove(a)

    seqs, is_prefill = sched.schedule()

    assert b.status == SequenceStatus.RUNNING
    assert b.swap_state == "none"
    assert b.cpu_block_table == []
    assert b.num_cached_tokens == 4  # restored from num_swapped_tokens
    assert b in sched.running
    assert sched.pending_swap_ins == [([0], [0])] or sched.pending_swap_ins  # some (cpu_ids, gpu_ids) pair queued
    # Crucially: it must NOT have gone through the prefill path -- that's
    # what would otherwise force a real (fake) prefill forward pass. When
    # scheduler is falling through to decode instead, `seqs` here is
    # whatever _schedule_decode admitted, not b re-entering as is_prefill=True.
    assert is_prefill is False


def test_swap_and_recompute_produce_identical_results():
    # Same contended workload run twice (swap enabled vs. disabled): both
    # must converge to bit-identical completion sequences and leave no
    # leaked GPU/CPU blocks, proving swap is a pure optimization, not an
    # observable behavior change.
    def run(kv_swap_enabled):
        import itertools
        from nanovllm.engine.sequence import Sequence
        Sequence.counter = itertools.count()  # comparable seq_ids across the two runs
        sched = make_scheduler(
            max_num_batched_tokens=100, block_size=4, num_kvcache_blocks=8,
            kv_swap_enabled=kv_swap_enabled, num_cpu_kvcache_blocks=16,
            use_real_block_manager=True,
        )
        seqs = [make_sequence(list(range(i, i + 4)), block_size=4, max_tokens=10) for i in range(0, 16, 4)]
        for s in seqs:
            sched.add(s)
        runner = FakeModelRunner(next_token_fn=lambda seq, is_prefill: 1000 + seq.seq_id * 100 + seq.num_completion_tokens)

        steps = 0
        while not sched.is_finished() and steps < 200:
            scheduled, is_prefill = sched.schedule()
            token_ids = runner.run(scheduled, is_prefill)
            sched.postprocess(scheduled, token_ids, is_prefill)
            sched.drain_pending_swap_outs()
            sched.drain_pending_swap_ins()
            steps += 1
        assert steps < 200

        return {
            "results": {s.seq_id: list(s.completion_token_ids) for s in seqs},
            "all_finished": all(s.is_finished for s in seqs),
            "leaked_gpu_blocks": len(sched.block_manager.used_block_ids),
            "leaked_cpu_blocks": 16 - len(sched.cpu_block_manager.free_block_ids) if sched.cpu_block_manager else 0,
        }

    swapped = run(kv_swap_enabled=True)
    recomputed = run(kv_swap_enabled=False)

    assert swapped["all_finished"] and recomputed["all_finished"]
    assert swapped["leaked_gpu_blocks"] == 0
    assert swapped["leaked_cpu_blocks"] == 0
    assert recomputed["leaked_gpu_blocks"] == 0
    assert swapped["results"] == recomputed["results"]
