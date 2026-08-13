"""Unit tests for optional chunked-prefill-sharing (Config.chunk_prefill_tokens).

Disabled by default (chunk_prefill_tokens=-1): a long prefill sequence's
per-step budget saturates the entire step, so any sequence queued behind it
gets zero look-in until the long one finishes entirely -- this is the
existing (unchanged) behavior, also covered by
tests/milestones/test_02_scheduler_basic.py.

Enabled (chunk_prefill_tokens>0): a long sequence's per-step contribution is
capped below that, leaving budget in the *same* step for other waiting
sequences -- so a short prompt queued behind a long one no longer has to
wait for the long one to fully finish prefilling.
"""

from __future__ import annotations

from nanovllm.engine.sequence import SequenceStatus
from tests.helpers import make_scheduler, make_sequence

LONG_PROMPT = list(range(20))  # 20 tokens
SHORT_PROMPT = [100, 101, 102]  # 3 tokens


def test_disabled_by_default_long_seq_blocks_short_seq_for_the_whole_step():
    sched = make_scheduler(max_num_batched_tokens=8, block_size=4, num_kvcache_blocks=64)
    long_seq = make_sequence(LONG_PROMPT, block_size=4)
    short_seq = make_sequence(SHORT_PROMPT, block_size=4)
    sched.add(long_seq)
    sched.add(short_seq)

    seqs, is_prefill = sched.schedule()

    assert is_prefill
    assert seqs == [long_seq]
    assert long_seq.num_scheduled_tokens == 8  # capped only by the overall step budget
    assert long_seq.status == SequenceStatus.WAITING  # still has 12 tokens left
    # short_seq never even got examined: long_seq is still waiting[0] next step.
    assert list(sched.waiting) == [long_seq, short_seq]


def test_chunk_cap_lets_short_seq_share_the_same_step():
    sched = make_scheduler(
        max_num_batched_tokens=8, block_size=4, num_kvcache_blocks=64,
        chunk_prefill_tokens=4,
    )
    long_seq = make_sequence(LONG_PROMPT, block_size=4)
    short_seq = make_sequence(SHORT_PROMPT, block_size=4)
    sched.add(long_seq)
    sched.add(short_seq)

    seqs, is_prefill = sched.schedule()

    assert is_prefill
    assert seqs == [long_seq, short_seq]
    assert long_seq.num_scheduled_tokens == 4  # capped by chunk_prefill_tokens, not the step budget
    assert long_seq.status == SequenceStatus.WAITING  # still has 16 tokens left
    assert short_seq.num_scheduled_tokens == 3  # fits fully in the leftover 4 tokens of budget
    assert short_seq.status == SequenceStatus.RUNNING  # fully admitted this step
    assert list(sched.waiting) == [long_seq]  # only the unfinished one remains queued


def test_fifo_order_preserved_when_no_leftover_budget():
    # chunk cap == step budget, so every chunk saturates the step and there's
    # never leftover for short_seq to sneak into -- chunking bounds how much
    # of the budget one request can hog, it doesn't reorder the queue.
    sched = make_scheduler(
        max_num_batched_tokens=4, block_size=4, num_kvcache_blocks=64,
        chunk_prefill_tokens=4,
    )
    long_seq = make_sequence(LONG_PROMPT, block_size=4)  # 20 tokens, 5 steps @ 4/step
    short_seq = make_sequence(SHORT_PROMPT, block_size=4)
    sched.add(long_seq)
    sched.add(short_seq)

    seqs, is_prefill = sched.schedule()
    assert seqs == [long_seq]
    assert list(sched.waiting) == [long_seq, short_seq]
    sched.postprocess(seqs, [999], is_prefill=True)

    seqs, is_prefill = sched.schedule()
    assert seqs == [long_seq]
    assert list(sched.waiting) == [long_seq, short_seq]


def test_multi_step_chunking_eventually_completes_and_totals_match():
    sched = make_scheduler(
        max_num_batched_tokens=100, block_size=4, num_kvcache_blocks=64,
        chunk_prefill_tokens=4,
    )
    seq = make_sequence(list(range(10)), block_size=4, max_tokens=5)  # 10 prompt tokens
    sched.add(seq)

    scheduled_total = 0
    steps = 0
    while seq.num_cached_tokens < seq.num_prompt_tokens:
        seqs, is_prefill = sched.schedule()
        assert is_prefill
        assert seqs == [seq]
        assert seq.num_scheduled_tokens <= 4  # never exceeds the chunk cap
        scheduled_total += seq.num_scheduled_tokens
        sched.postprocess(seqs, [999], is_prefill=True)
        steps += 1
        assert steps <= 10  # safety valve in case of a scheduling bug

    assert scheduled_total == 10  # 4 + 4 + 2, matching the full prompt length
    assert steps == 3
    assert seq.status == SequenceStatus.RUNNING
    # The step that completes the prompt also samples the first token
    # (same as the classic scheduler) -- no extra decode step needed.
    assert seq.completion_token_ids == [999]


def test_max_num_seqs_still_respected_in_chunked_mode():
    sched = make_scheduler(
        max_num_seqs=1, max_num_batched_tokens=100, block_size=4,
        num_kvcache_blocks=64, chunk_prefill_tokens=4,
    )
    long_seq = make_sequence(LONG_PROMPT, block_size=4)
    short_seq = make_sequence(SHORT_PROMPT, block_size=4)
    sched.add(long_seq)
    sched.add(short_seq)

    seqs, is_prefill = sched.schedule()

    assert seqs == [long_seq]
    assert list(sched.waiting) == [long_seq, short_seq]


def test_kv_cache_exhaustion_still_stops_admission_for_the_round():
    # Only enough blocks for the long sequence: short_seq must wait even
    # though it arrived with token budget to spare. Chunked-prefill sharing
    # only relaxes the *token budget* restriction -- it doesn't change KV
    # admission order/semantics, matching the classic scheduler's "break".
    sched = make_scheduler(
        max_num_batched_tokens=100, block_size=4, num_kvcache_blocks=5,
        chunk_prefill_tokens=4,
    )
    long_seq = make_sequence(LONG_PROMPT, block_size=4)  # needs 5 blocks (20/4)
    short_seq = make_sequence(SHORT_PROMPT, block_size=4)  # needs 1 block
    sched.add(long_seq)
    sched.add(short_seq)

    seqs, is_prefill = sched.schedule()

    assert seqs == [long_seq]
    assert list(sched.waiting) == [long_seq, short_seq]
