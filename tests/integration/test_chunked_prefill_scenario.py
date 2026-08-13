"""Scenario test: chunked-prefill-sharing cuts short-request latency behind
a long prefill, using the full continuous-batching loop (Scheduler +
FakeModelRunner), not just Scheduler.schedule() in isolation.

Workload: one long prompt arrives, immediately followed by a short prompt.
The metric is "steps until the short prompt is admitted" (time-to-first-token
proxy) -- each engine step is one unit of latency regardless of how long the
real model's forward pass takes, so this isolates the *scheduling policy's*
contribution to short-request latency from model compute time.
"""

from __future__ import annotations

from nanovllm.engine.sequence import SequenceStatus
from tests.helpers import make_scheduler, make_sequence
from tests.oracles.fake_model_runner import FakeModelRunner

LONG_PROMPT = list(range(400))
SHORT_PROMPT = list(range(500, 510))  # 10 tokens, queued right behind the long one


def _run_until(sched, runner, stop_condition, max_steps=200):
    steps = 0
    while not stop_condition() and steps < max_steps:
        seqs, is_prefill = sched.schedule()
        token_ids = runner.run(seqs, is_prefill)
        sched.postprocess(seqs, token_ids, is_prefill)
        steps += 1
    assert steps < max_steps, "scenario did not converge -- possible scheduling bug"
    return steps


def _steps_until_short_seq_admitted(chunk_prefill_tokens: int) -> int:
    sched = make_scheduler(
        max_num_batched_tokens=64, block_size=4, num_kvcache_blocks=256,
        chunk_prefill_tokens=chunk_prefill_tokens,
    )
    long_seq = make_sequence(LONG_PROMPT, block_size=4, max_tokens=1)
    short_seq = make_sequence(SHORT_PROMPT, block_size=4, max_tokens=1)
    sched.add(long_seq)
    sched.add(short_seq)
    runner = FakeModelRunner()
    return _run_until(sched, runner, lambda: short_seq.status != SequenceStatus.WAITING)


def test_chunked_prefill_sharing_cuts_short_request_latency():
    steps_classic = _steps_until_short_seq_admitted(chunk_prefill_tokens=-1)
    steps_chunked = _steps_until_short_seq_admitted(chunk_prefill_tokens=16)

    # Classic: short_seq is stuck behind the *entire* long prefill --
    # ceil(400 / 64) == 7 steps of pure long-prompt chunks before there's
    # any leftover budget for it.
    assert steps_classic == 7
    # Chunked: long_seq's first chunk is capped at 16 tokens, leaving
    # 64 - 16 == 48 tokens of budget in that very first step -- easily
    # covering the 10-token short prompt.
    assert steps_chunked == 1
    assert steps_chunked < steps_classic


def test_chunked_prefill_sharing_does_not_delay_the_long_request():
    # The long request should finish in the same number of steps whether or
    # not a short request is sharing its budget -- interleaving a small
    # request shouldn't measurably slow the big one down when there's spare
    # per-step budget for it.
    def run(with_short_seq: bool) -> int:
        sched = make_scheduler(
            max_num_batched_tokens=64, block_size=4, num_kvcache_blocks=256,
            chunk_prefill_tokens=16,
        )
        long_seq = make_sequence(LONG_PROMPT, block_size=4, max_tokens=1)
        sched.add(long_seq)
        if with_short_seq:
            sched.add(make_sequence(SHORT_PROMPT, block_size=4, max_tokens=1))
        runner = FakeModelRunner()
        steps = _run_until(sched, runner, sched.is_finished)
        assert long_seq.is_finished
        return steps

    assert run(with_short_seq=False) == run(with_short_seq=True) == 25
