"""Integration-style CPU test using fake runner."""

from tests.helpers import make_scheduler, make_sequence
from tests.oracles.fake_model_runner import FakeModelRunner


def test_submit_schedule_append_complete():
    sched = make_scheduler(block_size=4, num_kvcache_blocks=16, eos=999)
    seq = make_sequence([1, 2, 3, 4], block_size=4, max_tokens=3)
    sched.add(seq)
    runner = FakeModelRunner(next_token_fn=lambda s, p: 50 + s.num_completion_tokens)

    while not sched.is_finished():
        seqs, is_prefill = sched.schedule()
        tokens = runner.run(seqs, is_prefill)
        sched.postprocess(seqs, tokens, is_prefill)

    assert seq.num_completion_tokens == 3
    assert seq.block_table == []
