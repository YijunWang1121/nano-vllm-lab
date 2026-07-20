"""Fake model runner for CPU end-to-end continuous-batching tests."""

from __future__ import annotations

from nanovllm.engine.sequence import Sequence

class FakeModelRunner:
    """Returns deterministic next-token IDs without a real model."""

    def __init__(self, next_token_fn=None):
        # Default: emit token_id = 1000 + seq_id for every step.
        self.next_token_fn = next_token_fn or (lambda seq, is_prefill: 1000 + seq.seq_id)

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        return [self.next_token_fn(seq, is_prefill) for seq in seqs]

    def call(self, method_name: str, *args):
        return getattr(self, method_name)(*args)
