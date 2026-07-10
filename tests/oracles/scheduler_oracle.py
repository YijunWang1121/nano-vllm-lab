"""Naive scheduler simulator for educational comparison."""

from __future__ import annotations

from collections import deque


class NaiveSchedulerSim:
    """FCFS prefill-then-decode simulator without KV capacity checks."""

    def __init__(self, max_num_seqs: int, max_num_batched_tokens: int):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.waiting: deque[int] = deque()
        self.running: deque[int] = deque()
        self.prompt_lens: dict[int, int] = {}
        self.cached: dict[int, int] = {}

    def add(self, seq_id: int, prompt_len: int):
        self.waiting.append(seq_id)
        self.prompt_lens[seq_id] = prompt_len
        self.cached[seq_id] = 0

    def schedule(self):
        batch = []
        tokens = 0
        while self.waiting and len(batch) < self.max_num_seqs:
            sid = self.waiting[0]
            need = self.prompt_lens[sid] - self.cached[sid]
            if tokens + need > self.max_num_batched_tokens and batch:
                break
            take = min(need, self.max_num_batched_tokens - tokens)
            self.cached[sid] += take
            tokens += take
            batch.append(sid)
            if self.cached[sid] == self.prompt_lens[sid]:
                self.waiting.popleft()
                self.running.append(sid)
        if batch:
            return batch, True
        batch = list(self.running)[: self.max_num_seqs]
        return batch, False
