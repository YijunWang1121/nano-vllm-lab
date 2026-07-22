from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.utils.debug import debug_log

class Scheduler:

    def __init__(self, config: Config, block_manager=None):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = block_manager or BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.num_preemptions = 0

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)
        debug_log("scheduler", "add", seq_id=seq.seq_id, num_tokens=seq.num_tokens)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # Prefer prefill; only decode when no prefill work was scheduled.
        scheduled_seqs, is_prefill = self._schedule_prefill()
        if scheduled_seqs:
            return scheduled_seqs, True
        return self._schedule_decode()

    def _schedule_prefill(self) -> tuple[list[Sequence], bool]:
        scheduled = []
        num_batch_tokens = 0
        while self.waiting and len(scheduled) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batch_tokens
            if remaining <= 0: 
                break
            # first time
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                # cannot assign kvcache
                if num_cached_blocks == -1:
                    break
                # num of cached kv cache
                num_tokens_todo = seq.num_tokens - num_cached_blocks * seq.block_size
            # chunked prefill continued
            else:
                num_tokens_todo = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens_todo and scheduled: # chunked prefill 
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens_todo, remaining)
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.running.append(self.waiting.popleft())
            scheduled.append(seq)
            num_batch_tokens += seq.num_scheduled_tokens
        # 2. Return (scheduled, True) if scheduled else ([], False) — caller checks
        return (scheduled, True) if scheduled else ([], False)
        #
        # Return for this helper: (scheduled_seqs, ignored_bool) — schedule()
        # treats non-empty as prefill. Return ([], False) when nothing scheduled.
        #
        # Invariants:
        # - Never exceed max_num_seqs or max_num_batched_tokens
        # - Preserve waiting FIFO order
        # - A seq cannot be in both waiting and running
        #
        # Complexity: O(#waiting examined)
        # Read: docs/tutorial/03_scheduler.md
        # Tests: pytest tests/milestones/test_02_scheduler_basic.py

    def _schedule_decode(self) -> tuple[list[Sequence], bool]:
        scheduled = []
        while self.running and len(scheduled) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    return scheduled,
            # can append
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled.append(seq)
        assert scheduled
        self.running.extendleft(reversed(scheduled))
        return scheduled, False

    def preempt(self, seq: Sequence):
        self.num_preemptions += 1
        debug_log("scheduler", "preempt", seq_id=seq.seq_id, num_tokens=seq.num_tokens)
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        return

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        return
