from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.utils.debug import debug_log
from nanovllm.course.exceptions import CourseNotImplementedError


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
        # TODO-L2-SCHED-01: Build the prefill batch.
        #
        # Goal:
        # Admit waiting sequences into a prefill (or chunked-prefill) batch.
        #
        # Called from: Scheduler.schedule
        # Next: ModelRunner.run(..., is_prefill=True) then Scheduler.postprocess
        #
        # Inputs:
        # - self.waiting: deque of WAITING sequences (FIFO)
        # - self.max_num_seqs, self.max_num_batched_tokens, self.block_size
        # - self.block_manager.can_allocate / allocate
        #
        # Required behavior:
        # 1. While waiting is non-empty and len(scheduled) < max_num_seqs:
        #    a. Peek waiting[0]
        #    b. remaining = max_num_batched_tokens - num_batched_tokens; break if 0
        #    c. If seq.block_table is empty:
        #         num_cached_blocks = block_manager.can_allocate(seq)
        #         if -1: break (not enough KV space)
        #         num_tokens = seq.num_tokens - num_cached_blocks * block_size
        #       Else (chunk continuation):
        #         num_tokens = seq.num_tokens - seq.num_cached_tokens
        #    d. If remaining < num_tokens and scheduled already non-empty: break
        #       (only the first scheduled seq may be chunked)
        #    e. If block_table empty: block_manager.allocate(seq, num_cached_blocks)
        #    f. seq.num_scheduled_tokens = min(num_tokens, remaining)
        #    g. If cached+scheduled == num_tokens: move WAITING->RUNNING
        #    h. Append seq to scheduled
        # 2. Return (scheduled, True) if scheduled else ([], False) — caller checks
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
        raise CourseNotImplementedError(
            "TODO-L2-SCHED-01",
            subsystem="scheduler",
            tutorial_path="docs/tutorial/03_scheduler.md",
            milestone_test="pytest tests/milestones/test_02_scheduler_basic.py",
            hint="Peek waiting[0]; allocate KV on first visit; support chunked prefill only for the first seq in the batch.",
        )

    def _schedule_decode(self) -> tuple[list[Sequence], bool]:
        # TODO-L2-SCHED-02: Build the decode batch.
        #
        # Goal:
        # Schedule currently running sequences for one new token each.
        #
        # Required behavior:
        # 1. While running and len(scheduled) < max_num_seqs:
        #    a. seq = running.popleft()
        #    b. While not block_manager.can_append(seq):
        #         preempt another running seq (prefer pop from right),
        #         or preempt seq itself and break
        #    c. else (can append):
        #         seq.num_scheduled_tokens = 1
        #         seq.is_prefill = False
        #         block_manager.may_append(seq)
        #         scheduled.append(seq)
        # 2. assert scheduled non-empty
        # 3. running.extendleft(reversed(scheduled))  # restore order
        # 4. return scheduled, False
        #
        # Read: docs/tutorial/04_prefill_and_decode.md
        # Tests: pytest tests/milestones/test_03_prefill_decode.py
        raise CourseNotImplementedError(
            "TODO-L2-SCHED-02",
            subsystem="scheduler",
            tutorial_path="docs/tutorial/04_prefill_and_decode.md",
            milestone_test="pytest tests/milestones/test_03_prefill_decode.py",
            hint="Pop from running, ensure KV append capacity, schedule exactly one token per seq.",
        )

    def preempt(self, seq: Sequence):
        # TODO-L2-SCHED-03: Preempt a running sequence under memory pressure.
        #
        # Required behavior:
        # - status = WAITING
        # - is_prefill = True
        # - block_manager.deallocate(seq)
        # - waiting.appendleft(seq)  # highest priority re-entry
        #
        # Read: docs/tutorial/04_prefill_and_decode.md
        # Tests: pytest tests/milestones/test_03_prefill_decode.py
        raise CourseNotImplementedError(
            "TODO-L2-SCHED-03",
            subsystem="scheduler",
            tutorial_path="docs/tutorial/04_prefill_and_decode.md",
            milestone_test="pytest tests/milestones/test_03_prefill_decode.py",
            hint="Free KV blocks and put the sequence at the front of waiting.",
        )

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        # TODO-L2-SCHED-04: Apply sampled tokens and finish sequences.
        #
        # For each (seq, token_id):
        # 1. block_manager.hash_blocks(seq)  # prefix-cache bookkeeping
        # 2. seq.num_cached_tokens += seq.num_scheduled_tokens
        # 3. seq.num_scheduled_tokens = 0
        # 4. If is_prefill and num_cached_tokens < num_tokens: continue
        #    (chunked prefill not finished — do NOT append/sample yet)
        # 5. seq.append_token(token_id)
        # 6. If (not ignore_eos and token_id == eos) or num_completion_tokens == max_tokens:
        #       status = FINISHED; deallocate; running.remove(seq)
        #
        # Read: docs/tutorial/04_prefill_and_decode.md
        # Tests: pytest tests/milestones/test_03_prefill_decode.py
        raise CourseNotImplementedError(
            "TODO-L2-SCHED-04",
            subsystem="scheduler",
            tutorial_path="docs/tutorial/04_prefill_and_decode.md",
            milestone_test="pytest tests/milestones/test_03_prefill_decode.py",
            hint="Skip append on incomplete chunked prefill; finish on EOS or max_tokens.",
        )
