from collections import deque

from nanovllm.config import Config
from nanovllm.exceptions import RequestTooLargeError
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.cpu_block_manager import CpuBlockManager
from nanovllm.utils.debug import debug_log

class Scheduler:

    def __init__(self, config: Config, block_manager=None):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.chunk_prefill_tokens = config.chunk_prefill_tokens
        self.max_model_len = config.max_model_len
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = block_manager or BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size,
            enable_prefix_caching=getattr(config, "enable_prefix_caching", True),
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.num_preemptions = 0
        # Sequences aborted this step (request rejected as unservable, or a
        # decode-phase deadlock -- see abort()/_schedule_decode) drained by
        # LLMEngine and reported to the caller instead of silently dropped.
        self.pending_aborts: list[Sequence] = []

        # Optional CPU KV-cache swap: on preemption, private (non-shared)
        # blocks are copied to a CPU pool instead of discarded, so
        # re-admission can skip the recompute forward pass. Off by default
        # (Config.kv_swap_enabled) -- see preempt()/_should_swap().
        self.kv_swap_enabled = config.kv_swap_enabled
        self.kv_swap_min_tokens = config.kv_swap_min_tokens
        self.cpu_block_manager = CpuBlockManager(config.num_cpu_kvcache_blocks) if config.kv_swap_enabled else None
        self.num_swaps = 0
        # (gpu_block_ids, cpu_block_ids) / (cpu_block_ids, gpu_block_ids) pairs
        # queued here by preempt()/_schedule_swap_ins(); LLMEngine.step() drains
        # and executes them via ModelRunner.call("swap_out"/"swap_in", ...)
        # before the next model forward pass -- see drain_pending_swap_outs/ins.
        self.pending_swap_outs: list[tuple[list[int], list[int]]] = []
        self.pending_swap_ins: list[tuple[list[int], list[int]]] = []

    def is_finished(self):
        return not self.waiting and not self.running

    def _pool_capacity(self) -> int:
        # Total blocks in the GPU KV-cache pool, free or in-use. Uses
        # free_count() (O(1) on both the real BlockManager's LRU node map
        # and the WorkingBlockManager test oracle's deque) rather than
        # `len(self.block_manager.blocks)`, which only exists on the real
        # BlockManager -- tests inject a WorkingBlockManager oracle
        # (tests/oracles/) that has no `.blocks`.
        return self.block_manager.free_count() + len(self.block_manager.used_block_ids)

    def add(self, seq: Sequence):
        # Reject requests that can never be served, rather than admitting
        # them into `waiting` where BlockManager.can_allocate would return
        # -1 forever and Scheduler.is_finished() would never become True.
        if seq.num_prompt_tokens > self.max_model_len:
            raise RequestTooLargeError(
                f"seq {seq.seq_id}: prompt has {seq.num_prompt_tokens} tokens, "
                f"exceeds max_model_len={self.max_model_len}"
            )
        pool_capacity = self._pool_capacity()
        if seq.num_blocks > pool_capacity:
            raise RequestTooLargeError(
                f"seq {seq.seq_id}: prompt needs {seq.num_blocks} KV-cache blocks, "
                f"but this engine's pool only has {pool_capacity} blocks total"
            )
        self.waiting.append(seq)
        debug_log("scheduler", "add", seq_id=seq.seq_id, num_tokens=seq.num_tokens)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # Prefer prefill; only decode when no prefill work was scheduled.
        scheduled_seqs, is_prefill = self._schedule_prefill()
        if scheduled_seqs:
            return scheduled_seqs, True
        return self._schedule_decode()

    def _schedule_prefill(self) -> tuple[list[Sequence], bool]:
        self._schedule_swap_ins()
        if self.chunk_prefill_tokens > 0:
            return self._schedule_prefill_chunked()
        return self._schedule_prefill_classic()

    def _schedule_swap_ins(self) -> None:
        # Restore any swapped-out waiting sequences whose CPU-side KV data
        # can now be copied back to GPU, *before* the normal prefill loop
        # runs -- swap-in isn't token-budget work, it's a memory copy, so it
        # doesn't compete with chunk_prefill_tokens/max_num_batched_tokens.
        #
        # A preempted (decode-phase) sequence always has num_cached_tokens
        # == num_tokens - 1 at swap-out time: decode is one step "behind"
        # (the most recently sampled token's own K/V isn't written until
        # the *next* decode step processes it as last_token). That pending
        # token, and any block growth it needs, is exactly what an ordinary
        # decode step already handles via can_append/may_append -- so a
        # restored sequence goes straight into self.running and takes its
        # next turn through the normal decode path, rather than through
        # _schedule_prefill_classic/chunked's "continuation" branch, which
        # never calls may_append and so can't grow block_table across a new
        # block boundary. Restore exactly the blocks that were saved (never
        # top up here): may_append's own len(seq) % block_size == 1 check
        # already fires on this seq's very next decode turn if -- and only
        # if -- it's genuinely short a block, so pre-growing here would
        # double-count and make can_append wrongly demand a block that
        # block_table already doesn't need.
        if not self.kv_swap_enabled or not self.waiting:
            return
        still_waiting: deque[Sequence] = deque()
        for seq in self.waiting:
            if seq.swap_state != "swapped":
                still_waiting.append(seq)
                continue
            num_blocks = len(seq.cpu_block_table)
            if self.block_manager.free_count() < num_blocks:
                still_waiting.append(seq)  # not enough free GPU blocks yet; retry next tick
                continue
            gpu_block_ids = self.block_manager.allocate_fresh_blocks(num_blocks)
            self.pending_swap_ins.append((list(seq.cpu_block_table), list(gpu_block_ids)))
            debug_log("scheduler", "swap_in", seq_id=seq.seq_id, num_blocks=num_blocks)
            seq.block_table = gpu_block_ids
            self.block_manager.track_request(seq)  # allocate_fresh_blocks bypasses allocate(), sync manually
            seq.num_cached_tokens = seq.num_swapped_tokens
            seq.num_scheduled_tokens = 0
            self.cpu_block_manager.deallocate(seq.cpu_block_table)
            seq.cpu_block_table = []
            seq.num_swapped_tokens = 0
            seq.swap_state = "none"
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
        self.waiting = still_waiting

    def _schedule_prefill_classic(self) -> tuple[list[Sequence], bool]:
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

    def _schedule_prefill_chunked(self) -> tuple[list[Sequence], bool]:
        # Like _schedule_prefill_classic, but caps any single sequence's
        # per-step contribution at self.chunk_prefill_tokens instead of
        # letting the first (possibly very long) waiting sequence consume
        # the whole max_num_batched_tokens budget. That leaves room in the
        # same step for shorter sequences queued behind it, so a long
        # prefill no longer fully head-of-line-blocks short ones behind it
        # for its entire duration — it only "steals" one chunk's worth of
        # budget per step.
        #
        # Still processes self.waiting in FIFO order (no reordering by
        # size), and still stops admitting entirely once the KV cache is
        # out of free blocks (matches classic's "break" there) so block
        # allocation ordering semantics are unchanged.
        scheduled = []
        num_batch_tokens = 0
        still_waiting: deque[Sequence] = deque()
        blocked = False  # KV cache exhausted: stop admitting further seqs this step

        for seq in self.waiting:
            if blocked or len(scheduled) >= self.max_num_seqs:
                still_waiting.append(seq)
                continue
            remaining = self.max_num_batched_tokens - num_batch_tokens
            if remaining <= 0:
                still_waiting.append(seq)
                continue
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    still_waiting.append(seq)
                    blocked = True
                    continue
                num_tokens_todo = seq.num_tokens - num_cached_blocks * seq.block_size
            else:
                num_tokens_todo = seq.num_tokens - seq.num_cached_tokens
            chunk = min(num_tokens_todo, remaining, self.chunk_prefill_tokens)
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = chunk
            num_batch_tokens += chunk
            if seq.num_cached_tokens + chunk == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            else:
                still_waiting.append(seq)  # not done: keep its place in line for the next step
            scheduled.append(seq)

        self.waiting = still_waiting
        return (scheduled, True) if scheduled else ([], False)

    def _schedule_decode(self) -> tuple[list[Sequence], bool]:
        scheduled = []
        while self.running and len(scheduled) < self.max_num_seqs:
            seq = self.running.popleft()
            aborted_seq = False
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                elif len(seq.block_table) >= self._pool_capacity():
                    # seq alone already occupies every block the pool will
                    # ever have -- no future preemption or completion of
                    # anything else can ever free a block for it (there is
                    # nothing else to free). Preempting it here would just
                    # bounce it back into waiting, get re-admitted at the
                    # same size, and hit this exact wall again next decode
                    # step, forever. Abort it instead.
                    self.abort(seq, reason="decode_oom")
                    aborted_seq = True
                    break
                else:
                    self.preempt(seq)
                    # Sequences already popped into `scheduled` this round
                    # must go back into self.running here too -- this early
                    # return used to skip the extendleft below entirely,
                    # silently orphaning them (still marked RUNNING but
                    # tracked in neither self.running nor self.waiting).
                    if scheduled:
                        self.running.extendleft(reversed(scheduled))
                    return scheduled, False
            # can append
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled.append(seq)
            if aborted_seq:
                if scheduled:
                    self.running.extendleft(reversed(scheduled))
                return scheduled, False
        assert scheduled
        self.running.extendleft(reversed(scheduled))
        return scheduled, False

    def _schedule_unified(self) -> list[Sequence]:
        # Experimental: admit every running (decode) sequence AND
        # waiting/continuing prefill work into ONE batch, all driven through
        # a single flash_attn_varlen_func call (see ModelRunner.run_model --
        # passing is_prefill=True for the whole batch). This works because
        # decode is just the degenerate case of prefill with
        # num_scheduled_tokens=1 against a long cached prefix: the varlen
        # kernel already reads historical K/V via block_table and writes the
        # new token via slot_mapping for prefix-cache-hit prefill, which is
        # exactly what a decode step needs too -- see attention.py.
        # Always eager: run_model's `if is_prefill: eager` check means this
        # mode never hits a captured CUDA graph, by construction.
        self._schedule_swap_ins()
        scheduled: list[Sequence] = []
        num_batch_tokens = 0

        # Decode turns: same preemption logic as _schedule_decode, but each
        # admitted seq is folded into the single unified `scheduled` list
        # (num_scheduled_tokens=1) instead of being run as its own batch.
        decode_admitted: list[Sequence] = []
        while self.running:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                elif len(seq.block_table) >= self._pool_capacity():
                    # Same unrecoverable case as _schedule_decode: seq alone
                    # (after evicting everyone else, if any) still needs
                    # more blocks than the pool will ever have. Abort rather
                    # than preempt-and-loop-forever.
                    self.abort(seq, reason="decode_oom")
                    seq = None
                    break
                else:
                    self.preempt(seq)
                    seq = None
                    break
            if seq is None:
                break
            seq.num_scheduled_tokens = 1
            seq.is_prefill = False  # per-Sequence flag for TP pickling only; orthogonal to this batch's kernel routing
            self.block_manager.may_append(seq)
            decode_admitted.append(seq)
            num_batch_tokens += 1
        self.running.extendleft(reversed(decode_admitted))
        scheduled.extend(decode_admitted)

        # Prefill continuations / new admissions with whatever budget is left.
        still_waiting: deque[Sequence] = deque()
        blocked = False
        for seq in self.waiting:
            if blocked or len(scheduled) >= self.max_num_seqs:
                still_waiting.append(seq)
                continue
            remaining = self.max_num_batched_tokens - num_batch_tokens
            if remaining <= 0:
                still_waiting.append(seq)
                continue
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    still_waiting.append(seq)
                    blocked = True
                    continue
                num_tokens_todo = seq.num_tokens - num_cached_blocks * seq.block_size
            else:
                num_tokens_todo = seq.num_tokens - seq.num_cached_tokens
            cap = self.chunk_prefill_tokens if self.chunk_prefill_tokens > 0 else num_tokens_todo
            chunk = min(num_tokens_todo, remaining, cap)
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = chunk
            seq.is_prefill = True
            num_batch_tokens += chunk
            if seq.num_cached_tokens + chunk == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            else:
                still_waiting.append(seq)
            scheduled.append(seq)
        self.waiting = still_waiting

        return scheduled

    def preempt(self, seq: Sequence):
        self.num_preemptions += 1
        debug_log("scheduler", "preempt", seq_id=seq.seq_id, num_tokens=seq.num_tokens)
        seq.status = SequenceStatus.WAITING
        if self._should_swap(seq):
            self._swap_out(seq)
        else:
            seq.is_prefill = True
            self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        return

    def abort(self, seq: Sequence, reason: str = "") -> None:
        # A sequence that can never be served (e.g. it alone needs more
        # blocks than the KV-cache pool will ever have) or whose forward
        # pass hit a real CUDA OOM. Mark it FINISHED and free its resources
        # instead of cycling it back through preempt() forever -- see
        # _schedule_decode/_schedule_unified's decode-admission loops and
        # LLMEngine's step methods (real-OOM path).
        seq.aborted = True
        seq.status = SequenceStatus.FINISHED
        if seq.block_table:
            self.block_manager.deallocate(seq)
        if seq in self.running:
            self.running.remove(seq)
        if seq in self.waiting:
            self.waiting.remove(seq)
        self.pending_aborts.append(seq)
        debug_log("scheduler", "abort", seq_id=seq.seq_id, reason=reason)

    def abort_batch(self, seqs: list[Sequence], reason: str = "") -> None:
        for seq in seqs:
            self.abort(seq, reason)

    def drain_pending_aborts(self) -> list[Sequence]:
        out, self.pending_aborts = self.pending_aborts, []
        return out

    def _should_swap(self, seq: Sequence) -> bool:
        if self.cpu_block_manager is None or not seq.block_table:
            return False
        if seq.num_cached_tokens < self.kv_swap_min_tokens:
            return False
        # A sequence that already needs the *entire* GPU pool by itself
        # can't be helped by swapping: it would just be evicted again on
        # its very next decode turn (still needing one more block than the
        # pool can ever hold at once), forever. Recompute-preempt is no
        # more feasible here either, but at least fails the same way this
        # engine now fails for any oversized single request (hits
        # Scheduler.abort() via _schedule_decode's deadlock check) rather
        # than looping silently.
        if len(seq.block_table) >= self._pool_capacity():
            return False
        # Only swap sequences whose blocks are entirely private. A shared
        # (prefix-cached) block is still serving other live sequences --
        # swapping it out from under them would corrupt their generation.
        # Falling back to plain recompute for any partially-shared sequence
        # keeps swap strictly all-or-nothing (no partial/mixed bookkeeping).
        if not all(self.block_manager.blocks[bid].ref_count == 1 for bid in seq.block_table):
            return False
        return self.cpu_block_manager.can_allocate(len(seq.block_table))

    def _swap_out(self, seq: Sequence):
        self.num_swaps += 1
        num_blocks = len(seq.block_table)
        cpu_block_ids = self.cpu_block_manager.allocate(num_blocks)
        self.pending_swap_outs.append((list(seq.block_table), list(cpu_block_ids)))
        debug_log("scheduler", "swap_out", seq_id=seq.seq_id, num_blocks=num_blocks)
        seq.cpu_block_table = cpu_block_ids
        seq.num_swapped_tokens = seq.num_cached_tokens
        seq.swap_state = "swapped"
        # Safe to free the GPU blocks now, same as plain deallocate: the
        # copy is queued and will run (and sync) in LLMEngine.step() before
        # the next model forward pass touches these physical block ids
        # again, so nothing can observe half-copied data. See
        # ModelRunner.swap_out/step() ordering.
        self.block_manager.deallocate(seq)

    def drain_pending_swap_outs(self) -> list[tuple[list[int], list[int]]]:
        out, self.pending_swap_outs = self.pending_swap_outs, []
        return out

    def drain_pending_swap_ins(self) -> list[tuple[list[int], list[int]]]:
        out, self.pending_swap_ins = self.pending_swap_ins, []
        return out

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
