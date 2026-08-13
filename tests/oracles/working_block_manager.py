"""Working block manager for early scheduler milestones.

This oracle intentionally omits prefix caching. Do not copy it into
nanovllm/engine/block_manager.py — implement the real manager yourself
in milestone 4 (including prefix caching).
"""

from __future__ import annotations

from collections import deque

from nanovllm.engine.sequence import Sequence


class WorkingBlockManager:
    """Simple free-list allocator sufficient for scheduler tests."""

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        # Interface parity with the real BlockManager (Scheduler talks to
        # whichever manager is injected without knowing which); this oracle
        # still does no prefix caching, so there's no hash table here.
        self.req_to_block_ids: dict[int, list[int]] = {}

    def free_count(self) -> int:
        return len(self.free_block_ids)

    def track_request(self, seq: Sequence) -> None:
        self.req_to_block_ids[seq.seq_id] = list(seq.block_table)

    def can_allocate(self, seq: Sequence) -> int:
        if len(self.free_block_ids) < seq.num_blocks:
            return -1
        return 0  # no prefix cache in the stub

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        assert num_cached_blocks == 0
        for _ in range(seq.num_blocks):
            block_id = self.free_block_ids.popleft()
            self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        seq.num_cached_tokens = 0
        self.track_request(seq)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            self.used_block_ids.discard(block_id)
            self.free_block_ids.append(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
        self.req_to_block_ids.pop(seq.seq_id, None)

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            block_id = self.free_block_ids.popleft()
            self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
            self.track_request(seq)

    def hash_blocks(self, seq: Sequence):
        return  # stub: no prefix caching
