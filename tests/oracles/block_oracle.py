"""Naive block-allocator oracle for validating student BlockManager behavior.

This is intentionally slow and does not implement prefix caching.
"""

from __future__ import annotations

from collections import deque


class NaiveBlockAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.free = deque(range(num_blocks))
        self.tables: dict[int, list[int]] = {}

    def allocate(self, seq_id: int, num_tokens: int) -> list[int]:
        need = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free) < need:
            raise RuntimeError("out of blocks")
        table = [self.free.popleft() for _ in range(need)]
        self.tables[seq_id] = table
        return table

    def may_append(self, seq_id: int, num_tokens: int) -> None:
        if num_tokens % self.block_size == 1:
            if not self.free:
                raise RuntimeError("out of blocks")
            self.tables[seq_id].append(self.free.popleft())

    def free_seq(self, seq_id: int) -> None:
        for bid in reversed(self.tables.pop(seq_id, [])):
            self.free.append(bid)
