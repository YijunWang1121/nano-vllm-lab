from collections import deque


class CpuBlockManager:
    """Free-list allocator for the CPU-side KV-cache swap pool.

    Unlike BlockManager, blocks here are never shared/ref-counted or
    prefix-cache hashed: swap ownership is always exclusive to exactly one
    sequence at a time (Scheduler only swaps sequences whose GPU blocks are
    all private, ref_count == 1 -- see Scheduler._should_swap), so a plain
    free-list is sufficient.
    """

    def __init__(self, num_blocks: int):
        self.free_block_ids: deque[int] = deque(range(num_blocks))

    def can_allocate(self, num_blocks: int) -> bool:
        return len(self.free_block_ids) >= num_blocks

    def allocate(self, num_blocks: int) -> list[int]:
        return [self.free_block_ids.popleft() for _ in range(num_blocks)]

    def deallocate(self, block_ids: list[int]):
        self.free_block_ids.extend(block_ids)
