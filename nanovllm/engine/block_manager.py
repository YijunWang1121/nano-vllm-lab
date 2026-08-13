import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence
from nanovllm.utils.debug import debug_log

class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class _LruNode:
    """One entry in the free-block LRU doubly linked list."""
    __slots__ = ("block_id", "prev", "next")

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.prev: "_LruNode | None" = None
        self.next: "_LruNode | None" = None


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.enable_prefix_caching = enable_prefix_caching
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # Content-addressed prefix-cache index: chained hash (see
        # compute_hash -- each block's hash is seeded with its predecessor's,
        # so a hash match implies the *whole* prefix up to that block
        # matches, not just that one block's own tokens) -> the one physical
        # block currently holding that prefix's KV data.
        self.hash_to_block_id: dict[int, int] = dict()
        # Which physical blocks a given request (Sequence.seq_id) currently
        # holds -- a BlockManager-side mirror of Sequence.block_table, kept
        # in sync by allocate()/may_append()/deallocate() via
        # track_request(). The KV-cache swap-in path bypasses allocate() (it
        # already knows exactly which blocks to restore) and calls
        # track_request() directly -- see Scheduler._schedule_swap_ins.
        self.req_to_block_ids: dict[int, list[int]] = dict()
        self.used_block_ids: set[int] = set()

        # Free-block pool as an LRU doubly linked list: _lru_head.next is
        # the least-recently-freed (evict-first) block, _lru_tail.prev is
        # the most-recently-freed (evict-last). Two dummy sentinel nodes
        # make insert/remove branch-free at both ends. `_lru_node_of` maps a
        # block_id to its node, so a prefix-cache hit that reuses a
        # still-free block ("touch") can splice it out of the *middle* of
        # the list in O(1) -- the whole reason for a real linked list here
        # instead of a plain list/deque, which needs an O(n) scan to remove
        # an arbitrary element.
        self._lru_head = _LruNode(-1)
        self._lru_tail = _LruNode(-1)
        self._lru_head.next = self._lru_tail
        self._lru_tail.prev = self._lru_head
        self._lru_node_of: dict[int, _LruNode] = {}
        for i in range(num_blocks):
            self._lru_push_mru(i)

        # Cumulative prefix-cache stats (over all allocated sequences).
        self.total_prompt_tokens = 0
        self.total_cached_tokens = 0

    # -- LRU free-list primitives (all O(1)) --------------------------------

    def _lru_push_mru(self, block_id: int) -> None:
        """Insert block_id as the most-recently-freed (evict-last) block."""
        node = _LruNode(block_id)
        self._lru_node_of[block_id] = node
        prev = self._lru_tail.prev
        prev.next = node
        node.prev = prev
        node.next = self._lru_tail
        self._lru_tail.prev = node

    def _lru_remove(self, block_id: int) -> None:
        """Splice a block out of the free list from wherever it sits --
        used both by real eviction (from the head) and by a cache-hit touch
        (from anywhere in the middle)."""
        node = self._lru_node_of.pop(block_id)
        node.prev.next = node.next
        node.next.prev = node.prev

    def _lru_pop_lru(self) -> int:
        """Evict and return the least-recently-freed block."""
        node = self._lru_head.next
        assert node is not self._lru_tail, "no free blocks"
        self._lru_remove(node.block_id)
        return node.block_id

    def free_count(self) -> int:
        return len(self._lru_node_of)

    @property
    def free_block_ids(self) -> list[int]:
        """Read-only snapshot of currently-free block ids, oldest-freed
        (evict-first) to newest-freed (evict-last). O(n) -- for
        inspection/tests only; internal code uses _lru_node_of/free_count()
        for O(1) membership/count checks instead."""
        ids = []
        node = self._lru_head.next
        while node is not self._lru_tail:
            ids.append(node.block_id)
            node = node.next
        return ids

    def track_request(self, seq: Sequence) -> None:
        """Sync req_to_block_ids from seq.block_table."""
        self.req_to_block_ids[seq.seq_id] = list(seq.block_table)

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self._lru_pop_lru()
        block = self.blocks[block_id]
        assert block.ref_count==0
        if block.hash in self.hash_to_block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        victim_block = self.blocks[block_id]
        assert victim_block.ref_count == 0
        self.used_block_ids.remove(block_id)
        self._lru_push_mru(block_id)
        return

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        if not self.enable_prefix_caching:
            return -1 if self.free_count() < num_new_blocks else 0
        for i in range(seq.num_blocks - 1):
            tokens = seq.block(i)
            h = self.compute_hash(tokens, prefix=h)
            if h in self.hash_to_block_id and self.blocks[self.hash_to_block_id.get(h)].token_ids == tokens:
                num_cached_blocks += 1
                if self.hash_to_block_id.get(h) in self.used_block_ids:
                    num_new_blocks -= 1
            else:
                break
        if self.free_count() < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            tokens = seq.block(i)
            h = self.compute_hash(tokens, prefix=h)
            cached_block_id = self.hash_to_block_id[h]
            self.blocks[cached_block_id].ref_count += 1
            seq.block_table.append(cached_block_id)
            if cached_block_id in self._lru_node_of:
                # Still sitting in the free list (ref_count was 0 a moment
                # ago) -- this hit "touches" it, removing it from eviction
                # candidacy in O(1) instead of a linear scan-and-remove.
                self._lru_remove(cached_block_id)
                self.used_block_ids.add(cached_block_id)

        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size
        self.total_prompt_tokens += seq.num_tokens
        self.total_cached_tokens += seq.num_cached_tokens
        self.track_request(seq)
        return

    def prefix_cache_stats(self) -> dict[str, int | float]:
        hit_rate = self.total_cached_tokens / self.total_prompt_tokens if self.total_prompt_tokens else 0.0
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "cached_tokens": self.total_cached_tokens,
            "hit_rate": hit_rate,
        }

    def reset_prefix_cache_stats(self):
        self.total_prompt_tokens = 0
        self.total_cached_tokens = 0

    def allocate_fresh_blocks(self, num_blocks: int) -> list[int]:
        """Allocate `num_blocks` blocks with no prefix-cache lookup.

        Used by KV-cache swap-in, where the caller already knows exactly
        which logical blocks it's restoring from the CPU pool -- no hash
        matching applies. Caller must ensure free_count() >= num_blocks
        first (mirrors the can_append/may_append precondition pattern
        already used elsewhere in this class), and must call
        track_request(seq) after assigning the result to seq.block_table.
        """
        return [self._allocate_block() for _ in range(num_blocks)]

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            self.blocks[block_id].ref_count -= 1
            if self.blocks[block_id].ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table = []
        self.req_to_block_ids.pop(seq.seq_id, None)
        return

    def can_append(self, seq: Sequence) -> bool:
        if len(seq)%seq.block_size==1:
            return self.free_count() >= 1
        return True

    def may_append(self, seq: Sequence):
        if len(seq) % seq.block_size == 1:
            seq.block_table.append(self._allocate_block())
            self.track_request(seq)
        return

    def hash_blocks(self, seq: Sequence):
        if not self.enable_prefix_caching:
            return
        start = seq.num_cached_tokens // seq.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // seq.block_size
        if start == end:
            # the same block
            return
        h = -1
        if start > 0:
            h = self.blocks[seq.block_table[start-1]].hash
        for i in range(start, end):
            block_idx: int = seq.block_table[i]
            block = self.blocks[block_idx]
            token_ids = seq.block(i)

            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_idx
        return
