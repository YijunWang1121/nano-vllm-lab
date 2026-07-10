from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence
from nanovllm.utils.debug import debug_log
from nanovllm.course.exceptions import CourseNotImplementedError


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


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        # TODO-L2-KVCACHE-01: Allocate one physical block from the free list.
        #
        # Required behavior:
        # 1. Pop free_block_ids
        # 2. Assert ref_count == 0
        # 3. If block still mapped in hash_to_block_id, delete stale mapping
        # 4. block.reset()  # sets ref_count=1, clears hash/token_ids
        # 5. used_block_ids.add(block_id)
        # 6. Return block_id
        #
        # Read: docs/tutorial/05_kv_cache_blocks.md
        # Tests: pytest tests/milestones/test_04_kv_cache.py
        raise CourseNotImplementedError(
            "TODO-L2-KVCACHE-01",
            subsystem="kvcache",
            tutorial_path="docs/tutorial/05_kv_cache_blocks.md",
            milestone_test="pytest tests/milestones/test_04_kv_cache.py",
            hint="Pop from free_block_ids, reset the Block, mark used.",
        )

    def _deallocate_block(self, block_id: int):
        # TODO-L2-KVCACHE-01 (continued): Return a physical block to the free list.
        #
        # Assert ref_count == 0; remove from used; append to free.
        raise CourseNotImplementedError(
            "TODO-L2-KVCACHE-01",
            subsystem="kvcache",
            tutorial_path="docs/tutorial/05_kv_cache_blocks.md",
            milestone_test="pytest tests/milestones/test_04_kv_cache.py",
            hint="Only free when ref_count hits zero.",
        )

    def can_allocate(self, seq: Sequence) -> int:
        # TODO-L3-KVCACHE-02: Check capacity and count prefix-cache hits.
        #
        # Returns:
        # -1 if not enough free blocks for the uncached portion
        # otherwise num_cached_blocks (full blocks matched via hash chain)
        #
        # Algorithm sketch:
        # - h=-1; num_cached_blocks=0; num_new_blocks=seq.num_blocks
        # - For i in range(seq.num_blocks - 1):  # only full blocks
        #     hash block tokens with prefix h; look up hash_to_block_id
        #     miss or token mismatch -> break
        #     hit: num_cached_blocks += 1; if block already used: num_new_blocks -= 1
        # - If len(free_block_ids) < num_new_blocks: return -1
        # - return num_cached_blocks
        #
        # Read: docs/tutorial/05_kv_cache_blocks.md and docs/tutorial/12_prefix_caching.md
        raise CourseNotImplementedError(
            "TODO-L3-KVCACHE-02",
            subsystem="kvcache",
            tutorial_path="docs/tutorial/05_kv_cache_blocks.md",
            milestone_test="pytest tests/milestones/test_04_kv_cache.py",
            hint="Walk full blocks only; shared used blocks reduce num_new_blocks.",
        )

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        # TODO-L3-KVCACHE-03: Build seq.block_table from cache hits + new blocks.
        #
        # Preconditions: seq.block_table is empty
        # 1. For i in [0, num_cached_blocks): share hashed block (bump ref_count /
        #    move from free->used if currently free)
        # 2. For remaining logical blocks: append _allocate_block()
        # 3. seq.num_cached_tokens = num_cached_blocks * block_size
        #
        # Read: docs/tutorial/05_kv_cache_blocks.md
        raise CourseNotImplementedError(
            "TODO-L3-KVCACHE-03",
            subsystem="kvcache",
            tutorial_path="docs/tutorial/05_kv_cache_blocks.md",
            milestone_test="pytest tests/milestones/test_04_kv_cache.py",
            hint="Reuse hashed blocks first, then allocate fresh physical blocks.",
        )

    def deallocate(self, seq: Sequence):
        # TODO-L2-KVCACHE-04: Drop refs for all blocks in seq.block_table.
        #
        # Walk block_table in reverse; decrement ref_count; free when 0.
        # Clear num_cached_tokens and block_table.
        raise CourseNotImplementedError(
            "TODO-L2-KVCACHE-04",
            subsystem="kvcache",
            tutorial_path="docs/tutorial/05_kv_cache_blocks.md",
            milestone_test="pytest tests/milestones/test_04_kv_cache.py",
            hint="Reverse order helps keep free-list behavior predictable.",
        )

    def can_append(self, seq: Sequence) -> bool:
        # TODO-L2-KVCACHE-05: Return whether decode may proceed.
        #
        # Need a free block iff len(seq) % block_size == 1
        # (about to write the first token of a new page).
        raise CourseNotImplementedError(
            "TODO-L2-KVCACHE-05",
            subsystem="kvcache",
            tutorial_path="docs/tutorial/05_kv_cache_blocks.md",
            milestone_test="pytest tests/milestones/test_04_kv_cache.py",
            hint="len(free) >= (len(seq) % block_size == 1)",
        )

    def may_append(self, seq: Sequence):
        # TODO-L2-KVCACHE-05 (continued): Append a physical block when needed.
        #
        # If len(seq) % block_size == 1: seq.block_table.append(_allocate_block())
        # Called during decode scheduling BEFORE the new token is appended.
        raise CourseNotImplementedError(
            "TODO-L2-KVCACHE-05",
            subsystem="kvcache",
            tutorial_path="docs/tutorial/05_kv_cache_blocks.md",
            milestone_test="pytest tests/milestones/test_04_kv_cache.py",
            hint="Allocate the next page when the next token starts a new block.",
        )

    def hash_blocks(self, seq: Sequence):
        # TODO-L3-KVCACHE-06: Hash newly completed full blocks for prefix cache.
        #
        # start = num_cached_tokens // block_size
        # end = (num_cached_tokens + num_scheduled_tokens) // block_size
        # if start == end: return
        # Chain hashes across blocks; update Block and hash_to_block_id.
        #
        # Read: docs/tutorial/12_prefix_caching.md
        raise CourseNotImplementedError(
            "TODO-L3-KVCACHE-06",
            subsystem="kvcache",
            tutorial_path="docs/tutorial/12_prefix_caching.md",
            milestone_test="pytest tests/milestones/test_04_kv_cache.py",
            hint="Only full blocks become prefix-cache keys.",
        )
