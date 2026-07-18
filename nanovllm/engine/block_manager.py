from collections import deque
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
        block_id = self.free_block_ids.popleft()
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
        self.free_block_ids.append(block_id)
        return

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            tokens = seq.block(i)
            h = self.compute_hash(tokens, prefix=h)
            if h in self.hash_to_block_id and self.blocks[self.hash_to_block_id.get(h)].token_ids == tokens:
                num_cached_blocks += 1
                if self.hash_to_block_id.get(h) in self.used_block_ids:
                    num_new_blocks -= 1
            else:
                break
        if len(self.free_block_ids) < num_new_blocks:
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
            if cached_block_id in self.free_block_ids:
                self.free_block_ids.remove(cached_block_id)
                self.used_block_ids.add(cached_block_id)
            
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size
        return

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            self.blocks[block_id].ref_count -= 1
            if self.blocks[block_id].ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table = []
        return

    def can_append(self, seq: Sequence) -> bool:
        if len(seq)%seq.block_size==1:
            return len(self.free_block_ids)>=1
        return True

    def may_append(self, seq: Sequence):
        if len(seq) % seq.block_size == 1:
            seq.block_table.append(self._allocate_block())
        return

    def hash_blocks(self, seq: Sequence):
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
