"""Unit tests for BlockManager's LRU free-block pool (doubly linked list +
hash-table indices), covering the properties a plain deque-based free list
can't give you in O(1):

  - eviction picks the least-recently-freed block first (LRU order), and
  - a prefix-cache hit that reuses a still-free block ("touch") removes it
    from the *middle* of the free list in O(1), not via a linear scan.

Also covers the two dict-based indices BlockManager maintains:
  - hash_to_block_id: content hash -> the physical block holding it.
  - req_to_block_ids: request (Sequence.seq_id) -> the physical blocks it
    currently holds, kept in sync with seq.block_table.
"""

from __future__ import annotations

from tests.helpers import make_block_manager, make_sequence


def _hash_first_block(bm, seq):
    seq.num_scheduled_tokens = 4
    bm.hash_blocks(seq)
    seq.num_cached_tokens = 4
    seq.num_scheduled_tokens = 0


def test_lru_evicts_oldest_freed_block_first():
    bm = make_block_manager(num_blocks=3, block_size=4)
    a = make_sequence(list(range(4)), block_size=4)
    b = make_sequence(list(range(4, 8)), block_size=4)
    c = make_sequence(list(range(8, 12)), block_size=4)
    bm.allocate(a, 0)
    bm.allocate(b, 0)
    bm.allocate(c, 0)
    a_block, b_block, c_block = a.block_table[0], b.block_table[0], c.block_table[0]

    bm.deallocate(a)  # freed 1st -> oldest, evict-first
    bm.deallocate(b)  # freed 2nd
    bm.deallocate(c)  # freed 3rd -> newest, evict-last
    assert bm.free_block_ids == [a_block, b_block, c_block]

    d = make_sequence(list(range(12, 16)), block_size=4)
    bm.allocate(d, 0)
    assert d.block_table == [a_block]
    e = make_sequence(list(range(16, 20)), block_size=4)
    bm.allocate(e, 0)
    assert e.block_table == [b_block]


def test_lru_touch_removes_block_from_middle_of_free_list_in_o1():
    bm = make_block_manager(num_blocks=3, block_size=4)
    a = make_sequence(list(range(4)), block_size=4)
    b = make_sequence(list(range(4, 8)), block_size=4)
    c = make_sequence(list(range(8, 12)), block_size=4)
    bm.allocate(a, 0)
    bm.allocate(b, 0)
    bm.allocate(c, 0)
    a_block, b_block, c_block = a.block_table[0], b.block_table[0], c.block_table[0]
    for seq in (a, b, c):
        _hash_first_block(bm, seq)

    bm.deallocate(a)
    bm.deallocate(b)
    bm.deallocate(c)
    assert bm.free_block_ids == [a_block, b_block, c_block]

    # A 2-block sequence whose first block matches b's cached content: this
    # must "touch" b_block -- splice it out of the *middle* of the free
    # list -- and separately evict a genuinely new block (from the LRU
    # front) for its second block.
    b2 = make_sequence(list(range(4, 8)) + [999, 998, 997, 996], block_size=4)
    n = bm.can_allocate(b2)
    assert n == 1  # only the first block is a cache hit
    bm.allocate(b2, n)
    assert b2.block_table[0] == b_block  # reused via touch, not a fresh evict
    assert b2.block_table[1] == a_block  # second block: LRU-front eviction (a, oldest-freed)
    # b_block is gone from the free list even though it sat in the middle,
    # not the head -- only c_block (never touched or evicted) remains.
    assert bm.free_block_ids == [c_block]


def test_hash_to_block_id_registers_full_blocks_only():
    bm = make_block_manager(num_blocks=8, block_size=4)
    seq = make_sequence(list(range(4)), block_size=4)  # exactly 1 full block
    bm.allocate(seq, 0)
    assert bm.hash_to_block_id == {}  # nothing hashed until postprocess-time

    _hash_first_block(bm, seq)
    h = bm.compute_hash(seq.block(0))
    assert bm.hash_to_block_id == {h: seq.block_table[0]}


def test_req_to_block_ids_tracks_allocate_growth_and_deallocate():
    bm = make_block_manager(num_blocks=8, block_size=4)
    seq = make_sequence(list(range(4)), block_size=4, max_tokens=20)
    bm.allocate(seq, 0)
    assert bm.req_to_block_ids[seq.seq_id] == seq.block_table

    seq.append_token(99)  # len=5, 5 % 4 == 1 -> next may_append grows a block
    assert bm.can_append(seq)
    bm.may_append(seq)
    assert bm.req_to_block_ids[seq.seq_id] == seq.block_table
    assert len(bm.req_to_block_ids[seq.seq_id]) == 2

    bm.deallocate(seq)
    assert seq.seq_id not in bm.req_to_block_ids


def test_free_count_matches_pool_arithmetic():
    bm = make_block_manager(num_blocks=5, block_size=4)
    assert bm.free_count() == 5
    seq = make_sequence(list(range(8)), block_size=4)  # 2 blocks
    bm.allocate(seq, 0)
    assert bm.free_count() == 3
    bm.deallocate(seq)
    assert bm.free_count() == 5
