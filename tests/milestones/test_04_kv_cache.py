"""Milestone 4: KV-cache block manager."""

from __future__ import annotations

import pytest

from nanovllm.course.exceptions import CourseNotImplementedError
from tests.helpers import EXAMPLE_A_PROMPT, EXAMPLE_B_PROMPT, make_block_manager, make_sequence
from tests.oracles.block_oracle import NaiveBlockAllocator


def test_allocate_and_deallocate_basic():
    bm = make_block_manager(num_blocks=8, block_size=4)
    seq = make_sequence(EXAMPLE_A_PROMPT, block_size=4)  # 2 blocks
    try:
        n = bm.can_allocate(seq)
        assert n == 0
        bm.allocate(seq, n)
    except CourseNotImplementedError as e:
        pytest.fail(str(e))
    assert len(seq.block_table) == 2
    assert seq.num_cached_tokens == 0
    assert len(bm.free_block_ids) == 6
    table = list(seq.block_table)
    bm.deallocate(seq)
    assert seq.block_table == []
    assert seq.num_cached_tokens == 0
    assert len(bm.free_block_ids) == 8
    # Freed blocks returned (order may differ from LIFO append).
    assert set(table).issubset(set(bm.free_block_ids) | set(range(8)))


def test_can_allocate_rejects_when_out_of_memory():
    bm = make_block_manager(num_blocks=1, block_size=4)
    seq = make_sequence(EXAMPLE_A_PROMPT, block_size=4)  # needs 2 blocks
    assert bm.can_allocate(seq) == -1


def test_may_append_allocates_new_block_on_boundary():
    bm = make_block_manager(num_blocks=8, block_size=4)
    # 4 tokens -> exactly one full block; next append needs a new block when len % block_size == 1
    seq = make_sequence([1, 2, 3, 4], block_size=4)
    bm.allocate(seq, 0)
    assert len(seq.block_table) == 1
    # After one generated token, len becomes 5, and may_append is called when len%bs==1
    # In the engine, may_append is called BEFORE append_token, when current len % bs == 1.
    # Start with len=4 (full block). After append in postprocess len=5.
    # Before decode of the token that makes len go 4->5, len is 4, 4%4==0, no new block.
    # Before decode when len=5? Actually: can_append/may_append check len(seq)%block_size==1
    # meaning "about to write the first token of a new block".
    seq2 = make_sequence([1, 2, 3, 4, 5], block_size=4)  # len=5, 5%4==1
    bm2 = make_block_manager(num_blocks=8, block_size=4)
    bm2.allocate(seq2, 0)
    assert len(seq2.block_table) == 2  # allocate already gave 2 blocks for 5 tokens
    # Fresh: allocate 4 tokens, then may_append when len%4==1 after conceptually having 4 tokens
    # and scheduling the 5th: len is still 4 during schedule... wait:
    # In decode schedule: may_append is called when len % block_size == 1.
    # So the sequence already has been appended to length where len%bs==1.
    # Example: after first completion token on a 4-token prompt, len=5, 5%4==1.
    seq3 = make_sequence([1, 2, 3, 4], block_size=4)
    bm3 = make_block_manager(num_blocks=8, block_size=4)
    bm3.allocate(seq3, 0)
    seq3.append_token(5)  # now len=5
    assert len(seq3) % 4 == 1
    assert bm3.can_append(seq3)
    before = len(seq3.block_table)
    bm3.may_append(seq3)
    assert len(seq3.block_table) == before + 1


def test_naive_oracle_agrees_on_block_count():
    oracle = NaiveBlockAllocator(8, 4)
    table = oracle.allocate(0, 5)
    assert len(table) == 2
    oracle.free_seq(0)
    assert len(oracle.free) == 8


def test_prefix_cache_reuse_between_sequences():
    bm = make_block_manager(num_blocks=16, block_size=4)
    # Shared full first block [10,11,12,13]
    a = make_sequence(EXAMPLE_A_PROMPT, block_size=4)  # [10..14]
    bm.allocate(a, 0)
    # Simulate hashing the first full block after prefill of first 4 tokens.
    a.num_scheduled_tokens = 4
    a.num_cached_tokens = 0
    try:
        bm.hash_blocks(a)
    except CourseNotImplementedError as e:
        pytest.fail(str(e))
    a.num_cached_tokens = 4
    a.num_scheduled_tokens = 0
    # Finish remaining token + hash if needed — for prefix we only need full blocks hashed.
    # Second sequence shares the first block tokens.
    b = make_sequence([10, 11, 12, 13, 99], block_size=4)
    n = bm.can_allocate(b)
    assert n == 1
    free_before = len(bm.free_block_ids)
    bm.allocate(b, n)
    assert b.block_table[0] == a.block_table[0]
    assert b.num_cached_tokens == 4
    # Only one new block allocated for B's second block.
    assert len(bm.free_block_ids) == free_before - 1


def test_two_request_block_assignment_shapes():
    bm = make_block_manager(num_blocks=16, block_size=4)
    a = make_sequence(EXAMPLE_A_PROMPT, block_size=4)
    b = make_sequence(EXAMPLE_B_PROMPT, block_size=4)
    bm.allocate(a, 0)
    bm.allocate(b, 0)
    # A logical blocks: [0,1] -> 2 physical; B logical: [0] -> 1 physical
    assert len(a.block_table) == 2
    assert len(b.block_table) == 1
    assert len(set(a.block_table) & set(b.block_table)) == 0
