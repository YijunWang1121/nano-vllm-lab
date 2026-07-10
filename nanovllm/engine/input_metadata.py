"""CPU-side helpers for building model-runner input metadata.

These pure-Python builders are used by ModelRunner and by CPU milestone tests.
They never touch CUDA.
"""

from __future__ import annotations

from nanovllm.engine.sequence import Sequence


def build_block_tables(seqs: list[Sequence]) -> list[list[int]]:
    """Pad per-sequence block tables to a rectangular list-of-lists.

    Returns:
        block_tables[seq_idx][block_idx] -> physical block id, or -1 padding.
    """
    max_len = max(len(seq.block_table) for seq in seqs)
    return [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]


def build_prefill_metadata(
    seqs: list[Sequence],
    block_size: int,
) -> dict:
    """Build flattened prefill tensors (as Python lists) and attention metadata.

    Returns dict with keys:
        input_ids, positions, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, slot_mapping, need_block_tables
    """
    input_ids: list[int] = []
    positions: list[int] = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping: list[int] = []

    for seq in seqs:
        start = seq.num_cached_tokens
        seqlen_q = seq.num_scheduled_tokens
        end = start + seqlen_q
        seqlen_k = end
        input_ids.extend(seq[start:end])
        positions.extend(range(start, end))
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(seqlen_q, max_seqlen_q)
        max_seqlen_k = max(seqlen_k, max_seqlen_k)
        if not seq.block_table:  # warmup path without KV blocks
            continue
        start_block = start // block_size
        end_block = (end + block_size - 1) // block_size
        for i in range(start_block, end_block):
            slot_start = seq.block_table[i] * block_size
            if i == start_block:
                slot_start += start % block_size
            if i != end_block - 1:
                slot_end = seq.block_table[i] * block_size + block_size
            else:
                slot_end = seq.block_table[i] * block_size + end - i * block_size
            slot_mapping.extend(range(slot_start, slot_end))

    need_block_tables = cu_seqlens_k[-1] > cu_seqlens_q[-1]  # prefix cache
    return {
        "input_ids": input_ids,
        "positions": positions,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "slot_mapping": slot_mapping,
        "need_block_tables": need_block_tables,
    }


def build_decode_metadata(
    seqs: list[Sequence],
    block_size: int,
) -> dict:
    """Build decode-step inputs (as Python lists).

    Returns dict with keys:
        input_ids, positions, slot_mapping, context_lens
    """
    input_ids: list[int] = []
    positions: list[int] = []
    slot_mapping: list[int] = []
    context_lens: list[int] = []
    for seq in seqs:
        input_ids.append(seq.last_token)
        positions.append(len(seq) - 1)
        context_lens.append(len(seq))
        slot_mapping.append(seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1)
    return {
        "input_ids": input_ids,
        "positions": positions,
        "slot_mapping": slot_mapping,
        "context_lens": context_lens,
    }
