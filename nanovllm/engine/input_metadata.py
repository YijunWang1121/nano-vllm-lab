"""CPU-side helpers for building model-runner input metadata.

These pure-Python builders are used by ModelRunner and by CPU milestone tests.
They never touch CUDA.
"""

from __future__ import annotations

from nanovllm.engine.sequence import Sequence

def build_block_tables(seqs: list[Sequence]) -> list[list[int]]:
    block_tables = []
    max_len_seq = max(seqs, key=lambda seq: len(seq.block_table))
    max_len = len(max_len_seq.block_table)
    for seq in seqs:
        num_to_append = max_len - len(seq.block_table)
        block_tables.append(seq.block_table+[-1]*num_to_append)
    return block_tables
        

def build_prefill_metadata(
    seqs: list[Sequence],
    block_size: int,
) -> dict:
    metadata = {"input_ids": [], "positions":[], "cu_seqlens_q":[0], "cu_seqlens_k":[0],
        "max_seqlen_q":0, "max_seqlen_k":0, "slot_mapping":[], "need_block_tables":False
        }
    for seq in seqs:
        start = seq.num_cached_tokens
        seqlen_q = seq.num_scheduled_tokens
        end = start+ seqlen_q
        seqlen_k = end
        metadata["input_ids"] += seq[start:end]
        metadata["positions"] += range(start, end)
        metadata["cu_seqlens_q"].append(seqlen_q + metadata["cu_seqlens_q"][-1]) 
        metadata["cu_seqlens_k"].append(seqlen_k + metadata["cu_seqlens_k"][-1])
        metadata["max_seqlen_q"] = max(metadata["max_seqlen_q"], seqlen_q)
        metadata["max_seqlen_k"] = max(metadata["max_seqlen_k"], seqlen_k)
        if seq.block_table:
            for i in range(start, end):
                logical_block_id = i // seq.block_size
                physical_block_id = seq.block_table[logical_block_id]
                offset = i % seq.block_size
                slot = physical_block_id * seq.block_size + offset
                metadata["slot_mapping"].append(slot)
        metadata["need_block_tables"] = (metadata["cu_seqlens_q"][-1] < metadata["cu_seqlens_k"][-1])

    # Return dict with keys listed in the docstring below.
    #
    # Example (A->[7,2], B->[5], block_size=4, full prefill):
    #   slot_mapping starts [28,29,30,31,8, 20,21,22]
    #
    # Read: docs/tutorial/06_model_runner_inputs.md
    # Tests: pytest tests/milestones/test_05_model_inputs.py
    """
    Returns dict with keys:
        input_ids, positions, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, slot_mapping, need_block_tables
    """
    return metadata

def build_decode_metadata(
    seqs: list[Sequence],
    block_size: int,
) -> dict:
    metadata = {"input_ids": [], "positions":[], "context_lens":[], "slot_mapping":[]
        }
    for seq in seqs:
        metadata["input_ids"].append(seq.last_token)
        metadata["positions"].append(len(seq) - 1)
        metadata["context_lens"].append(len(seq))
        metadata["slot_mapping"].append(seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1)
        
    return metadata

