"""CPU-side helpers for building model-runner input metadata.

These pure-Python builders are used by ModelRunner and by CPU milestone tests.
They never touch CUDA.
"""

from __future__ import annotations

from nanovllm.engine.sequence import Sequence
from nanovllm.course.exceptions import CourseNotImplementedError


def build_block_tables(seqs: list[Sequence]) -> list[list[int]]:
    # TODO-L2-RUNNER-01: Pad per-sequence block tables to a rectangle.
    #
    # Goal:
    # Produce block_tables[batch][block] suitable for attention kernels.
    #
    # Algorithm:
    # max_len = max(len(seq.block_table) for seq in seqs)
    # pad each table on the right with -1
    #
    # Example:
    #   A.block_table=[7,2], B.block_table=[5] -> [[7,2],[5,-1]]
    #
    # Called from: ModelRunner.prepare_block_tables / prefix-cache prefill
    # Read: docs/tutorial/06_model_runner_inputs.md
    # Tests: pytest tests/milestones/test_05_model_inputs.py
    raise CourseNotImplementedError(
        "TODO-L2-RUNNER-01",
        subsystem="runner",
        tutorial_path="docs/tutorial/06_model_runner_inputs.md",
        milestone_test="pytest tests/milestones/test_05_model_inputs.py",
        hint="Pad shorter block tables with -1 up to the batch max length.",
    )


def build_prefill_metadata(
    seqs: list[Sequence],
    block_size: int,
) -> dict:
    # TODO-L2-RUNNER-02: Flatten prefill tokens and build slot_mapping.
    #
    # For each seq:
    #   start = num_cached_tokens
    #   seqlen_q = num_scheduled_tokens
    #   end = start + seqlen_q
    #   seqlen_k = end   # includes cached prefix length in the K side
    #   extend input_ids with seq[start:end]
    #   extend positions with range(start, end)
    #   append to cu_seqlens_q / cu_seqlens_k
    #   track max_seqlen_q / max_seqlen_k
    #   If block_table empty (warmup): skip slots
    #   Else map each scheduled token to a physical slot:
    #     slot = physical_block_id * block_size + offset_within_block
    #
    # need_block_tables = (cu_seqlens_k[-1] > cu_seqlens_q[-1])  # prefix cache
    #
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
    raise CourseNotImplementedError(
        "TODO-L2-RUNNER-02",
        subsystem="runner",
        tutorial_path="docs/tutorial/06_model_runner_inputs.md",
        milestone_test="pytest tests/milestones/test_05_model_inputs.py",
        hint="Flatten uncached tokens; slot = block_id * block_size + offset.",
    )


def build_decode_metadata(
    seqs: list[Sequence],
    block_size: int,
) -> dict:
    # TODO-L2-RUNNER-03: Build per-sequence decode inputs.
    #
    # For each seq:
    #   input_ids.append(last_token)
    #   positions.append(len(seq) - 1)
    #   context_lens.append(len(seq))
    #   slot_mapping.append(block_table[-1] * block_size + last_block_num_tokens - 1)
    #
    # Example (A len=5, table=[7,2]): slot = 2*4 + 1 - 1 = 8
    #
    # Read: docs/tutorial/06_model_runner_inputs.md
    # Tests: pytest tests/milestones/test_05_model_inputs.py
    raise CourseNotImplementedError(
        "TODO-L2-RUNNER-03",
        subsystem="runner",
        tutorial_path="docs/tutorial/06_model_runner_inputs.md",
        milestone_test="pytest tests/milestones/test_05_model_inputs.py",
        hint="Decode writes one slot per sequence at the current last-token location.",
    )
