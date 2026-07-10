"""Shared helpers for CPU-only course tests."""

from __future__ import annotations

from types import SimpleNamespace

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.block_manager import BlockManager
from nanovllm.sampling_params import SamplingParams


def make_sampling_params(
    temperature: float = 0.8,
    max_tokens: int = 16,
    ignore_eos: bool = False,
) -> SamplingParams:
    return SamplingParams(temperature=temperature, max_tokens=max_tokens, ignore_eos=ignore_eos)


def make_sequence(
    token_ids: list[int],
    *,
    block_size: int = 4,
    max_tokens: int = 16,
    temperature: float = 0.8,
    ignore_eos: bool = False,
) -> Sequence:
    """Create a Sequence with a small educational block_size (default 4)."""
    Sequence.block_size = block_size
    return Sequence(token_ids, make_sampling_params(temperature, max_tokens, ignore_eos))


def make_scheduler_config(
    *,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 64,
    eos: int = 2,
    block_size: int = 4,
    num_kvcache_blocks: int = 32,
) -> SimpleNamespace:
    return SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos=eos,
        kvcache_block_size=block_size,
        num_kvcache_blocks=num_kvcache_blocks,
    )


def make_scheduler(**kwargs) -> Scheduler:
    Sequence.block_size = kwargs.get("block_size", 4)
    return Scheduler(make_scheduler_config(**kwargs))


def make_block_manager(num_blocks: int = 32, block_size: int = 4) -> BlockManager:
    Sequence.block_size = block_size
    return BlockManager(num_blocks, block_size)


# Running example used throughout the tutorial:
# Request A prompt length 5, Request B prompt length 3, block size 4.
EXAMPLE_A_PROMPT = [10, 11, 12, 13, 14]
EXAMPLE_B_PROMPT = [20, 21, 22]
