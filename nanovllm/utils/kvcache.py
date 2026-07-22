"""KV cache sizing helpers."""

from __future__ import annotations

from transformers import PretrainedConfig


def block_bytes(
    hf_config: PretrainedConfig,
    *,
    block_size: int,
    tensor_parallel_size: int = 1,
) -> int:
    """Bytes for one paged KV block (K + V across all layers)."""
    num_kv_heads = hf_config.num_key_value_heads // tensor_parallel_size
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    dtype_size = hf_config.dtype.itemsize
    return 2 * hf_config.num_hidden_layers * block_size * num_kv_heads * head_dim * dtype_size


def kvcache_capacity(
    hf_config: PretrainedConfig,
    *,
    num_kvcache_blocks: int,
    block_size: int,
    max_model_len: int,
    tensor_parallel_size: int = 1,
) -> dict[str, int | float]:
    """Estimate how many tokens the allocated paged KV cache can hold."""
    per_block = block_bytes(
        hf_config,
        block_size=block_size,
        tensor_parallel_size=tensor_parallel_size,
    )
    total_slots = num_kvcache_blocks * block_size
    total_bytes = num_kvcache_blocks * per_block
    return {
        "num_blocks": num_kvcache_blocks,
        "block_size": block_size,
        "bytes_per_block": per_block,
        "total_kv_bytes": total_bytes,
        "total_token_slots": total_slots,
        "max_model_len": max_model_len,
        "max_concurrent_seqs_at_max_len": total_slots // max_model_len if max_model_len else 0,
    }


def format_kvcache_capacity(stats: dict[str, int | float]) -> str:
    gib = stats["total_kv_bytes"] / (1024**3)
    slots = stats["total_token_slots"]
    max_len = stats["max_model_len"]
    concurrent = stats["max_concurrent_seqs_at_max_len"]
    return (
        f"KV cache: {stats['num_blocks']} blocks x {stats['block_size']} tokens "
        f"= {slots} total slots ({gib:.2f} GiB), "
        f"~{concurrent} seqs at max_model_len={max_len}"
    )
