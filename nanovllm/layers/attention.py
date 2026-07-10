import torch
from torch import nn

from nanovllm.utils.context import get_context
from nanovllm.utils.debug import debug_log

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only / course environments
    triton = None
    tl = None

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:  # CPU-only / course environments
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None


if triton is not None:

    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1: return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # TODO-L2-ATTN-01: Select the attention execution path from Context.
        #
        # Goal:
        # Integrate KV-cache writes and choose prefill vs decode kernels.
        # Do NOT reimplement FlashAttention — call the provided functions.
        #
        # Required behavior:
        # 1. context = get_context()
        # 2. If k_cache/v_cache are allocated: store_kvcache(k, v, ..., context.slot_mapping)
        # 3. If context.is_prefill:
        #      - If context.block_tables is not None (prefix cache): read K/V from cache
        #      - o = flash_attn_varlen_func(... cu_seqlens / max_seqlen / block_table ...)
        #    Else (decode):
        #      - o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
        #             cache_seqlens=context.context_lens, block_table=context.block_tables, ...)
        # 4. Return o
        #
        # Tensor contracts:
        # - q/k/v: [N_tokens, num_heads, head_dim] (prefill) or decode q with N=batch
        # - slot_mapping: [N_tokens] int32 physical slots
        #
        # Read: docs/tutorial/07_attention_metadata.md
        # Tests: pytest tests/milestones/test_06_attention_metadata.py
        from nanovllm.course.exceptions import CourseNotImplementedError
        raise CourseNotImplementedError(
            "TODO-L2-ATTN-01",
            subsystem="attention",
            tutorial_path="docs/tutorial/07_attention_metadata.md",
            milestone_test="pytest tests/milestones/test_06_attention_metadata.py",
            hint="Prefill uses varlen flash-attn; decode uses flash_attn_with_kvcache.",
        )
