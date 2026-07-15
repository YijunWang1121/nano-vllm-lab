import importlib
import os

import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl

from nanovllm.utils.context import get_context
from nanovllm.utils.debug import debug_enabled, debug_log

# (varlen_func, with_kvcache_func, source_module_name) once resolved.
_FLASH_IMPL: tuple | None = None
_FLASH_RESOLVED = False


def _try_import_flash(module_name: str):
    mod = importlib.import_module(module_name)
    return (
        getattr(mod, "flash_attn_varlen_func"),
        getattr(mod, "flash_attn_with_kvcache"),
        module_name,
    )


def _resolve_flash_impl():
    """Resolve flash attention implementation.

    - flash / auto: pip `flash_attn`
    - vllm_flash: vLLM's bundled `vllm_flash_attn` (same .so as vLLM FLASH_ATTN backend)
    """
    global _FLASH_IMPL, _FLASH_RESOLVED
    if _FLASH_RESOLVED:
        return _FLASH_IMPL
    _FLASH_RESOLVED = True
    env = os.environ.get("NANOVLLM_ATTN_BACKEND", "auto").strip().lower()
    candidates: list[str] = []
    if env in ("vllm_flash", "vllm-flash", "same"):
        candidates = ["vllm.vllm_flash_attn", "vllm_flash_attn"]
    elif env in ("flash", "auto", ""):
        candidates = ["flash_attn"]
        if env == "auto":
            # Prefer pip flash_attn; fall back to vLLM bundle if present.
            candidates += ["vllm.vllm_flash_attn", "vllm_flash_attn"]
    else:
        # torch/sdpa/eager → no flash
        _FLASH_IMPL = None
        return None
    errors = []
    for name in candidates:
        try:
            _FLASH_IMPL = _try_import_flash(name)
            return _FLASH_IMPL
        except Exception as exc:  # ImportError / ABI mismatch
            errors.append(f"{name}: {exc}")
            continue
    _FLASH_IMPL = None
    if env in ("flash", "vllm_flash", "vllm-flash", "same"):
        raise ImportError(
            f"NANOVLLM_ATTN_BACKEND={env} but no flash impl importable. Tried: "
            + "; ".join(errors)
        )
    return None


def flash_attn_source() -> str | None:
    impl = _resolve_flash_impl()
    return None if impl is None else impl[2]


def _attn_backend() -> str:
    """flash | torch. Default: flash if importable, else torch SDPA."""
    env = os.environ.get("NANOVLLM_ATTN_BACKEND", "auto").strip().lower()
    if env in ("torch", "sdpa", "eager"):
        return "torch"
    if _resolve_flash_impl() is not None:
        return "flash"
    return "torch"


def _flash_ops():
    impl = _resolve_flash_impl()
    if impl is None:
        raise ImportError("flash attention ops requested but not available")
    return impl[0], impl[1]


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


def _gather_seq_kv(k_cache, v_cache, block_table_row, seqlen: int, block_size: int):
    """Gather contiguous K/V for one sequence from paged cache."""
    if seqlen <= 0:
        empty = k_cache.new_empty(0, k_cache.shape[2], k_cache.shape[3])
        return empty, empty
    num_blocks = (seqlen + block_size - 1) // block_size
    pieces_k = []
    pieces_v = []
    remaining = seqlen
    for b in range(num_blocks):
        bid = int(block_table_row[b].item())
        n = min(block_size, remaining)
        pieces_k.append(k_cache[bid, :n])
        pieces_v.append(v_cache[bid, :n])
        remaining -= n
    return torch.cat(pieces_k, dim=0), torch.cat(pieces_v, dim=0)


def _sdpa_prefill_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, scale, num_heads, num_kv_heads):
    """Packed prefill without paged block_table (dense K/V)."""
    outs = []
    n_rep = num_heads // num_kv_heads
    n = cu_seqlens_q.numel() - 1
    for i in range(n):
        qs, qe = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        ks, ke = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
        qi = q[qs:qe].transpose(0, 1).unsqueeze(0)  # [1, H, Sq, D]
        ki = k[ks:ke].transpose(0, 1).unsqueeze(0)  # [1, Hkv, Sk, D]
        vi = v[ks:ke].transpose(0, 1).unsqueeze(0)
        if n_rep > 1:
            ki = ki.repeat_interleave(n_rep, dim=1)
            vi = vi.repeat_interleave(n_rep, dim=1)
        oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=True, scale=scale)
        outs.append(oi.squeeze(0).transpose(0, 1))
    return torch.cat(outs, dim=0)


def _sdpa_prefill_paged(
    q, k_cache, v_cache, cu_seqlens_q, cu_seqlens_k, block_tables, scale, num_heads, num_kv_heads
):
    """Prefill when K/V live in paged cache (prefix-cache path)."""
    outs = []
    n_rep = num_heads // num_kv_heads
    block_size = k_cache.shape[1]
    n = cu_seqlens_q.numel() - 1
    for i in range(n):
        qs, qe = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        klen = int(cu_seqlens_k[i + 1] - cu_seqlens_k[i])
        qi = q[qs:qe].transpose(0, 1).unsqueeze(0)
        ki, vi = _gather_seq_kv(k_cache, v_cache, block_tables[i], klen, block_size)
        ki = ki.transpose(0, 1).unsqueeze(0)
        vi = vi.transpose(0, 1).unsqueeze(0)
        if n_rep > 1:
            ki = ki.repeat_interleave(n_rep, dim=1)
            vi = vi.repeat_interleave(n_rep, dim=1)
        sq, sk = qi.shape[2], ki.shape[2]
        if sq == sk:
            oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=True, scale=scale)
        else:
            # Queries are the suffix of the full key sequence.
            abs_q = torch.arange(sk - sq, sk, device=qi.device)[:, None]
            abs_k = torch.arange(sk, device=qi.device)[None, :]
            mask = abs_k <= abs_q
            oi = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=mask, scale=scale)
        outs.append(oi.squeeze(0).transpose(0, 1))
    return torch.cat(outs, dim=0)


def _sdpa_decode(q, k_cache, v_cache, context_lens, block_tables, scale, num_heads, num_kv_heads):
    """Decode with paged KV: q is [B, H, D]."""
    outs = []
    n_rep = num_heads // num_kv_heads
    block_size = k_cache.shape[1]
    bsz = q.shape[0]
    for i in range(bsz):
        sl = int(context_lens[i].item())
        ki, vi = _gather_seq_kv(k_cache, v_cache, block_tables[i], sl, block_size)
        qi = q[i : i + 1].unsqueeze(2)  # [1, H, 1, D]
        ki = ki.transpose(0, 1).unsqueeze(0)
        vi = vi.transpose(0, 1).unsqueeze(0)
        if n_rep > 1:
            ki = ki.repeat_interleave(n_rep, dim=1)
            vi = vi.repeat_interleave(n_rep, dim=1)
        oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=False, scale=scale)
        outs.append(oi.squeeze(2).squeeze(0))
    return torch.stack(outs, dim=0)


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
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        backend = _attn_backend()
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            debug_log(
                "attention",
                "prefill",
                backend=backend,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                use_block_table=context.block_tables is not None,
            )
            if backend == "flash":
                flash_attn_varlen_func, _ = _flash_ops()
                # vllm_flash_attn requires seqused_k whenever block_table is set (prefix-cache prefill).
                seqused_k = None
                if context.block_tables is not None and context.cu_seqlens_k is not None:
                    seqused_k = context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]
                fa_kwargs = dict(
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=context.block_tables,
                )
                if seqused_k is not None:
                    fa_kwargs["seqused_k"] = seqused_k
                try:
                    o = flash_attn_varlen_func(q, k, v, **fa_kwargs)
                except TypeError:
                    # Older pip flash-attn may not accept seqused_k.
                    fa_kwargs.pop("seqused_k", None)
                    o = flash_attn_varlen_func(q, k, v, **fa_kwargs)
            elif context.block_tables is not None:
                o = _sdpa_prefill_paged(
                    q, k_cache, v_cache,
                    context.cu_seqlens_q, context.cu_seqlens_k, context.block_tables,
                    self.scale, self.num_heads, self.num_kv_heads,
                )
            else:
                o = _sdpa_prefill_varlen(
                    q, k, v, context.cu_seqlens_q, context.cu_seqlens_k,
                    self.scale, self.num_heads, self.num_kv_heads,
                )
        else:    # decode
            # Never call .tolist() on CUDA tensors here: argument evaluation
            # happens even when debug is off, and CUDA graph capture forbids it.
            if debug_enabled("attention"):
                cl = context.context_lens
                debug_log(
                    "attention",
                    "decode",
                    backend=backend,
                    context_lens_shape=None if cl is None else tuple(cl.shape),
                )
            if backend == "flash":
                _, flash_attn_with_kvcache = _flash_ops()
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True)
            else:
                o = _sdpa_decode(
                    q, k_cache, v_cache,
                    context.context_lens, context.block_tables,
                    self.scale, self.num_heads, self.num_kv_heads,
                )
        return o
