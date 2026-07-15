#!/usr/bin/env python3
"""Print how large the paged KV cache pool is on this GPU/model.

KV size is not a fixed constant: after model load + warmup, leftover GPU memory
(scaled by gpu_memory_utilization) is carved into blocks of kvcache_block_size
tokens (default 256).

    num_kvcache_blocks ≈ (total*util - model_resident) // bytes_per_block
    kv_pool_tokens     = num_kvcache_blocks * block_size

Per-sequence length is still capped by max_model_len (default 4096, also clamped
to the HF config max_position_embeddings).
"""

from __future__ import annotations

import argparse
import gc
import os

from nanovllm import LLM


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default=os.path.expanduser(os.environ.get("NANOVLLM_TEST_MODEL", "~/huggingface/Qwen3-0.6B/")),
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--enforce-eager", action="store_true", default=True)
    args = p.parse_args()
    model = os.path.expanduser(args.model)

    llm = LLM(
        model,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=1,
    )
    try:
        cfg = llm.model_runner.config
        hf = cfg.hf_config
        block_size = cfg.kvcache_block_size
        n_blocks = cfg.num_kvcache_blocks
        pool_tokens = n_blocks * block_size
        per_seq_blocks = (cfg.max_model_len + block_size - 1) // block_size
        print(f"model:                 {model}")
        print(f"dtype:                 {hf.dtype}")
        print(f"layers:                {hf.num_hidden_layers}")
        print(f"kv_heads (total):      {hf.num_key_value_heads}")
        print(f"gpu_memory_utilization:{cfg.gpu_memory_utilization}")
        print(f"kvcache_block_size:    {block_size}")
        print(f"num_kvcache_blocks:    {n_blocks}")
        print(f"kv_pool_tokens:        {pool_tokens}  (= blocks * block_size)")
        print(f"max_model_len:         {cfg.max_model_len}  (per-sequence cap)")
        print(f"approx_full_len_seqs:  {n_blocks // per_seq_blocks}")
        print()
        print("Notes:")
        print("- Raising gpu_memory_utilization or using a smaller model increases blocks.")
        print("- Raising max_model_len does NOT grow the pool; it only raises the per-seq cap")
        print("  and CUDA-graph block-table width (and warmup cost).")
        print("- Prefix cache shares blocks by hash; pool size is still the hard limit.")
    finally:
        llm.exit()
        gc.collect()


if __name__ == "__main__":
    main()
