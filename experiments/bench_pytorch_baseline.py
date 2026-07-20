#!/usr/bin/env python3
"""PyTorch / HuggingFace baseline: stock CausalLM.generate.

Uses Transformers + plain ``nn.Linear`` / framework attention (SDPA or eager),
with no nano-vLLM continuous batching, paged KV, or CUDA-graph decode path.

Workload defaults mirror repo-root ``bench.py`` (seed, num_seqs, length ranges)
so you can compare tok/s directionally — not a perfectly fair kernel match.

Examples (Linux + CUDA + local model):

    python experiments/bench_pytorch_baseline.py
    python experiments/bench_pytorch_baseline.py --num-seqs 8 --max-input-len 128 --max-output-len 64
    python experiments/bench_pytorch_baseline.py --mode batched   # padded batch generate
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow `python experiments/bench_pytorch_baseline.py` from repo root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.common_workload import default_model_path, make_workload as shared_make_workload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=default_model_path())
    p.add_argument("--num-seqs", type=int, default=8)
    p.add_argument("--max-input-len", type=int, default=128)
    p.add_argument("--max-output-len", type=int, default=64)
    p.add_argument("--min-input-len", type=int, default=64)
    p.add_argument("--min-output-len", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--mode",
        choices=("sequential", "batched"),
        default="sequential",
        help="sequential: one request at a time (true no-batching baseline); "
        "batched: one padded batch (still no paged KV / continuous batching).",
    )
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=("sdpa", "eager", "flash_attention_2"),
        help="Transformers attention backend (stock PyTorch path is usually sdpa/eager).",
    )
    return p


@torch.inference_mode()
def run_sequential(model, prompts: list[list[int]], max_tokens: list[int], device: torch.device):
    for ids, n in zip(prompts, max_tokens):
        input_ids = torch.tensor([ids], device=device, dtype=torch.long)
        model.generate(
            input_ids=input_ids,
            max_new_tokens=n,
            do_sample=False,
            use_cache=True,
        )


@torch.inference_mode()
def run_batched(model, tokenizer, prompts: list[list[int]], max_tokens: list[int], device: torch.device):
    # One padded batch; decode length = max over requests (HF generate is not per-row max_tokens).
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    max_prompt = max(len(p) for p in prompts)
    batch = torch.full((len(prompts), max_prompt), pad_id, device=device, dtype=torch.long)
    attn = torch.zeros((len(prompts), max_prompt), device=device, dtype=torch.long)
    for i, ids in enumerate(prompts):
        batch[i, -len(ids) :] = torch.tensor(ids, device=device, dtype=torch.long)
        attn[i, -len(ids) :] = 1
    model.generate(
        input_ids=batch,
        attention_mask=attn,
        max_new_tokens=max(max_tokens),
        do_sample=False,
        use_cache=True,
        pad_token_id=pad_id,
    )


def main():
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this baseline (no CUDA device found).")
    if not os.path.isdir(args.model):
        raise SystemExit(f"Model directory not found: {args.model}")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = torch.device("cuda")

    print(
        f"baseline=hf_pytorch mode={args.mode} attn={args.attn_implementation} "
        f"dtype={args.dtype} model={args.model}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            attn_implementation=args.attn_implementation,
            trust_remote_code=True,
        ).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
            trust_remote_code=True,
        ).to(device)
    model.eval()

    prompts, max_tokens = shared_make_workload(
        args.num_seqs,
        args.min_input_len,
        args.max_input_len,
        args.min_output_len,
        args.max_output_len,
        args.seed,
    )
    total_tokens = sum(max_tokens)

    # Warmup
    warm = torch.tensor([prompts[0][:32]], device=device, dtype=torch.long)
    model.generate(input_ids=warm, max_new_tokens=8, do_sample=False, use_cache=True)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    if args.mode == "sequential":
        run_sequential(model, prompts, max_tokens, device)
    else:
        run_batched(model, tokenizer, prompts, max_tokens, device)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # sequential: each request generates exactly max_tokens[i]
    # batched: HF uses a single max_new_tokens = max(max_tokens); report both.
    if args.mode == "batched":
        billed = len(prompts) * max(max_tokens)
        print(
            f"Total (sum max_tokens): {total_tokens}tok, "
            f"Total (batched bill = N*max): {billed}tok, "
            f"Time: {elapsed:.2f}s, "
            f"Throughput (sum/max_tokens): {total_tokens / elapsed:.2f}tok/s, "
            f"Throughput (batched bill): {billed / elapsed:.2f}tok/s",
            flush=True,
        )
    else:
        print(
            f"Total: {total_tokens}tok, Time: {elapsed:.2f}s, "
            f"Throughput: {total_tokens / elapsed:.2f}tok/s",
            flush=True,
        )


if __name__ == "__main__":
    main()
