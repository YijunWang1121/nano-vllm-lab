#!/usr/bin/env python3
"""PyTorch / HuggingFace baseline: stock CausalLM.generate.

Uses Transformers + plain ``nn.Linear`` / framework attention (SDPA or eager),
with no nano-vLLM continuous batching, paged KV, or CUDA-graph decode path.

Examples (Linux + CUDA + local model):

    python experiments/bench_pytorch_baseline.py
    python experiments/bench_pytorch_baseline.py --workload chat --chat-turns 3
    python experiments/bench_pytorch_baseline.py --mode batched
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.common_workload import (  # noqa: E402
    apply_chat_template,
    default_model_path,
    make_chat_workload,
    make_workload,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=default_model_path())
    p.add_argument("--workload", choices=("random", "chat"), default="random")
    p.add_argument("--num-seqs", type=int, default=8)
    p.add_argument("--max-input-len", type=int, default=128)
    p.add_argument("--max-output-len", type=int, default=64)
    p.add_argument("--min-input-len", type=int, default=64)
    p.add_argument("--min-output-len", type=int, default=32)
    p.add_argument("--chat-turns", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--mode",
        choices=("sequential", "batched"),
        default="sequential",
    )
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=("sdpa", "eager", "flash_attention_2"),
    )
    return p


@torch.inference_mode()
def run_sequential_token(model, prompts: list[list[int]], max_tokens: list[int], device: torch.device):
    for ids, n in zip(prompts, max_tokens):
        input_ids = torch.tensor([ids], device=device, dtype=torch.long)
        model.generate(
            input_ids=input_ids,
            max_new_tokens=n,
            do_sample=False,
            use_cache=True,
        )


@torch.inference_mode()
def run_sequential_text(model, tokenizer, prompts: list[str], max_tokens: list[int], device: torch.device):
    for text, n in zip(prompts, max_tokens):
        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=n,
            do_sample=False,
            use_cache=True,
        )


@torch.inference_mode()
def run_batched_token(model, tokenizer, prompts: list[list[int]], max_tokens: list[int], device: torch.device):
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


@torch.inference_mode()
def run_batched_text(model, tokenizer, prompts: list[str], max_tokens: list[int], device: torch.device):
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
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
        f"baseline=hf_pytorch workload={args.workload} mode={args.mode} "
        f"attn={args.attn_implementation} dtype={args.dtype} model={args.model}",
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

    if args.workload == "chat":
        messages_list, max_tokens = make_chat_workload(
            args.num_seqs,
            args.min_output_len,
            args.max_output_len,
            args.seed,
            args.chat_turns,
        )
        text_prompts = apply_chat_template(tokenizer, messages_list)
        token_prompts = None
    else:
        token_prompts, max_tokens = make_workload(
            args.num_seqs,
            args.min_input_len,
            args.max_input_len,
            args.min_output_len,
            args.max_output_len,
            args.seed,
        )
        text_prompts = None

    total_tokens = sum(max_tokens)

    if args.workload == "chat":
        warm = tokenizer(text_prompts[0], return_tensors="pt")
        warm_ids = warm["input_ids"].to(device)
        warm_attn = warm.get("attention_mask")
        if warm_attn is not None:
            warm_attn = warm_attn.to(device)
        model.generate(
            input_ids=warm_ids,
            attention_mask=warm_attn,
            max_new_tokens=8,
            do_sample=False,
            use_cache=True,
        )
    else:
        warm = torch.tensor([token_prompts[0][:32]], device=device, dtype=torch.long)
        model.generate(input_ids=warm, max_new_tokens=8, do_sample=False, use_cache=True)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    if args.workload == "chat":
        if args.mode == "sequential":
            run_sequential_text(model, tokenizer, text_prompts, max_tokens, device)
        else:
            run_batched_text(model, tokenizer, text_prompts, max_tokens, device)
    elif args.mode == "sequential":
        run_sequential_token(model, token_prompts, max_tokens, device)
    else:
        run_batched_token(model, tokenizer, token_prompts, max_tokens, device)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    if args.mode == "batched":
        billed = len(max_tokens) * max(max_tokens)
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
