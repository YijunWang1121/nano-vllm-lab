#!/usr/bin/env python3
"""Throughput bench for nano-vLLM (matched knobs with other experiments/* benches)."""

from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common_workload import (  # noqa: E402
    apply_chat_template,
    default_model_path,
    make_chat_workload,
    make_workload,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=default_model_path())
    p.add_argument("--workload", choices=("random", "chat"), default="random")
    p.add_argument("--num-seqs", type=int, default=8)
    p.add_argument("--min-input-len", type=int, default=64)
    p.add_argument("--max-input-len", type=int, default=128)
    p.add_argument("--min-output-len", type=int, default=32)
    p.add_argument("--max-output-len", type=int, default=64)
    p.add_argument("--chat-turns", type=int, default=3, help="Multi-turn rounds for --workload chat")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default True for stable smoke; use --no-enforce-eager for CUDA graphs.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not os.path.isdir(args.model):
        raise SystemExit(f"Model directory not found: {args.model}")

    from nanovllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    if args.workload == "chat":
        messages_list, max_tokens = make_chat_workload(
            args.num_seqs,
            args.min_output_len,
            args.max_output_len,
            args.seed,
            args.chat_turns,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        prompts = apply_chat_template(tokenizer, messages_list)
    else:
        prompts, max_tokens = make_workload(
            args.num_seqs,
            args.min_input_len,
            args.max_input_len,
            args.min_output_len,
            args.max_output_len,
            args.seed,
        )

    total = sum(max_tokens)
    sps = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=n) for n in max_tokens
    ]

    print(
        f"engine=nanovllm workload={args.workload} enforce_eager={args.enforce_eager} "
        f"num_seqs={args.num_seqs} model={args.model}",
        flush=True,
    )
    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
    )
    try:
        warm = prompts[0] if isinstance(prompts[0], str) else prompts[0][:32]
        llm.generate([warm], SamplingParams(max_tokens=4), use_tqdm=False)
        t0 = time.perf_counter()
        llm.generate(prompts, sps, use_tqdm=False)
        elapsed = time.perf_counter() - t0
    finally:
        llm.exit()

    print(
        f"nanovllm Total={total}tok Time={elapsed:.2f}s "
        f"Throughput={total / elapsed:.2f}tok/s",
        flush=True,
    )


if __name__ == "__main__":
    main()
