#!/usr/bin/env python3
"""Measure the chunk_prefill_tokens tradeoff: short-request latency vs.
long-request throughput, comparing "separate" (prefill-priority baseline)
against "unified" (decode-priority: existing decode always served first,
chunked prefill only gets whatever max_num_batched_tokens budget is left --
see Scheduler._schedule_unified) step modes.

Workload: ONE long prompt + many short prompts submitted together (all
already queued behind the long one, not staggered arrival -- this
specifically targets "who gets admitted first out of a fixed backlog", the
scenario chunk_prefill_tokens exists for). --num-short/--gpu-memory-utilization
are set to also create real KV-cache pressure so eviction/swap counts are
non-trivial, not just always 0. Config.kv_swap_enabled=True throughout.

Per (chunk_prefill_tokens, step_mode) cell, records:
  - short_ttft_mean_ms: mean time-to-first-token across the short requests
  - long_ttft_ms: time-to-first-token for the long request
  - long_total_s: long request's total completion time (prefill + decode)
  - batch_wall_s: wall time for the entire batch (long + all short) to finish
  - tok_s: total output tokens / batch_wall_s
  - num_preemptions / num_swaps / num_recomputes (= preemptions - swaps)

Usage:
    export NANOVLLM_TEST_MODEL=~/huggingface/Qwen3-0.6B
    python experiments/bench_chunk_prefill_tradeoff.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common_workload import default_model_path  # noqa: E402

RESULT_PREFIX = "RESULT_JSON:"
CHUNK_VALUES = [256, 1024, 2048, 4096, -1]
STEP_MODES = ["separate", "unified"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=default_model_path())
    p.add_argument("--long-len", type=int, default=3000)
    p.add_argument("--long-max-tokens", type=int, default=40)
    p.add_argument("--num-short", type=int, default=40)
    p.add_argument("--short-len", type=int, default=100)
    p.add_argument("--short-max-tokens", type=int, default=300, help="needs to be long enough that short seqs hold blocks concurrently long enough to actually contend for the pool -- a fast-finishing short workload never builds up peak pressure regardless of count")
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.6, help="calibrated so this workload's ~33-block pool is smaller than long(12 blocks)+short(<=1 block each) worst-case demand")
    p.add_argument("--cpu-kvcache-gib", type=float, default=1.0)
    p.add_argument("--kv-swap-min-tokens", type=int, default=4)
    p.add_argument(
        "--kv-swap-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="preemption strategy: swap (default) or recompute (--no-kv-swap-enabled)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--single-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=argparse.SUPPRESS,  # internal: used for subprocess-isolated single measurements
    )
    p.add_argument("--chunk-prefill-tokens", type=int, default=-1, help=argparse.SUPPRESS)
    p.add_argument("--step-mode", choices=STEP_MODES, default="separate", help=argparse.SUPPRESS)
    return p


def run_once(args, chunk_prefill_tokens: int, step_mode: str) -> dict:
    from nanovllm import LLM, SamplingParams

    rng_seed = args.seed
    import random
    rng = random.Random(rng_seed)
    long_prompt = [rng.randint(0, 10000) for _ in range(args.long_len)]
    short_prompts = [[rng.randint(0, 10000) for _ in range(args.short_len)] for _ in range(args.num_short)]

    llm = LLM(
        args.model,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        max_num_batched_tokens=args.max_num_batched_tokens,
        chunk_prefill_tokens=chunk_prefill_tokens,
        step_mode=step_mode,
        kv_swap_enabled=args.kv_swap_enabled,
        kv_swap_min_tokens=args.kv_swap_min_tokens,
        cpu_kvcache_gib=args.cpu_kvcache_gib,
    )
    try:
        long_sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.long_max_tokens)
        llm.add_request(long_prompt, long_sp)
        long_id = llm.scheduler.waiting[-1].seq_id

        short_sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.short_max_tokens)
        short_ids = []
        for prompt in short_prompts:
            llm.add_request(prompt, short_sp)
            short_ids.append(llm.scheduler.waiting[-1].seq_id)

        seq_by_id = {s.seq_id: s for s in list(llm.scheduler.waiting)}
        total_output_tokens = args.long_max_tokens + args.num_short * args.short_max_tokens

        ttft = {}  # seq_id -> wall time of first sampled token
        t0 = time.perf_counter()
        while not llm.is_finished():
            llm.step()
            now = time.perf_counter() - t0
            for sid in [long_id] + short_ids:
                if sid not in ttft:
                    seq = seq_by_id.get(sid)
                    if seq is not None and seq.num_completion_tokens >= 1:
                        ttft[sid] = now
        batch_wall_s = time.perf_counter() - t0

        long_total_s = ttft.get(long_id, batch_wall_s)  # fallback shouldn't trigger; long always gets >=1 token
        short_ttfts = [ttft[sid] for sid in short_ids if sid in ttft]

        num_preemptions = llm.scheduler.num_preemptions
        num_swaps = llm.scheduler.num_swaps
    finally:
        llm.exit()

    return {
        "short_ttft_mean_ms": 1000 * sum(short_ttfts) / len(short_ttfts) if short_ttfts else None,
        "long_ttft_ms": 1000 * ttft.get(long_id, float("nan")),
        "long_total_s": long_total_s,
        "batch_wall_s": batch_wall_s,
        "tok_s": total_output_tokens / batch_wall_s,
        "num_preemptions": num_preemptions,
        "num_swaps": num_swaps,
        "num_recomputes": num_preemptions - num_swaps,
    }


def run_isolated(args, chunk_prefill_tokens: int, step_mode: str) -> dict:
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--single-run",
        "--chunk-prefill-tokens", str(chunk_prefill_tokens),
        "--step-mode", step_mode,
        "--model", args.model,
        "--long-len", str(args.long_len),
        "--long-max-tokens", str(args.long_max_tokens),
        "--num-short", str(args.num_short),
        "--short-len", str(args.short_len),
        "--short-max-tokens", str(args.short_max_tokens),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--cpu-kvcache-gib", str(args.cpu_kvcache_gib),
        "--kv-swap-min-tokens", str(args.kv_swap_min_tokens),
        "--kv-swap-enabled" if args.kv_swap_enabled else "--no-kv-swap-enabled",
        "--seed", str(args.seed),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    raise RuntimeError(
        f"subprocess produced no result (exit={proc.returncode}):\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr[-4000:]}"
    )


def main() -> None:
    args = build_parser().parse_args()
    if not os.path.isdir(args.model):
        raise SystemExit(f"Model directory not found: {args.model}")

    if args.single_run:
        result = run_once(args, args.chunk_prefill_tokens, args.step_mode)
        print(RESULT_PREFIX + json.dumps(result), flush=True)
        return

    print(
        f"model={args.model} long_len={args.long_len} num_short={args.num_short} short_len={args.short_len} "
        f"gpu_memory_utilization={args.gpu_memory_utilization} "
        f"preemption_strategy={'swap' if args.kv_swap_enabled else 'recompute'}",
        flush=True,
    )
    header = (
        f"{'chunk':>7} {'mode':>9} {'short_ttft':>11} {'long_ttft':>10} {'long_total':>11} "
        f"{'batch_wall':>11} {'tok/s':>8} {'evict':>6} {'swap':>5} {'recompute':>9}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for chunk in CHUNK_VALUES:
        for step_mode in STEP_MODES:
            r = run_isolated(args, chunk, step_mode)
            short_ttft = f"{r['short_ttft_mean_ms']:.0f}ms" if r["short_ttft_mean_ms"] is not None else "n/a"
            print(
                f"{chunk!s:>7} {step_mode:>9} {short_ttft:>11} {r['long_ttft_ms']:>9.0f}ms "
                f"{r['long_total_s']:>10.2f}s {r['batch_wall_s']:>10.2f}s {r['tok_s']:>8.1f} "
                f"{r['num_preemptions']:>6d} {r['num_swaps']:>5d} {r['num_recomputes']:>9d}",
                flush=True,
            )


if __name__ == "__main__":
    main()
