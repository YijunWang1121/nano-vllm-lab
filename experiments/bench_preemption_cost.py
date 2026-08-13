#!/usr/bin/env python3
"""Measure how much recompute-based preemption actually costs, before
deciding whether CPU KV-cache swap is worth building.

Scheduler.preempt() (nanovllm/engine/scheduler.py) is recompute-only: when a
decoding sequence needs a new KV block and the pool is out of free blocks,
the scheduler evicts the tail of `running`, frees ALL its GPU blocks, and
re-admits it as a fresh prefill -- its token_ids (prompt + everything
generated so far) are preserved, but their KV cache is fully recomputed via
a real forward pass. This script runs the SAME decode-heavy workload twice
-- once with enough GPU memory that preemption never fires, once with
`--gpu-memory-utilization` deliberately lowered so the KV-cache pool runs
dry under concurrent long decodes -- and reports:

  - Scheduler.num_preemptions (how often it actually fires)
  - "wasted" recompute tokens: total prefill tokens processed across the
    whole run minus sum(prompt_len) across all requests -- anything beyond
    that sum is tokens re-prefilled purely because of preemption, not
    genuine first-time prompt processing
  - the wall-clock time delta between the two configs

A third run repeats the *same* stress config with Config.kv_swap_enabled=True
(CPU KV-cache swap instead of recompute-on-preemption) and reports the actual
wall-time delta against the recompute-based stress run -- proving whether
swap is actually faster in wall-clock terms, not just "avoids recompute FLOPs
in theory." Pass --no-compare-swap to skip it.

Both configs use the SAME --gpu-memory-utilization (nano-vLLM's own
default-ish 0.85) -- what's varied is *concurrency* (--baseline-num-seqs vs.
--stress-num-seqs), since that's what actually determines whether the fixed
KV-cache pool this produces is enough for the workload. On a small GPU the
pool can be modest (a few thousand token slots), so realistic-looking
concurrency can already exceed it without needing to artificially starve
gpu_memory_utilization.

Usage:
    export NANOVLLM_TEST_MODEL=~/huggingface/Qwen3-0.6B
    python experiments/bench_preemption_cost.py
    python experiments/bench_preemption_cost.py --baseline-num-seqs 16 --stress-num-seqs 40 --output-len 768
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

from experiments.common_workload import default_model_path, make_workload  # noqa: E402

RESULT_PREFIX = "RESULT_JSON:"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=default_model_path())
    p.add_argument("--baseline-num-seqs", type=int, default=16, help="concurrency that fits the KV-cache pool with no preemption")
    p.add_argument("--stress-num-seqs", type=int, default=40, help="concurrency that exceeds the pool and triggers preemption")
    p.add_argument("--input-len", type=int, default=128)
    p.add_argument("--output-len", type=int, default=768, help="long decode, so sequences cross many block boundaries")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--compare-swap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also run the stress config with Config.kv_swap_enabled=True and compare wall time",
    )
    p.add_argument("--cpu-kvcache-gib", type=float, default=1.0)
    p.add_argument(
        "--single-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=argparse.SUPPRESS,  # internal: used for subprocess-isolated single measurements
    )
    p.add_argument("--num-seqs", type=int, default=16, help=argparse.SUPPRESS)
    p.add_argument("--kv-swap-enabled", action=argparse.BooleanOptionalAction, default=False, help=argparse.SUPPRESS)
    return p


def run_once(model_path: str, num_seqs: int, input_len: int, output_len: int, gpu_mem: float, seed: int,
             kv_swap_enabled: bool = False, cpu_kvcache_gib: float = 1.0):
    from nanovllm import LLM, SamplingParams

    prompts, _ = make_workload(num_seqs, input_len, input_len, output_len, output_len, seed)
    sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len) for _ in prompts]
    prompt_tokens_total = sum(len(p) for p in prompts)

    llm = LLM(
        model_path,
        enforce_eager=True,  # keep this isolated from the CUDA-graph question; recompute happens on the eager prefill path either way
        gpu_memory_utilization=gpu_mem,
        tensor_parallel_size=1,
        kv_swap_enabled=kv_swap_enabled,
        cpu_kvcache_gib=cpu_kvcache_gib,
    )
    try:
        for prompt, sp in zip(prompts, sps):
            llm.add_request(prompt, sp)

        prefill_tokens = 0
        decode_tokens = 0
        t_start = time.perf_counter()
        while not llm.is_finished():
            _, num_tokens = llm.step()
            if num_tokens > 0:
                prefill_tokens += num_tokens
            else:
                decode_tokens += -num_tokens
        wall_time = time.perf_counter() - t_start
        num_preemptions = llm.scheduler.num_preemptions
        num_swaps = llm.scheduler.num_swaps
    finally:
        llm.exit()

    wasted_tokens = prefill_tokens - prompt_tokens_total
    return {
        "wall_time": wall_time,
        "num_preemptions": num_preemptions,
        "num_swaps": num_swaps,
        "prompt_tokens_total": prompt_tokens_total,
        "prefill_tokens_processed": prefill_tokens,
        "wasted_recompute_tokens": wasted_tokens,
        "wasted_recompute_pct": 100.0 * wasted_tokens / prefill_tokens if prefill_tokens else 0.0,
        "decode_tokens": decode_tokens,
    }


def run_isolated(args, num_seqs: int, kv_swap_enabled: bool = False) -> dict:
    """Run one measurement in a fresh subprocess -- this 6GB card doesn't
    fully release GPU memory across repeated in-process LLM() construction
    (same issue noted for bench_cuda_graph_decode.py), and the stress
    config additionally needs a clean, unshared memory budget to behave
    reproducibly."""
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--single-run",
        "--model", args.model,
        "--num-seqs", str(num_seqs),
        "--input-len", str(args.input_len),
        "--output-len", str(args.output_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--seed", str(args.seed),
        "--cpu-kvcache-gib", str(args.cpu_kvcache_gib),
        "--kv-swap-enabled" if kv_swap_enabled else "--no-kv-swap-enabled",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    raise RuntimeError(
        f"subprocess produced no result (exit={proc.returncode}):\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr[-4000:]}"
    )


def print_result(label: str, r: dict) -> None:
    print(
        f"{label:>16}: wall={r['wall_time']:7.2f}s  preemptions={r['num_preemptions']:4d}  swaps={r['num_swaps']:4d}  "
        f"prefill_tokens={r['prefill_tokens_processed']:7d} "
        f"(prompt_total={r['prompt_tokens_total']}, wasted={r['wasted_recompute_tokens']} "
        f"= {r['wasted_recompute_pct']:.1f}%)",
        flush=True,
    )


def main() -> None:
    args = build_parser().parse_args()
    if not os.path.isdir(args.model):
        raise SystemExit(f"Model directory not found: {args.model}")

    if args.single_run:
        result = run_once(
            args.model, args.num_seqs, args.input_len, args.output_len, args.gpu_memory_utilization, args.seed,
            kv_swap_enabled=args.kv_swap_enabled, cpu_kvcache_gib=args.cpu_kvcache_gib,
        )
        print(RESULT_PREFIX + json.dumps(result), flush=True)
        return

    print(
        f"model={args.model} input_len={args.input_len} output_len={args.output_len} "
        f"gpu_memory_utilization={args.gpu_memory_utilization}",
        flush=True,
    )

    baseline = run_isolated(args, args.baseline_num_seqs)
    print_result(f"baseline (n={args.baseline_num_seqs})", baseline)

    stress = run_isolated(args, args.stress_num_seqs)
    print_result(f"stress/recompute (n={args.stress_num_seqs})", stress)

    if stress["num_preemptions"] == 0:
        print(
            "\nWARNING: stress config triggered 0 preemptions -- raise "
            "--stress-num-seqs or --output-len.",
            flush=True,
        )
        return

    slowdown = stress["wall_time"] / baseline["wall_time"] if baseline["wall_time"] else float("nan")
    seq_ratio = args.stress_num_seqs / args.baseline_num_seqs
    print(
        f"\npreemption cost: {stress['num_preemptions']} preemptions, "
        f"{stress['wasted_recompute_tokens']} wasted recompute tokens "
        f"({stress['wasted_recompute_pct']:.1f}% of all prefill work), "
        f"{slowdown:.2f}x wall-time vs. {seq_ratio:.2f}x more concurrent requests",
        flush=True,
    )

    if not args.compare_swap:
        return

    swap = run_isolated(args, args.stress_num_seqs, kv_swap_enabled=True)
    print_result(f"stress/swap (n={args.stress_num_seqs})", swap)

    swap_speedup = stress["wall_time"] / swap["wall_time"] if swap["wall_time"] else float("nan")
    print(
        f"\nswap vs recompute at identical load: "
        f"{swap['num_swaps']} swaps ({swap['num_preemptions']} preemptions), "
        f"wall time {stress['wall_time']:.2f}s -> {swap['wall_time']:.2f}s "
        f"({swap_speedup:.2f}x {'faster' if swap_speedup >= 1 else 'SLOWER'})",
        flush=True,
    )


if __name__ == "__main__":
    main()
