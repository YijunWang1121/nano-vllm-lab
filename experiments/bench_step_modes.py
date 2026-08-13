#!/usr/bin/env python3
"""Compare tok/s across Config.step_mode ("separate" / "combined" / "unified")
crossed with CUDA graphs on/off, under a realistic *online-serving* workload.

Background: nano-vLLM's engine step is normally either all-prefill
(flash_attn_varlen_func) or all-decode (flash_attn_with_kvcache / CUDA
graph) -- see Scheduler.schedule(). This script measures three alternatives:

  separate  Current baseline: a step is all-prefill or all-decode, never
            both (chunk_prefill_tokens still applies within the prefill
            side if set). Supports both eager and CUDA graphs.
  combined  Both prefill and decode admitted every step, as two separate
            kernel calls (prefill via varlen, decode via
            with_kvcache/CUDA graph unchanged). Supports both eager and
            CUDA graphs (graphs apply to the decode sub-batch).
  unified   Both admitted into ONE batch through ONE flash_attn_varlen_func
            call (decode is just seqlen_q=1 prefill against cached
            context -- see Scheduler._schedule_unified). Always eager:
            run_model's `if is_prefill: eager` check means this mode can
            never hit a captured CUDA graph, so only 1 variant exists here,
            not 2 -- this script skips the graphs=on case for "unified"
            automatically (5 measurements total, not 6).

Workload: matches vLLM's own benchmark_serving.py conventions (see
experiments/common_workload.py::make_poisson_serving_workload) --
ShareGPT-like Poisson-distributed prompt/output lengths, requests arriving
as a Poisson process at --request-rate req/s rather than all submitted at
t=0. This matters specifically for this benchmark: "combined"/"unified"
exist to let new prefill work interleave with ongoing decode instead of
blocking behind it -- an upfront burst has every request start prefilling
together, so it can't exercise (or measure any benefit from) that at all.

Metric is overall throughput (total output tokens / total wall time from
the first possible arrival to the last completion).

Usage:
    export NANOVLLM_TEST_MODEL=~/huggingface/Qwen3-0.6B
    python experiments/bench_step_modes.py
    python experiments/bench_step_modes.py --num-seqs 24 --request-rate 3.0 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common_workload import default_model_path, make_poisson_serving_workload  # noqa: E402

RESULT_PREFIX = "RESULT_JSON:"
MODES = ["separate", "combined", "unified"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=default_model_path())
    p.add_argument("--num-seqs", type=int, default=20)
    p.add_argument("--mean-input-len", type=int, default=150, help="ShareGPT-ish mean; kept below vLLM's 512 default to fit this GPU/model")
    p.add_argument("--mean-output-len", type=int, default=100, help="ShareGPT-ish mean; kept below vLLM's 256 default for iteration speed")
    p.add_argument("--request-rate", type=float, default=3.0, help="requests/sec, Poisson arrival; use inf for old submit-all-at-t=0 behavior")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--chunk-prefill-tokens", type=int, default=-1, help="passed through to Config; -1 disables chunking")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--single-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=argparse.SUPPRESS,  # internal: used for subprocess-isolated single measurements
    )
    p.add_argument("--step-mode", choices=MODES, default="separate", help=argparse.SUPPRESS)
    p.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True, help=argparse.SUPPRESS)
    return p


def run_once(model_path, num_seqs, mean_input_len, mean_output_len, request_rate, gpu_mem, seed,
             step_mode, enforce_eager, chunk_prefill_tokens):
    from nanovllm import LLM, SamplingParams

    requests = make_poisson_serving_workload(num_seqs, mean_input_len, mean_output_len, request_rate, seed)
    total_output_tokens = sum(max_tokens for _, max_tokens, _ in requests)

    llm = LLM(
        model_path,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_mem,
        tensor_parallel_size=1,
        step_mode=step_mode,
        chunk_prefill_tokens=chunk_prefill_tokens,
    )
    try:
        next_idx = 0
        t0 = time.perf_counter()
        while next_idx < len(requests) or not llm.is_finished():
            now = time.perf_counter() - t0
            while next_idx < len(requests) and requests[next_idx][2] <= now:
                prompt, max_tokens, _ = requests[next_idx]
                sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_tokens)
                llm.add_request(prompt, sp)
                next_idx += 1
            if llm.is_finished():
                # Nothing in flight; idle until the next scheduled arrival
                # rather than busy-spinning the Python loop for no reason.
                if next_idx < len(requests):
                    sleep_s = requests[next_idx][2] - (time.perf_counter() - t0)
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                continue
            llm.step()
        wall_time = time.perf_counter() - t0
    finally:
        llm.exit()

    return {"wall_time": wall_time, "total_output_tokens": total_output_tokens, "tok_s": total_output_tokens / wall_time}


def run_isolated(args, step_mode: str, enforce_eager: bool) -> dict:
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--single-run",
        "--step-mode", step_mode,
        "--enforce-eager" if enforce_eager else "--no-enforce-eager",
        "--model", args.model,
        "--num-seqs", str(args.num_seqs),
        "--mean-input-len", str(args.mean_input_len),
        "--mean-output-len", str(args.mean_output_len),
        "--request-rate", str(args.request_rate),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--chunk-prefill-tokens", str(args.chunk_prefill_tokens),
        "--seed", str(args.seed),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
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
        result = run_once(
            args.model, args.num_seqs, args.mean_input_len, args.mean_output_len, args.request_rate,
            args.gpu_memory_utilization, args.seed, args.step_mode, args.enforce_eager, args.chunk_prefill_tokens,
        )
        print(RESULT_PREFIX + json.dumps(result), flush=True)
        return

    print(
        f"model={args.model} num_seqs={args.num_seqs} mean_input_len={args.mean_input_len} "
        f"mean_output_len={args.mean_output_len} request_rate={args.request_rate}req/s "
        f"repeats={args.repeats} chunk_prefill_tokens={args.chunk_prefill_tokens}",
        flush=True,
    )

    configs = [
        ("separate", True), ("separate", False),
        ("combined", True), ("combined", False),
        ("unified", True),  # unified + graphs=off skipped: run_model forces eager whenever is_prefill=True, which unified always passes
    ]

    results = {}
    for step_mode, enforce_eager in configs:
        label = f"{step_mode}/{'eager' if enforce_eager else 'graph'}"
        samples = [run_isolated(args, step_mode, enforce_eager) for _ in range(args.repeats)]
        tok_s = [s["tok_s"] for s in samples]
        mean = statistics.mean(tok_s)
        std = statistics.stdev(tok_s) if len(tok_s) > 1 else 0.0
        results[label] = mean
        print(f"{label:>16}: {mean:8.1f} tok/s (std={std:6.1f})  n={len(samples)}", flush=True)

    baseline = results.get("separate/eager")
    print("\nspeedup vs separate/eager:", flush=True)
    for label, mean in results.items():
        if baseline:
            print(f"  {label:>16}: {mean / baseline:.2f}x", flush=True)


if __name__ == "__main__":
    main()
