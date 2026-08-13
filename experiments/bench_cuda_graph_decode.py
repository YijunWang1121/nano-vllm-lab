#!/usr/bin/env python3
"""Isolate decode-phase throughput with vs without CUDA graphs.

CUDA graph replay in this engine only ever applies to decode steps --
prefill always takes the eager path regardless of --enforce-eager (see
ModelRunner.run_model: `if is_prefill or self.enforce_eager or ...: eager`).
So a *total* throughput bench (like bench_nanovllm.py) mixes in prefill time
and undercounts the CUDA graph effect. This script drives the engine at the
add_request()/step() level so it can separately time prefill steps
(num_tokens > 0) from decode steps (num_tokens < 0, magnitude = seqs
decoded that step) and report decode-only tok/s.

Usage:
    export NANOVLLM_TEST_MODEL=~/huggingface/Qwen3-0.6B
    python experiments/bench_cuda_graph_decode.py
    python experiments/bench_cuda_graph_decode.py --num-seqs 8 --output-len 256 --repeats 3
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

from experiments.common_workload import default_model_path, make_workload  # noqa: E402

RESULT_PREFIX = "RESULT_JSON:"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=default_model_path())
    p.add_argument("--num-seqs", type=int, default=8)
    p.add_argument("--input-len", type=int, default=128)
    p.add_argument("--output-len", type=int, default=128, help="decode length; ignore_eos so every run is identical")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--repeats", type=int, default=3, help="repeats per configuration, reports mean +/- std")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--single-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=argparse.SUPPRESS,  # internal: used for subprocess-isolated single measurements
    )
    p.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True, help=argparse.SUPPRESS)
    return p


def run_once(model_path: str, enforce_eager: bool, num_seqs: int, input_len: int, output_len: int, gpu_mem: float, seed: int):
    from nanovllm import LLM, SamplingParams

    prompts, _ = make_workload(num_seqs, input_len, input_len, output_len, output_len, seed)
    sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len) for _ in prompts]

    llm = LLM(
        model_path,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_mem,
        tensor_parallel_size=1,
    )
    try:
        # Warm up (also captures CUDA graphs at LLM() construction time already;
        # this extra call warms the Python-side request/schedule path too).
        llm.generate(["hi"], SamplingParams(max_tokens=4), use_tqdm=False)

        for prompt, sp in zip(prompts, sps):
            llm.add_request(prompt, sp)

        prefill_time = 0.0
        decode_time = 0.0
        prefill_tokens = 0
        decode_tokens = 0
        while not llm.is_finished():
            t0 = time.perf_counter()
            _, num_tokens = llm.step()
            dt = time.perf_counter() - t0
            if num_tokens > 0:
                prefill_time += dt
                prefill_tokens += num_tokens
            else:
                decode_time += dt
                decode_tokens += -num_tokens
    finally:
        llm.exit()

    return {
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "prefill_tok_s": prefill_tokens / prefill_time if prefill_time else 0.0,
        "decode_tok_s": decode_tokens / decode_time if decode_time else 0.0,
    }


def summarize(label: str, samples: list[dict]) -> dict:
    tok_s = [s["decode_tok_s"] for s in samples]
    mean = statistics.mean(tok_s)
    std = statistics.stdev(tok_s) if len(tok_s) > 1 else 0.0
    prefill_mean = statistics.mean(s["prefill_tok_s"] for s in samples)
    print(
        f"{label:>18}: decode={mean:8.1f} tok/s (std={std:6.1f})  "
        f"[prefill={prefill_mean:8.1f} tok/s, not counted in decode]  "
        f"decode_tokens={samples[0]['decode_tokens']}  n={len(samples)}",
        flush=True,
    )
    return {"mean": mean, "std": std}


def run_isolated(args, enforce_eager: bool) -> dict:
    """Run one measurement in a fresh subprocess.

    Constructing several LLM()s back-to-back in one process on a small GPU
    can exhaust memory (this card is 6GB) since freed CUDA memory doesn't
    always fully return between constructions -- same class of issue as the
    HF-baseline-after-nano OOM noted in CLAUDE.md. A subprocess per
    measurement guarantees a clean slate every time.
    """
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--single-run",
        "--enforce-eager" if enforce_eager else "--no-enforce-eager",
        "--model", args.model,
        "--num-seqs", str(args.num_seqs),
        "--input-len", str(args.input_len),
        "--output-len", str(args.output_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
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
        # Internal mode: one isolated measurement, machine-readable result on stdout.
        result = run_once(
            args.model, args.enforce_eager, args.num_seqs, args.input_len,
            args.output_len, args.gpu_memory_utilization, args.seed,
        )
        print(RESULT_PREFIX + json.dumps(result), flush=True)
        return

    print(
        f"model={args.model} num_seqs={args.num_seqs} input_len={args.input_len} "
        f"output_len={args.output_len} repeats={args.repeats}",
        flush=True,
    )

    eager_samples = [run_isolated(args, enforce_eager=True) for _ in range(args.repeats)]
    eager = summarize("eager", eager_samples)

    graph_samples = [run_isolated(args, enforce_eager=False) for _ in range(args.repeats)]
    graphs = summarize("cuda graphs", graph_samples)

    speedup = graphs["mean"] / eager["mean"] if eager["mean"] else float("nan")
    print(f"\ndecode-phase speedup from CUDA graphs: {speedup:.2f}x", flush=True)


if __name__ == "__main__":
    main()
