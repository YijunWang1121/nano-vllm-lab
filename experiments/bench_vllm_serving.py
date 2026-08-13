#!/usr/bin/env python3
"""Same heavy multi-turn chat serving workload as bench_prefix_cache_serving.py,
run on production vLLM for an apples-to-apples comparison.

Mirrors the nano-vllm benchmark exactly: same system prompt construction,
same turn-wave structure (stateless-server style: every turn re-sends the
full chat-templated history), same sampling params, same metrics (TTFT,
TPOT, throughput, GPU utilization via nvidia-smi sampling), and the same
enable_prefix_caching ON/OFF A/B with each config subprocess-isolated.

vLLM-specific notes:
  - Uses the low-level LLMEngine add_request/step loop (same shape as our
    llm.step() loop) so TTFT is observed directly, not inferred.
  - Prefix-cache hit rate / preemption counts are internal to vLLM; we
    extract them best-effort from its Prometheus metrics when the running
    vLLM version exposes them, else report null. TTFT/TPOT/throughput/GPU
    util are always measured directly and never depend on vLLM internals.

Run with the dedicated venv (vLLM is deliberately NOT installed into the
main environment to avoid disturbing its torch/flash-attn pinning):

    ~/venv-vllm/bin/python experiments/bench_vllm_serving.py \
        --num-convs 32 --turns 4 --system-len 1024 --max-tokens 128 \
        --gpu-memory-utilization 0.85 --no-enforce-eager
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

from experiments.common_workload import default_model_path  # noqa: E402
from experiments.bench_prefix_cache_serving import (  # noqa: E402
    GpuMonitor,
    USER_TURNS,
    build_system_prompt,
)

RESULT_PREFIX = "RESULT_JSON:"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=default_model_path())
    p.add_argument("--num-convs", type=int, default=32)
    p.add_argument("--turns", type=int, default=4)
    p.add_argument("--system-len", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--gpu-poll-s", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--single-run", action=argparse.BooleanOptionalAction, default=False, help=argparse.SUPPRESS)
    p.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True, help=argparse.SUPPRESS)
    return p


def _try_vllm_metrics(engine) -> dict:
    """Best-effort extraction of prefix-cache/preemption counters; returns
    nulls when this vLLM version doesn't expose them here."""
    out = {"vllm_prefix_cache_hit_rate": None, "vllm_num_preemptions": None}
    try:
        metrics = engine.get_metrics()  # newer V1 API; raises/absent elsewhere
    except Exception:
        return out
    try:
        queries = hits = preempt = None
        for m in metrics:
            name = getattr(m, "name", "")
            value = getattr(m, "value", None)
            if name.endswith("prefix_cache_queries"):
                queries = value
            elif name.endswith("prefix_cache_hits"):
                hits = value
            elif name.endswith("num_preemptions") or name.endswith("preemptions_total"):
                preempt = value
        if queries and hits is not None:
            out["vllm_prefix_cache_hit_rate"] = hits / queries
        if preempt is not None:
            out["vllm_num_preemptions"] = int(preempt)
    except Exception:
        pass
    return out


def run_once(args) -> dict:
    from transformers import AutoTokenizer
    from vllm import EngineArgs, LLMEngine, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        enable_prefix_caching=args.enable_prefix_caching,
        disable_log_stats=False,
    ))

    system_prompt = build_system_prompt(tokenizer, args.system_len)
    convs = [[{"role": "system", "content": system_prompt}] for _ in range(args.num_convs)]
    sp = SamplingParams(temperature=0.6, max_tokens=args.max_tokens)

    # Warmup (weights paging, graph capture already done at init in vLLM)
    engine.add_request("warmup", "hello", SamplingParams(max_tokens=4))
    while engine.has_unfinished_requests():
        engine.step()

    ttfts: list[float] = []
    tpots: list[float] = []
    turn_rows = []
    total_out_tokens = 0
    next_id = 0
    t_bench0 = time.perf_counter()

    with GpuMonitor(args.gpu_poll_s) as gpu:
        for turn in range(args.turns):
            for i, conv in enumerate(convs):
                conv.append({"role": "user", "content": USER_TURNS[(turn + i) % len(USER_TURNS)]})
            prompts = [
                tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                for conv in convs
            ]
            prompt_lens = [len(tokenizer.encode(p)) for p in prompts]

            t0 = time.perf_counter()
            conv_of: dict[str, int] = {}
            for i, prompt in enumerate(prompts):
                rid = str(next_id)
                next_id += 1
                engine.add_request(rid, prompt, sp)
                conv_of[rid] = i

            first_tok: dict[str, float] = {}
            finish: dict[str, float] = {}
            out_tokens: dict[str, list[int]] = {}
            while engine.has_unfinished_requests():
                step_outputs = engine.step()
                now = time.perf_counter() - t0
                for ro in step_outputs:
                    rid = ro.request_id
                    if rid not in conv_of:
                        continue
                    n_out = len(ro.outputs[0].token_ids) if ro.outputs else 0
                    if n_out >= 1 and rid not in first_tok:
                        first_tok[rid] = now
                    if ro.finished:
                        finish[rid] = now
                        out_tokens[rid] = list(ro.outputs[0].token_ids)

            turn_wall = time.perf_counter() - t0
            turn_ttfts = []
            for rid, idx in conv_of.items():
                ids = out_tokens.get(rid, [])
                total_out_tokens += len(ids)
                if rid in first_tok:
                    turn_ttfts.append(first_tok[rid])
                    ttfts.append(first_tok[rid])
                    if rid in finish and len(ids) > 1:
                        tpots.append((finish[rid] - first_tok[rid]) / (len(ids) - 1))
                reply = tokenizer.decode(ids, skip_special_tokens=True)
                convs[idx].append({"role": "assistant", "content": reply})

            turn_rows.append({
                "turn": turn + 1,
                "mean_prompt_tokens": statistics.mean(prompt_lens),
                "wall_s": turn_wall,
                "ttft_mean_ms": 1000 * statistics.mean(turn_ttfts) if turn_ttfts else None,
            })

        bench_wall = time.perf_counter() - t_bench0
    gpu_stats = gpu.stats()
    vllm_stats = _try_vllm_metrics(engine)

    return {
        "engine": "vllm",
        "enable_prefix_caching": args.enable_prefix_caching,
        "bench_wall_s": bench_wall,
        "total_output_tokens": total_out_tokens,
        "throughput_tok_s": total_out_tokens / bench_wall,
        "ttft_mean_ms": 1000 * statistics.mean(ttfts) if ttfts else None,
        "ttft_p95_ms": 1000 * (statistics.quantiles(ttfts, n=20)[-1] if len(ttfts) >= 2 else ttfts[0]) if ttfts else None,
        "tpot_mean_ms": 1000 * statistics.mean(tpots) if tpots else None,
        "turns": turn_rows,
        **gpu_stats,
        **vllm_stats,
    }


def run_isolated(args, enable_prefix_caching: bool) -> dict:
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--single-run",
        "--enable-prefix-caching" if enable_prefix_caching else "--no-enable-prefix-caching",
        "--model", args.model,
        "--num-convs", str(args.num_convs),
        "--turns", str(args.turns),
        "--system-len", str(args.system_len),
        "--max-tokens", str(args.max_tokens),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--enforce-eager" if args.enforce_eager else "--no-enforce-eager",
        "--gpu-poll-s", str(args.gpu_poll_s),
        "--seed", str(args.seed),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    raise RuntimeError(
        f"subprocess produced no result (exit={proc.returncode}):\n"
        f"--- stdout ---\n{proc.stdout[-3000:]}\n--- stderr ---\n{proc.stderr[-5000:]}"
    )


def print_result(label: str, r: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"  throughput={r['throughput_tok_s']:.1f} tok/s  wall={r['bench_wall_s']:.2f}s  out_tokens={r['total_output_tokens']}")
    ttft_p95 = f"{r['ttft_p95_ms']:.0f}" if r["ttft_p95_ms"] is not None else "n/a"
    print(f"  TTFT mean={r['ttft_mean_ms']:.0f}ms p95={ttft_p95}ms   TPOT mean={r['tpot_mean_ms']:.1f}ms")
    hit = r.get("vllm_prefix_cache_hit_rate")
    preempt = r.get("vllm_num_preemptions")
    print(f"  vllm prefix hit_rate={f'{hit:.1%}' if hit is not None else 'n/a'}  preemptions={preempt if preempt is not None else 'n/a'}")
    util_mean = f"{r['gpu_util_mean']:.0f}" if r["gpu_util_mean"] is not None else "n/a"
    util_max = f"{r['gpu_util_max']:.0f}" if r["gpu_util_max"] is not None else "n/a"
    mem = f"{r['gpu_mem_peak_mib']:.0f}" if r["gpu_mem_peak_mib"] is not None else "n/a"
    print(f"  GPU util mean={util_mean}% max={util_max}%  peak_mem={mem}MiB  ({r['n_samples']} samples)")
    for t in r["turns"]:
        ttft = f"{t['ttft_mean_ms']:.0f}ms" if t["ttft_mean_ms"] is not None else "n/a"
        print(f"  turn {t['turn']}: prompt~{t['mean_prompt_tokens']:.0f}t wall={t['wall_s']:.2f}s ttft={ttft}")


def main() -> None:
    args = build_parser().parse_args()
    if not os.path.isdir(args.model):
        raise SystemExit(f"Model directory not found: {args.model}")

    if args.single_run:
        result = run_once(args)
        print(RESULT_PREFIX + json.dumps(result), flush=True)
        return

    print(
        f"engine=vllm model={args.model} num_convs={args.num_convs} turns={args.turns} "
        f"system~{args.system_len}tok max_tokens={args.max_tokens} "
        f"gpu_memory_utilization={args.gpu_memory_utilization} enforce_eager={args.enforce_eager}"
    )
    results = {}
    for enabled in (True, False):
        label = "vllm prefix_cache=ON" if enabled else "vllm prefix_cache=OFF"
        print(f"\nrunning {label} ...", flush=True)
        results[enabled] = run_isolated(args, enabled)
        print_result(label, results[enabled])

    on, off = results[True], results[False]
    print("\n=== vLLM ON vs OFF ===")
    print(f"  throughput: {on['throughput_tok_s']:.1f} vs {off['throughput_tok_s']:.1f} tok/s "
          f"({on['throughput_tok_s'] / off['throughput_tok_s']:.2f}x)")
    print(f"  TTFT mean:  {on['ttft_mean_ms']:.0f} vs {off['ttft_mean_ms']:.0f} ms")
    print(f"  TPOT mean:  {on['tpot_mean_ms']:.1f} vs {off['tpot_mean_ms']:.1f} ms")


if __name__ == "__main__":
    main()
