#!/usr/bin/env python3
"""Heavy multi-turn chat serving benchmark: prefix cache ON vs OFF.

Workload: N concurrent conversations, each with a shared (templated) system
prompt, run for T turns in the stateless-server style -- every turn re-sends
the FULL chat-templated history, so turn t's prompt embeds all previous
turns' text. This is exactly the pattern prefix caching exists for: with the
cache ON, turn t only prefills the new tail; OFF, it recomputes the whole
history from scratch every turn.

Per config (Config.enable_prefix_caching True/False), measured via a manual
llm.step() loop (llm.generate() can't observe per-token timing):

  TTFT        per request: submit -> first sampled token (mean + p95)
  TPOT        per request: (finish - first token) / (n_out - 1), mean
  throughput  total output tokens / total measured wall time
  hit_rate    BlockManager.prefix_cache_stats() per turn and overall
  preempt     Scheduler.num_preemptions (+ swaps if enabled)
  gpu_util    background thread samples nvidia-smi every --gpu-poll-s;
              reports mean/max GPU busy % and peak memory over the run

Each config runs in a subprocess (RESULT_JSON pattern shared with the other
bench_*.py scripts) so GPU memory is fully released between configs.

Usage:
    export NANOVLLM_TEST_MODEL=~/huggingface/Qwen3-0.6B
    python experiments/bench_prefix_cache_serving.py
    python experiments/bench_prefix_cache_serving.py --num-convs 12 --turns 5
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common_workload import default_model_path  # noqa: E402

RESULT_PREFIX = "RESULT_JSON:"

SYSTEM_SENTENCE = (
    "You are a meticulous engineering assistant for an LLM serving codebase; "
    "always reason about scheduling, paged KV cache blocks, prefix reuse, CUDA "
    "graphs, and tensor parallel layouts before you answer. "
)

USER_TURNS = [
    "What is a paged KV cache and why do serving engines use it?",
    "How does prefix caching interact with those KV blocks?",
    "When does a multi-turn conversation benefit from that cache?",
    "What metadata does the attention kernel need for a cached prefill?",
    "Why do CUDA graphs mostly help the decode phase?",
    "How would tensor parallelism change the KV block layout?",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=default_model_path())
    p.add_argument("--num-convs", type=int, default=8, help="concurrent conversations per turn wave")
    p.add_argument("--turns", type=int, default=4)
    p.add_argument("--system-len", type=int, default=768, help="approx tokens of shared system prompt (multiple of 256 recommended so full blocks are shareable)")
    p.add_argument("--max-tokens", type=int, default=60, help="output tokens per turn per conversation")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True,
                   help="--no-enforce-eager enables CUDA graphs for decode (biggest single throughput lever)")
    p.add_argument("--gpu-poll-s", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--single-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=argparse.SUPPRESS,  # internal: subprocess-isolated single measurement
    )
    p.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True, help=argparse.SUPPRESS)
    return p


class GpuMonitor:
    """Samples nvidia-smi utilization/memory in a background thread."""

    def __init__(self, poll_s: float):
        self.poll_s = poll_s
        self.samples: list[tuple[float, float]] = []  # (util %, mem MiB)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip().splitlines()[0]
                util, mem = (float(x) for x in out.split(","))
                self.samples.append((util, mem))
            except Exception:
                pass  # transient nvidia-smi failure: skip this sample
            self._stop.wait(self.poll_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)

    def stats(self) -> dict:
        if not self.samples:
            return {"gpu_util_mean": None, "gpu_util_max": None, "gpu_mem_peak_mib": None, "n_samples": 0}
        utils = [u for u, _ in self.samples]
        mems = [m for _, m in self.samples]
        return {
            "gpu_util_mean": statistics.mean(utils),
            "gpu_util_max": max(utils),
            "gpu_mem_peak_mib": max(mems),
            "n_samples": len(utils),
        }


def build_system_prompt(tokenizer, target_tokens: int) -> str:
    text = SYSTEM_SENTENCE
    while len(tokenizer.encode(text)) < target_tokens:
        text += SYSTEM_SENTENCE
    return text


def run_once(args) -> dict:
    from nanovllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        enable_prefix_caching=args.enable_prefix_caching,
    )
    try:
        llm.generate(["warmup"], SamplingParams(max_tokens=4), use_tqdm=False)
        llm.scheduler.block_manager.reset_prefix_cache_stats()

        system_prompt = build_system_prompt(tokenizer, args.system_len)
        # Each conversation gets a distinct opener so histories diverge
        # after turn 1 (only the system prompt + template header stays
        # shared across conversations; within one conversation, the whole
        # history is the shared prefix between its own consecutive turns).
        convs = [
            [{"role": "system", "content": system_prompt},
             ]
            for _ in range(args.num_convs)
        ]

        sp = SamplingParams(temperature=0.6, max_tokens=args.max_tokens)
        ttfts: list[float] = []
        tpots: list[float] = []
        turn_rows = []
        total_out_tokens = 0
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

                llm.scheduler.block_manager.reset_prefix_cache_stats()
                seq_of = {}
                t0 = time.perf_counter()
                for i, prompt in enumerate(prompts):
                    llm.add_request(prompt, sp)
                    seq_of[llm.scheduler.waiting[-1].seq_id] = i

                first_tok: dict[int, float] = {}
                finish: dict[int, float] = {}
                finished_tokens: dict[int, list[int]] = {}
                tracked = {s.seq_id: s for s in list(llm.scheduler.waiting)}

                while not llm.is_finished():
                    outputs, _ = llm.step()
                    now = time.perf_counter() - t0
                    for sid in seq_of:
                        if sid not in first_tok:
                            s = tracked.get(sid)
                            if s is not None and s.num_completion_tokens >= 1:
                                first_tok[sid] = now
                    for sid, token_ids, aborted in outputs:
                        if sid in seq_of and sid not in finish:
                            finish[sid] = now
                            finished_tokens[sid] = token_ids

                turn_wall = time.perf_counter() - t0
                stats = llm.scheduler.block_manager.prefix_cache_stats()

                turn_ttfts = []
                for sid, idx in seq_of.items():
                    out_ids = finished_tokens.get(sid, [])
                    n_out = len(out_ids)
                    total_out_tokens += n_out
                    if sid in first_tok:
                        turn_ttfts.append(first_tok[sid])
                        ttfts.append(first_tok[sid])
                        if sid in finish and n_out > 1:
                            tpots.append((finish[sid] - first_tok[sid]) / (n_out - 1))
                    reply = tokenizer.decode(out_ids, skip_special_tokens=True)
                    convs[idx].append({"role": "assistant", "content": reply})

                turn_rows.append({
                    "turn": turn + 1,
                    "mean_prompt_tokens": statistics.mean(prompt_lens),
                    "wall_s": turn_wall,
                    "ttft_mean_ms": 1000 * statistics.mean(turn_ttfts) if turn_ttfts else None,
                    "hit_rate": stats["hit_rate"],
                    "cached_tokens": stats["cached_tokens"],
                    "prompt_tokens": stats["prompt_tokens"],
                })

            bench_wall = time.perf_counter() - t_bench0
        gpu_stats = gpu.stats()

        result = {
            "enable_prefix_caching": args.enable_prefix_caching,
            "bench_wall_s": bench_wall,
            "total_output_tokens": total_out_tokens,
            "throughput_tok_s": total_out_tokens / bench_wall,
            "ttft_mean_ms": 1000 * statistics.mean(ttfts) if ttfts else None,
            "ttft_p95_ms": 1000 * (statistics.quantiles(ttfts, n=20)[-1] if len(ttfts) >= 2 else ttfts[0]) if ttfts else None,
            "tpot_mean_ms": 1000 * statistics.mean(tpots) if tpots else None,
            "num_preemptions": llm.scheduler.num_preemptions,
            "num_swaps": llm.scheduler.num_swaps,
            "turns": turn_rows,
            **gpu_stats,
        }
    finally:
        llm.exit()
    return result


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
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--enforce-eager" if args.enforce_eager else "--no-enforce-eager",
        "--gpu-poll-s", str(args.gpu_poll_s),
        "--seed", str(args.seed),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    raise RuntimeError(
        f"subprocess produced no result (exit={proc.returncode}):\n"
        f"--- stdout ---\n{proc.stdout[-3000:]}\n--- stderr ---\n{proc.stderr[-4000:]}"
    )


def print_result(label: str, r: dict) -> None:
    print(f"\n=== {label} ===")
    print(
        f"  throughput={r['throughput_tok_s']:.1f} tok/s  wall={r['bench_wall_s']:.2f}s  "
        f"out_tokens={r['total_output_tokens']}"
    )
    ttft_p95 = f"{r['ttft_p95_ms']:.0f}" if r["ttft_p95_ms"] is not None else "n/a"
    print(f"  TTFT mean={r['ttft_mean_ms']:.0f}ms p95={ttft_p95}ms   TPOT mean={r['tpot_mean_ms']:.1f}ms")
    print(f"  preemptions={r['num_preemptions']} swaps={r['num_swaps']}")
    util_mean = f"{r['gpu_util_mean']:.0f}" if r["gpu_util_mean"] is not None else "n/a"
    util_max = f"{r['gpu_util_max']:.0f}" if r["gpu_util_max"] is not None else "n/a"
    mem = f"{r['gpu_mem_peak_mib']:.0f}" if r["gpu_mem_peak_mib"] is not None else "n/a"
    print(f"  GPU util mean={util_mean}% max={util_max}%  peak_mem={mem}MiB  ({r['n_samples']} samples)")
    print(f"  {'turn':>4} {'prompt':>7} {'wall':>7} {'ttft':>7} {'hit_rate':>8} {'cached/prompt':>14}")
    for t in r["turns"]:
        ttft = f"{t['ttft_mean_ms']:.0f}ms" if t["ttft_mean_ms"] is not None else "n/a"
        print(
            f"  {t['turn']:>4} {t['mean_prompt_tokens']:>6.0f}t {t['wall_s']:>6.2f}s {ttft:>7} "
            f"{t['hit_rate']:>7.1%} {t['cached_tokens']:>6}/{t['prompt_tokens']:<7}"
        )


def main() -> None:
    args = build_parser().parse_args()
    if not os.path.isdir(args.model):
        raise SystemExit(f"Model directory not found: {args.model}")

    if args.single_run:
        result = run_once(args)
        print(RESULT_PREFIX + json.dumps(result), flush=True)
        return

    print(
        f"model={args.model} num_convs={args.num_convs} turns={args.turns} "
        f"system~{args.system_len}tok max_tokens={args.max_tokens} "
        f"gpu_memory_utilization={args.gpu_memory_utilization}"
    )
    results = {}
    for enabled in (True, False):
        label = "prefix_cache=ON" if enabled else "prefix_cache=OFF"
        print(f"\nrunning {label} ...", flush=True)
        results[enabled] = run_isolated(args, enabled)
        print_result(label, results[enabled])

    on, off = results[True], results[False]
    print("\n=== ON vs OFF ===")
    print(f"  throughput: {on['throughput_tok_s']:.1f} vs {off['throughput_tok_s']:.1f} tok/s "
          f"({on['throughput_tok_s'] / off['throughput_tok_s']:.2f}x)")
    print(f"  TTFT mean:  {on['ttft_mean_ms']:.0f} vs {off['ttft_mean_ms']:.0f} ms "
          f"({off['ttft_mean_ms'] / on['ttft_mean_ms']:.2f}x lower)")
    print(f"  TPOT mean:  {on['tpot_mean_ms']:.1f} vs {off['tpot_mean_ms']:.1f} ms")
    print(f"  preempt:    {on['num_preemptions']} vs {off['num_preemptions']}")


if __name__ == "__main__":
    main()
