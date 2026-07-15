#!/usr/bin/env python3
"""Run A/B and ablation throughput experiments for nano-vLLM.

Examples (on a GPU machine with the model downloaded):

    export PYTHONPATH=/nano-vllm-lab
    # Random prompts (prefix cache barely helps)
    python experiments/run_ablation.py --preset baseline,no_cudagraph --warmup

    # Shared system prompt (prefix cache should help)
    python experiments/run_ablation.py --preset baseline,no_prefix \\
        --workload shared_prefix --shared-prefix-len 512 --warmup

    # Multi-turn chat simulation
    python experiments/run_ablation.py --preset baseline,no_prefix \\
        --workload multi_turn --num-sessions 16 --num-turns 4 \\
        --shared-prefix-len 512 --warmup

Keep NANOVLLM_DEBUG_* off while measuring.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import asdict, dataclass
from random import randint, seed

from nanovllm import LLM, SamplingParams


@dataclass(frozen=True)
class RunConfig:
    name: str
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    enable_preemption: bool = True
    enable_chunked_prefill: bool = True
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 16384
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1


PRESETS: dict[str, RunConfig] = {
    "baseline": RunConfig(name="baseline"),
    "no_cudagraph": RunConfig(name="no_cudagraph", enforce_eager=True),
    "no_prefix": RunConfig(name="no_prefix", enable_prefix_caching=False),
    "no_preempt": RunConfig(name="no_preempt", enable_preemption=False),
    "no_chunked_prefill": RunConfig(name="no_chunked_prefill", enable_chunked_prefill=False),
    "single_seq": RunConfig(name="single_seq", max_num_seqs=1),
    "small_batch": RunConfig(name="small_batch", max_num_seqs=8, max_num_batched_tokens=2048),
}


def _parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected bool, got {value!r}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model",
        default=os.path.expanduser(os.environ.get("NANOVLLM_TEST_MODEL", "~/huggingface/Qwen3-0.6B/")),
        help="Local HF model directory",
    )
    p.add_argument(
        "--preset",
        default="baseline,no_cudagraph,no_prefix",
        help="Comma-separated presets, or 'all'. Available: " + ", ".join(PRESETS),
    )
    p.add_argument(
        "--workload",
        choices=("random", "shared_prefix", "multi_turn"),
        default="random",
        help="random: independent prompts; shared_prefix: same system prefix; "
        "multi_turn: chat sessions with growing history",
    )
    p.add_argument("--num-seqs", type=int, default=64, help="Requests for random/shared_prefix")
    p.add_argument("--num-sessions", type=int, default=16, help="Chat sessions for multi_turn")
    p.add_argument("--num-turns", type=int, default=4, help="Turns per session for multi_turn")
    p.add_argument(
        "--shared-prefix-len",
        type=int,
        default=512,
        help="Shared system/history prefix tokens (prefer multiple of 256 = block_size)",
    )
    p.add_argument("--suffix-len", type=int, default=64, help="Unique suffix / user-turn tokens")
    p.add_argument("--max-input-len", type=int, default=512, help="For workload=random only")
    p.add_argument("--max-output-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", action="store_true", help="Short warmup generate before timing")
    p.add_argument(
        "--prime-prefix",
        action="store_true",
        default=True,
        help="For shared_prefix/multi_turn: run shared prefix once before timing so hashes exist",
    )
    p.add_argument("--no-prime-prefix", action="store_false", dest="prime_prefix")
    p.add_argument("--json-out", default="", help="Optional path to write JSON results")
    p.add_argument("--enforce-eager", type=_parse_bool, default=None)
    p.add_argument("--prefix-caching", type=_parse_bool, default=None, dest="enable_prefix_caching")
    p.add_argument("--preemption", type=_parse_bool, default=None, dest="enable_preemption")
    p.add_argument("--chunked-prefill", type=_parse_bool, default=None, dest="enable_chunked_prefill")
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--max-num-batched-tokens", type=int, default=None)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
    return p


def resolve_presets(spec: str) -> list[RunConfig]:
    spec = spec.strip()
    if spec == "all":
        return list(PRESETS.values())
    names = [x.strip() for x in spec.split(",") if x.strip()]
    out: list[RunConfig] = []
    for name in names:
        if name not in PRESETS:
            raise SystemExit(f"unknown preset {name!r}; choose from: {', '.join(PRESETS)}")
        out.append(PRESETS[name])
    return out


def apply_overrides(cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    d = asdict(cfg)
    for key in (
        "enforce_eager",
        "enable_prefix_caching",
        "enable_preemption",
        "enable_chunked_prefill",
        "max_num_seqs",
        "max_num_batched_tokens",
        "max_model_len",
        "gpu_memory_utilization",
    ):
        val = getattr(args, key, None)
        if val is not None:
            d[key] = val
    return RunConfig(**d)


def make_random_workload(args: argparse.Namespace):
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(32, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(16, args.max_output_len))
        for _ in range(args.num_seqs)
    ]
    return {"kind": "random", "prompts": prompt_token_ids, "sampling_params": sampling_params, "shared_prefix": None}


def make_shared_prefix_workload(args: argparse.Namespace):
    # Align to block_size so full blocks can hash-hit.
    prefix_len = args.shared_prefix_len - (args.shared_prefix_len % 256)
    if prefix_len <= 0:
        raise SystemExit("--shared-prefix-len must be >= 256 (kvcache_block_size)")
    shared_prefix = [randint(0, 10000) for _ in range(prefix_len)]
    prompts = []
    for _ in range(args.num_seqs):
        suffix = [randint(0, 10000) for _ in range(args.suffix_len)]
        prompts.append(shared_prefix + suffix)
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(16, args.max_output_len))
        for _ in range(args.num_seqs)
    ]
    return {
        "kind": "shared_prefix",
        "prompts": prompts,
        "sampling_params": sampling_params,
        "shared_prefix": shared_prefix,
        "prefix_len": prefix_len,
    }


def make_multi_turn_workload(args: argparse.Namespace):
    """Simulate multi-turn chats: same system prefix, growing per-session history."""
    prefix_len = args.shared_prefix_len - (args.shared_prefix_len % 256)
    if prefix_len <= 0:
        raise SystemExit("--shared-prefix-len must be >= 256 (kvcache_block_size)")
    system = [randint(0, 10000) for _ in range(prefix_len)]
    # user_turns[session][turn] -> token ids
    user_turns = [
        [[randint(0, 10000) for _ in range(args.suffix_len)] for _ in range(args.num_turns)]
        for _ in range(args.num_sessions)
    ]
    sampling_params = SamplingParams(
        temperature=0.6,
        ignore_eos=True,
        max_tokens=min(64, args.max_output_len),
    )
    return {
        "kind": "multi_turn",
        "system": system,
        "user_turns": user_turns,
        "sampling_params": sampling_params,
        "shared_prefix": system,
        "prefix_len": prefix_len,
        "num_sessions": args.num_sessions,
        "num_turns": args.num_turns,
    }


def make_workload(args: argparse.Namespace):
    seed(args.seed)
    if args.workload == "random":
        return make_random_workload(args)
    if args.workload == "shared_prefix":
        return make_shared_prefix_workload(args)
    return make_multi_turn_workload(args)


def print_kv_capacity(llm: LLM) -> dict:
    """Report how large the paged KV pool is after allocate_kv_cache()."""
    cfg = llm.model_runner.config
    block_size = cfg.kvcache_block_size
    n_blocks = cfg.num_kvcache_blocks
    max_tokens = n_blocks * block_size
    # Rough concurrent full-length seqs that fit in the pool.
    per_seq_blocks = (cfg.max_model_len + block_size - 1) // block_size
    max_full_seqs = n_blocks // per_seq_blocks if per_seq_blocks else 0
    info = {
        "num_kvcache_blocks": n_blocks,
        "kvcache_block_size": block_size,
        "kv_pool_tokens": max_tokens,
        "max_model_len": cfg.max_model_len,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
        "approx_full_len_seqs": max_full_seqs,
    }
    print(
        "kv_cache:",
        f"blocks={n_blocks}",
        f"block_size={block_size}",
        f"pool_tokens={max_tokens}",
        f"max_model_len={cfg.max_model_len}",
        f"approx_full_len_seqs={max_full_seqs}",
    )
    return info


def _cleanup(llm: LLM) -> None:
    llm.exit()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def run_timed_generate(llm: LLM, prompts, sampling_params) -> tuple[float, int]:
    t0 = time.perf_counter()
    llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - t0
    if isinstance(sampling_params, list):
        total_tokens = sum(sp.max_tokens for sp in sampling_params)
    else:
        total_tokens = sampling_params.max_tokens * len(prompts)
    return elapsed, total_tokens


def run_multi_turn(llm: LLM, workload: dict) -> tuple[float, int]:
    """Grow history each turn; prefix cache can reuse system + prior turns."""
    system = workload["system"]
    user_turns = workload["user_turns"]
    sp = workload["sampling_params"]
    histories = [list(system) for _ in range(workload["num_sessions"])]
    total_out = 0
    t0 = time.perf_counter()
    for t in range(workload["num_turns"]):
        prompts = [hist + user_turns[i][t] for i, hist in enumerate(histories)]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        total_out += sp.max_tokens * len(prompts)
        for i, out in enumerate(outs):
            # Append user turn + generated tokens to history for next turn.
            histories[i] = prompts[i] + list(out["token_ids"])
    elapsed = time.perf_counter() - t0
    return elapsed, total_out


def run_one(model: str, cfg: RunConfig, workload: dict, args: argparse.Namespace) -> dict:
    print(f"\n=== {cfg.name} ===")
    print(
        "switches:",
        f"enforce_eager={cfg.enforce_eager}",
        f"prefix={cfg.enable_prefix_caching}",
        f"preempt={cfg.enable_preemption}",
        f"chunked_prefill={cfg.enable_chunked_prefill}",
        f"max_num_seqs={cfg.max_num_seqs}",
        f"max_num_batched_tokens={cfg.max_num_batched_tokens}",
        f"workload={workload['kind']}",
    )
    llm = LLM(
        model,
        enforce_eager=cfg.enforce_eager,
        enable_prefix_caching=cfg.enable_prefix_caching,
        enable_preemption=cfg.enable_preemption,
        enable_chunked_prefill=cfg.enable_chunked_prefill,
        max_num_seqs=cfg.max_num_seqs,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        tensor_parallel_size=cfg.tensor_parallel_size,
    )
    try:
        kv_info = print_kv_capacity(llm)
        if args.warmup:
            llm.generate(["warmup"], SamplingParams(max_tokens=4, temperature=0.6), use_tqdm=False)

        # Populate hash_to_block_id so the timed batch can hit shared prefix blocks.
        shared = workload.get("shared_prefix")
        if args.prime_prefix and shared is not None and cfg.enable_prefix_caching:
            llm.generate([shared], SamplingParams(max_tokens=1, temperature=0.6), use_tqdm=False)

        if workload["kind"] == "multi_turn":
            elapsed, total_tokens = run_multi_turn(llm, workload)
        else:
            elapsed, total_tokens = run_timed_generate(
                llm, workload["prompts"], workload["sampling_params"]
            )
    finally:
        _cleanup(llm)

    throughput = total_tokens / elapsed if elapsed > 0 else 0.0
    row = {
        "name": cfg.name,
        "seconds": round(elapsed, 4),
        "total_tokens": total_tokens,
        "tok_per_s": round(throughput, 2),
        "config": asdict(cfg),
        "workload": workload["kind"],
        "kv_cache": kv_info,
    }
    print(f"result: {row['total_tokens']} tok / {row['seconds']:.2f}s = {row['tok_per_s']:.2f} tok/s")
    return row


def print_table(rows: list[dict]) -> None:
    if not rows:
        return
    baseline = next((r for r in rows if r["name"] == "baseline"), rows[0])
    base_tps = baseline["tok_per_s"] or 1.0
    print("\n======== ablation comparison ========")
    print(f"{'name':22} {'tok/s':>10} {'sec':>8} {'vs_base':>8}")
    for r in rows:
        rel = r["tok_per_s"] / base_tps if base_tps else 0.0
        print(f"{r['name']:22} {r['tok_per_s']:10.2f} {r['seconds']:8.2f} {rel:8.2f}x")


def main():
    args = build_argparser().parse_args()
    model = os.path.expanduser(args.model)
    if not os.path.isdir(model):
        raise SystemExit(f"model dir not found: {model}")

    presets = [apply_overrides(cfg, args) for cfg in resolve_presets(args.preset)]
    workload = make_workload(args)
    if workload["kind"] != "random":
        print(
            f"workload={workload['kind']} shared_prefix_len={workload.get('prefix_len')} "
            f"prime_prefix={args.prime_prefix}"
        )

    rows = []
    for cfg in presets:
        rows.append(run_one(model, cfg, workload, args))

    print_table(rows)
    if args.json_out:
        path = os.path.expanduser(args.json_out)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"model": model, "results": rows}, f, indent=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
