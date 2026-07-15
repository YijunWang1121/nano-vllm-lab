#!/usr/bin/env python3
"""Run A/B and ablation throughput experiments for nano-vLLM.

Examples (on a GPU machine with the model downloaded):

    export PYTHONPATH=/nano-vllm-lab
    python experiments/run_ablation.py --preset all
    python experiments/run_ablation.py --preset baseline,no_cudagraph,no_prefix
    python experiments/run_ablation.py --enforce-eager --no-prefix-caching --num-seqs 64

Presets flip Config ablation switches; results print as a comparison table.
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
    p.add_argument("--num-seqs", type=int, default=64, help="Number of random requests per run")
    p.add_argument("--max-input-len", type=int, default=512)
    p.add_argument("--max-output-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", action="store_true", help="One short warmup generate before timing")
    p.add_argument("--json-out", default="", help="Optional path to write JSON results")
    # Manual overrides (applied on top of each preset when set)
    p.add_argument("--enforce-eager", type=_parse_bool, default=None)
    p.add_argument("--prefix-caching", type=_parse_bool, default=None, dest="enable_prefix_caching")
    p.add_argument("--preemption", type=_parse_bool, default=None, dest="enable_preemption")
    p.add_argument("--chunked-prefill", type=_parse_bool, default=None, dest="enable_chunked_prefill")
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--max-num-batched-tokens", type=int, default=None)
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
        "gpu_memory_utilization",
    ):
        val = getattr(args, key, None)
        if val is not None:
            d[key] = val
    return RunConfig(**d)


def make_workload(args: argparse.Namespace):
    seed(args.seed)
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(32, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=randint(16, args.max_output_len),
        )
        for _ in range(args.num_seqs)
    ]
    return prompt_token_ids, sampling_params


def run_one(model: str, cfg: RunConfig, prompts, sampling_params, warmup: bool) -> dict:
    print(f"\n=== {cfg.name} ===")
    print(
        "switches:",
        f"enforce_eager={cfg.enforce_eager}",
        f"prefix={cfg.enable_prefix_caching}",
        f"preempt={cfg.enable_preemption}",
        f"chunked_prefill={cfg.enable_chunked_prefill}",
        f"max_num_seqs={cfg.max_num_seqs}",
        f"max_num_batched_tokens={cfg.max_num_batched_tokens}",
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
        if warmup:
            llm.generate(["warmup"], SamplingParams(max_tokens=4, temperature=0.6), use_tqdm=False)
        t0 = time.perf_counter()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        elapsed = time.perf_counter() - t0
    finally:
        llm.exit()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / elapsed if elapsed > 0 else 0.0
    row = {
        "name": cfg.name,
        "seconds": round(elapsed, 4),
        "total_tokens": total_tokens,
        "tok_per_s": round(throughput, 2),
        "config": asdict(cfg),
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
    prompts, sampling_params = make_workload(args)

    rows = []
    for cfg in presets:
        rows.append(run_one(model, cfg, prompts, sampling_params, warmup=args.warmup))

    print_table(rows)
    if args.json_out:
        path = os.path.expanduser(args.json_out)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"model": model, "results": rows}, f, indent=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
