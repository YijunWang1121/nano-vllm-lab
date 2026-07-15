#!/usr/bin/env python3
"""Fair throughput comparison: course lab vs upstream main vs vLLM.

Engines:
  lab   — this checkout (course/reference-solutions + ablation switches)
  main  — upstream-style nano-vLLM from a git worktree of branch `main`
  vllm  — production vLLM (optional; requires `pip install vllm`)

Example:

    # prepare main worktree once
    git worktree add /tmp/nano-vllm-main main

    export PYTHONPATH=/nano-vllm-lab
    python experiments/compare_engines.py \\
      --engines lab,main,vllm \\
      --main-path /tmp/nano-vllm-main \\
      --num-seqs 64 --warmup

Metrics match bench.py: total output tokens / wall time (tok/s).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from random import randint, seed


ROOT = Path(__file__).resolve().parents[1]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model",
        default=os.path.expanduser(os.environ.get("NANOVLLM_TEST_MODEL", "~/huggingface/Qwen3-0.6B/")),
    )
    p.add_argument("--engines", default="lab,main,vllm", help="Comma list: lab,main,vllm")
    p.add_argument(
        "--main-path",
        default=os.environ.get("NANOVLLM_MAIN_PATH", "/tmp/nano-vllm-main"),
        help="Checkout/worktree of branch main for the upstream nano-vLLM engine",
    )
    p.add_argument("--num-seqs", type=int, default=64)
    p.add_argument("--max-input-len", type=int, default=512)
    p.add_argument("--max-output-len", type=int, default=128)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", action="store_true")
    p.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graphs for all engines")
    p.add_argument("--json-out", default="")
    p.add_argument("--skip-missing", action="store_true", help="Skip engines that are unavailable")
    return p


def make_workload(args: argparse.Namespace):
    seed(args.seed)
    prompts = [
        [randint(0, 10000) for _ in range(randint(32, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    max_tokens = [randint(16, args.max_output_len) for _ in range(args.num_seqs)]
    return prompts, max_tokens


def _cleanup_cuda():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def run_lab(model: str, prompts, max_tokens, args: argparse.Namespace) -> dict:
    """Current checkout (lab / course reference)."""
    # Ensure this repo wins over any other install.
    sys.path.insert(0, str(ROOT))
    from nanovllm import LLM, SamplingParams
    import nanovllm

    print(f"  package: {nanovllm.__file__}")
    sps = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=n)
        for n in max_tokens
    ]
    llm = LLM(
        model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        tensor_parallel_size=1,
    )
    try:
        if args.warmup:
            llm.generate(["warmup"], SamplingParams(max_tokens=4, temperature=0.6), use_tqdm=False)
        t0 = time.perf_counter()
        llm.generate(prompts, sps, use_tqdm=False)
        elapsed = time.perf_counter() - t0
    finally:
        llm.exit()
        _cleanup_cuda()
    total = sum(max_tokens)
    return {
        "engine": "lab",
        "label": "course/reference-solutions (this tree)",
        "seconds": round(elapsed, 4),
        "total_tokens": total,
        "tok_per_s": round(total / elapsed, 2) if elapsed else 0.0,
        "package": nanovllm.__file__,
    }


def run_main_subprocess(model: str, prompts, max_tokens, args: argparse.Namespace) -> dict:
    """Run upstream main nano-vLLM in an isolated subprocess + PYTHONPATH."""
    main_path = Path(os.path.expanduser(args.main_path)).resolve()
    if not (main_path / "nanovllm").is_dir():
        raise FileNotFoundError(
            f"main engine not found at {main_path}. Create with:\n"
            f"  git worktree add {main_path} main"
        )

    payload = {
        "model": model,
        "prompts": prompts,
        "max_tokens": max_tokens,
        "max_model_len": args.max_model_len,
        "enforce_eager": args.enforce_eager,
        "warmup": args.warmup,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        payload_path = f.name

    worker = r"""
import json, sys, time, gc
payload_path = sys.argv[1]
with open(payload_path) as f:
    p = json.load(f)
from nanovllm import LLM, SamplingParams
import nanovllm
print("PACKAGE", nanovllm.__file__, flush=True)
sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=n) for n in p["max_tokens"]]
llm = LLM(p["model"], enforce_eager=p["enforce_eager"], max_model_len=p["max_model_len"], tensor_parallel_size=1)
try:
    if p["warmup"]:
        llm.generate(["warmup"], SamplingParams(max_tokens=4, temperature=0.6), use_tqdm=False)
    t0 = time.perf_counter()
    llm.generate(p["prompts"], sps, use_tqdm=False)
    elapsed = time.perf_counter() - t0
finally:
    if hasattr(llm, "exit"):
        llm.exit()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
total = sum(p["max_tokens"])
print("RESULT", json.dumps({"seconds": elapsed, "total_tokens": total, "package": nanovllm.__file__}), flush=True)
"""
    env = os.environ.copy()
    # Prefer main tree; drop lab PYTHONPATH so editable lab install does not win.
    env["PYTHONPATH"] = str(main_path)
    proc = subprocess.run(
        [sys.executable, "-c", worker, payload_path],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        os.unlink(payload_path)
    except OSError:
        pass
    if proc.returncode != 0:
        raise RuntimeError(
            f"main engine failed (exit {proc.returncode}):\n"
            f"{proc.stderr[-2000:]}\n{proc.stdout[-2000:]}"
        )
    package = ""
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("PACKAGE "):
            package = line[len("PACKAGE ") :]
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT ") :])
    if result is None:
        raise RuntimeError(f"main engine produced no RESULT:\n{proc.stdout}\n{proc.stderr}")
    elapsed = result["seconds"]
    total = result["total_tokens"]
    return {
        "engine": "main",
        "label": f"git main @ {main_path}",
        "seconds": round(elapsed, 4),
        "total_tokens": total,
        "tok_per_s": round(total / elapsed, 2) if elapsed else 0.0,
        "package": package or result.get("package"),
    }


def run_vllm(model: str, prompts, max_tokens, args: argparse.Namespace) -> dict:
    try:
        from vllm import LLM, SamplingParams
        import vllm
    except ImportError as e:
        raise ImportError("vLLM not installed. On the GPU box: pip install vllm") from e

    print(f"  package: {vllm.__file__}")
    sps = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=n)
        for n in max_tokens
    ]
    # vLLM offline API: list[dict(prompt_token_ids=...)]
    reqs = [{"prompt_token_ids": p} for p in prompts]
    llm = LLM(
        model=model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        tensor_parallel_size=1,
        disable_log_stats=True,
    )
    try:
        if args.warmup:
            llm.generate(["warmup"], SamplingParams(max_tokens=4, temperature=0.6), use_tqdm=False)
        t0 = time.perf_counter()
        llm.generate(reqs, sps, use_tqdm=False)
        elapsed = time.perf_counter() - t0
    finally:
        # vLLM LLM has no universal exit(); best-effort cleanup.
        del llm
        _cleanup_cuda()
    total = sum(max_tokens)
    return {
        "engine": "vllm",
        "label": f"vLLM {getattr(vllm, '__version__', '?')}",
        "seconds": round(elapsed, 4),
        "total_tokens": total,
        "tok_per_s": round(total / elapsed, 2) if elapsed else 0.0,
        "package": vllm.__file__,
    }


RUNNERS = {
    "lab": run_lab,
    "main": run_main_subprocess,
    "vllm": run_vllm,
}


def print_table(rows: list[dict]) -> None:
    if not rows:
        return
    ref = next((r for r in rows if r["engine"] == "lab"), rows[0])
    ref_tps = ref["tok_per_s"] or 1.0
    print("\n======== engine comparison ========")
    print(f"(relative to {ref['engine']})")
    print(f"{'engine':10} {'tok/s':>10} {'sec':>8} {'vs_lab':>8}  label")
    for r in rows:
        if r.get("error"):
            print(f"{r['engine']:10} {'FAIL':>10} {'-':>8} {'-':>8}  {r['error'][:70]}")
            continue
        rel = r["tok_per_s"] / ref_tps
        print(
            f"{r['engine']:10} {r['tok_per_s']:10.2f} {r['seconds']:8.2f} {rel:8.2f}x  {r.get('label', '')}"
        )


def main():
    args = build_argparser().parse_args()
    model = os.path.expanduser(args.model)
    if not os.path.isdir(model):
        raise SystemExit(f"model dir not found: {model}")

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    for e in engines:
        if e not in RUNNERS:
            raise SystemExit(f"unknown engine {e!r}; choose from lab,main,vllm")

    prompts, max_tokens = make_workload(args)
    print(
        f"workload: num_seqs={args.num_seqs} max_input={args.max_input_len} "
        f"max_output={args.max_output_len} enforce_eager={args.enforce_eager}"
    )

    rows = []
    for name in engines:
        print(f"\n=== {name} ===")
        try:
            row = RUNNERS[name](model, prompts, max_tokens, args)
            print(
                f"result: {row['total_tokens']} tok / {row['seconds']:.2f}s = {row['tok_per_s']:.2f} tok/s"
            )
            rows.append(row)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"FAILED: {msg}")
            if not args.skip_missing:
                raise
            rows.append({"engine": name, "error": msg, "tok_per_s": 0.0, "seconds": None})

    print_table(rows)
    if args.json_out:
        path = os.path.expanduser(args.json_out)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"model": model, "results": rows}, f, indent=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
