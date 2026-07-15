#!/usr/bin/env python3
"""Fair throughput comparison: course lab vs upstream main vs vLLM.

No-flash-attn mode (recommended after vLLM upgrades torch):

    export NANOVLLM_ATTN_BACKEND=torch
    python experiments/compare_engines.py --engines lab,vllm --enforce-eager --warmup

SDPA attention is incompatible with nano-vLLM CUDA graphs → always use --enforce-eager
when NANOVLLM_ATTN_BACKEND=torch.
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
    p.add_argument("--engines", default="lab,vllm", help="Comma list: lab,main,vllm")
    p.add_argument(
        "--main-path",
        default=os.environ.get("NANOVLLM_MAIN_PATH", "/tmp/nano-vllm-main"),
        help="Checkout/worktree of branch main for the upstream nano-vLLM engine",
    )
    p.add_argument("--num-seqs", type=int, default=64)
    p.add_argument("--max-input-len", type=int, default=512)
    p.add_argument("--max-output-len", type=int, default=128)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", action="store_true")
    p.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Disable CUDA graphs. Default True when NANOVLLM_ATTN_BACKEND=torch.",
    )
    p.add_argument("--json-out", default="")
    p.add_argument("--skip-missing", action="store_true", help="Skip engines that fail")
    return p


def resolve_enforce_eager(args: argparse.Namespace) -> bool:
    if args.enforce_eager is not None:
        return bool(args.enforce_eager)
    backend = os.environ.get("NANOVLLM_ATTN_BACKEND", "auto").strip().lower()
    if backend in ("torch", "sdpa", "eager"):
        return True
    return False


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
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            free, total = torch.cuda.mem_get_info()
            print(f"  cuda free after cleanup: {free/1024**3:.2f}/{total/1024**3:.2f} GiB")
    except Exception as exc:
        print(f"  cuda cleanup warning: {exc}")


def _run_isolated_worker(worker_src: str, payload: dict, env: dict | None = None) -> dict:
    """Run one engine in a fresh process so GPU memory is released on exit."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        payload_path = f.name
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    proc = subprocess.run(
        [sys.executable, "-c", worker_src, payload_path],
        env=run_env,
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
            f"worker failed (exit {proc.returncode}):\n"
            f"{proc.stderr[-2500:]}\n{proc.stdout[-2500:]}"
        )
    package = ""
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("PACKAGE "):
            package = line[len("PACKAGE ") :]
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT ") :])
    if result is None:
        raise RuntimeError(f"worker produced no RESULT:\n{proc.stdout}\n{proc.stderr}")
    result["package"] = package or result.get("package")
    return result


LAB_WORKER = r"""
import json, sys, time, gc, os
payload_path = sys.argv[1]
with open(payload_path) as f:
    p = json.load(f)
# Prefer lab tree from env PYTHONPATH
from nanovllm import LLM, SamplingParams
import nanovllm
print("PACKAGE", nanovllm.__file__, flush=True)
sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=n) for n in p["max_tokens"]]
llm = LLM(
    p["model"],
    enforce_eager=p["enforce_eager"],
    max_model_len=p["max_model_len"],
    gpu_memory_utilization=p["gpu_memory_utilization"],
    tensor_parallel_size=1,
)
try:
    if p["warmup"]:
        llm.generate(["warmup"], SamplingParams(max_tokens=4, temperature=0.6), use_tqdm=False)
    t0 = time.perf_counter()
    llm.generate(p["prompts"], sps, use_tqdm=False)
    elapsed = time.perf_counter() - t0
finally:
    llm.exit()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
total = sum(p["max_tokens"])
print("RESULT", json.dumps({"seconds": elapsed, "total_tokens": total}), flush=True)
"""


MAIN_WORKER = LAB_WORKER  # same API; only PYTHONPATH differs


VLLM_WORKER = r"""
import json, sys, time, gc
payload_path = sys.argv[1]
with open(payload_path) as f:
    p = json.load(f)
from vllm import LLM, SamplingParams
import vllm
print("PACKAGE", vllm.__file__, flush=True)
sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=n) for n in p["max_tokens"]]
reqs = [{"prompt_token_ids": x} for x in p["prompts"]]
llm = LLM(
    model=p["model"],
    enforce_eager=p["enforce_eager"],
    max_model_len=p["max_model_len"],
    gpu_memory_utilization=p["gpu_memory_utilization"],
    tensor_parallel_size=1,
    disable_log_stats=True,
)
try:
    if p["warmup"]:
        llm.generate(["warmup"], SamplingParams(max_tokens=4, temperature=0.6), use_tqdm=False)
    t0 = time.perf_counter()
    llm.generate(reqs, sps, use_tqdm=False)
    elapsed = time.perf_counter() - t0
finally:
    del llm
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
total = sum(p["max_tokens"])
print("RESULT", json.dumps({"seconds": elapsed, "total_tokens": total, "version": getattr(vllm, "__version__", "?")}), flush=True)
"""


def _payload(model, prompts, max_tokens, args, enforce_eager: bool) -> dict:
    return {
        "model": model,
        "prompts": prompts,
        "max_tokens": max_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": enforce_eager,
        "warmup": args.warmup,
    }


def run_lab(model: str, prompts, max_tokens, args: argparse.Namespace) -> dict:
    enforce_eager = resolve_enforce_eager(args)
    print(f"  backend={os.environ.get('NANOVLLM_ATTN_BACKEND', 'auto')} enforce_eager={enforce_eager}")
    env = {
        "PYTHONPATH": str(ROOT),
        "NANOVLLM_ATTN_BACKEND": os.environ.get("NANOVLLM_ATTN_BACKEND", "torch"),
    }
    result = _run_isolated_worker(
        LAB_WORKER, _payload(model, prompts, max_tokens, args, enforce_eager), env=env
    )
    elapsed = result["seconds"]
    total = result["total_tokens"]
    return {
        "engine": "lab",
        "label": f"lab SDPA/torch (enforce_eager={enforce_eager})",
        "seconds": round(elapsed, 4),
        "total_tokens": total,
        "tok_per_s": round(total / elapsed, 2) if elapsed else 0.0,
        "package": result.get("package"),
    }


def run_main_subprocess(model: str, prompts, max_tokens, args: argparse.Namespace) -> dict:
    main_path = Path(os.path.expanduser(args.main_path)).resolve()
    if not (main_path / "nanovllm").is_dir():
        raise FileNotFoundError(
            f"main engine not found at {main_path}. Create with:\n"
            f"  git worktree add {main_path} main"
        )
    enforce_eager = resolve_enforce_eager(args)
    env = {"PYTHONPATH": str(main_path)}
    # Clear lab-specific backend so main uses its own flash-attn path.
    env.pop("NANOVLLM_ATTN_BACKEND", None)
    result = _run_isolated_worker(
        MAIN_WORKER, _payload(model, prompts, max_tokens, args, enforce_eager), env=env
    )
    elapsed = result["seconds"]
    total = result["total_tokens"]
    return {
        "engine": "main",
        "label": f"git main @ {main_path}",
        "seconds": round(elapsed, 4),
        "total_tokens": total,
        "tok_per_s": round(total / elapsed, 2) if elapsed else 0.0,
        "package": result.get("package"),
    }


def run_vllm(model: str, prompts, max_tokens, args: argparse.Namespace) -> dict:
    enforce_eager = resolve_enforce_eager(args)
    print(f"  enforce_eager={enforce_eager} gpu_memory_utilization={args.gpu_memory_utilization}")
    result = _run_isolated_worker(
        VLLM_WORKER, _payload(model, prompts, max_tokens, args, enforce_eager), env=None
    )
    elapsed = result["seconds"]
    total = result["total_tokens"]
    return {
        "engine": "vllm",
        "label": f"vLLM {result.get('version', '?')}",
        "seconds": round(elapsed, 4),
        "total_tokens": total,
        "tok_per_s": round(total / elapsed, 2) if elapsed else 0.0,
        "package": result.get("package"),
    }


RUNNERS = {
    "lab": run_lab,
    "main": run_main_subprocess,
    "vllm": run_vllm,
}


def print_table(rows: list[dict]) -> None:
    if not rows:
        return
    ref = next((r for r in rows if r["engine"] == "lab" and not r.get("error")), None)
    if ref is None:
        ref = next((r for r in rows if not r.get("error")), rows[0])
    ref_tps = (ref.get("tok_per_s") or 0.0) or 1.0
    print("\n======== engine comparison ========")
    print(f"(relative to {ref.get('engine', '?')})")
    print(f"{'engine':10} {'tok/s':>10} {'sec':>8} {'vs_ref':>8}  label")
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

    # Default no-flash path
    if "NANOVLLM_ATTN_BACKEND" not in os.environ:
        os.environ["NANOVLLM_ATTN_BACKEND"] = "torch"

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    for e in engines:
        if e not in RUNNERS:
            raise SystemExit(f"unknown engine {e!r}; choose from lab,main,vllm")

    enforce_eager = resolve_enforce_eager(args)
    prompts, max_tokens = make_workload(args)
    print(
        f"workload: num_seqs={args.num_seqs} max_input={args.max_input_len} "
        f"max_output={args.max_output_len} enforce_eager={enforce_eager} "
        f"attn={os.environ.get('NANOVLLM_ATTN_BACKEND')} "
        f"gpu_mem_util={args.gpu_memory_utilization}"
    )

    rows = []
    for name in engines:
        print(f"\n=== {name} ===")
        _cleanup_cuda()
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
        finally:
            _cleanup_cuda()

    print_table(rows)
    if args.json_out:
        path = os.path.expanduser(args.json_out)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"model": model, "results": rows}, f, indent=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
