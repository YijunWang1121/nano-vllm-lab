#!/usr/bin/env python3
"""Fair throughput comparison: course lab vs upstream main vs vLLM.

Matched attention kernel + enforce_eager on both sides:

    # Same FlashAttention binary as vLLM (preferred for engine compare):
    export NANOVLLM_ATTN_BACKEND=vllm_flash   # lab imports vllm_flash_attn
    # vLLM worker sets VLLM_ATTENTION_BACKEND=FLASH_ATTN

    # pip flash-attn on lab (different .so than vLLM's bundled copy):
    export NANOVLLM_ATTN_BACKEND=flash

    # Fallback (not the same kernel): lab SDPA + vLLM XFORMERS
    export NANOVLLM_ATTN_BACKEND=torch

SDPA is incompatible with nano-vLLM CUDA graphs → use --enforce-eager with torch/SDPA.
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
    p.add_argument(
        "--min-input-len",
        type=int,
        default=32,
        help="Min random prompt length when --input-len is unset (upstream bench.py uses 100).",
    )
    p.add_argument(
        "--min-output-len",
        type=int,
        default=16,
        help="Min random max_tokens when --output-len is unset (upstream bench.py uses 100).",
    )
    p.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="Fixed prompt length for every sequence (overrides random min..max-input-len).",
    )
    p.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="Fixed max_tokens for every sequence (overrides random min..max-output-len).",
    )
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
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Only print high-level progress (hide worker load spam).",
    )
    return p


def resolve_enforce_eager(args: argparse.Namespace) -> bool:
    if args.enforce_eager is not None:
        return bool(args.enforce_eager)
    backend = os.environ.get("NANOVLLM_ATTN_BACKEND", "auto").strip().lower()
    if backend in ("torch", "sdpa", "eager"):
        return True
    return False


def make_workload(args: argparse.Namespace):
    """One shared synthetic batch for all engines (same prompt ids + max_tokens)."""
    seed(args.seed)
    n = args.num_seqs
    if args.input_len is not None:
        prompts = [[randint(0, 10000) for _ in range(args.input_len)] for _ in range(n)]
    else:
        lo_i = max(1, int(args.min_input_len))
        hi_i = max(lo_i, int(args.max_input_len))
        prompts = [
            [randint(0, 10000) for _ in range(randint(lo_i, hi_i))]
            for _ in range(n)
        ]
    if args.output_len is not None:
        max_tokens = [args.output_len] * n
    else:
        lo_o = max(1, int(args.min_output_len))
        hi_o = max(lo_o, int(args.max_output_len))
        max_tokens = [randint(lo_o, hi_o) for _ in range(n)]
    return prompts, max_tokens


def workload_fingerprint(prompts, max_tokens) -> dict:
    """Stable summary so logs prove lab/vllm saw the same batch."""
    import hashlib
    import struct

    h = hashlib.sha256()
    for p, n in zip(prompts, max_tokens):
        h.update(struct.pack("<II", n, len(p)))
        h.update(struct.pack(f"<{len(p)}I", *p))
    return {
        "num_seqs": len(prompts),
        "prompt_tokens": sum(len(p) for p in prompts),
        "output_tokens": sum(max_tokens),
        "sha256": h.hexdigest()[:16],
    }


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


def _run_isolated_worker(
    worker_src: str,
    payload: dict,
    env: dict | None = None,
    quiet: bool = False,
) -> dict:
    """Run one engine in a fresh process so GPU memory is released on exit."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        payload_path = f.name
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    # Stream logs live so model-load does not look hung; still collect for RESULT.
    proc = subprocess.Popen(
        [sys.executable, "-c", worker_src, payload_path],
        env=run_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    collected: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        collected.append(line)
        show = (not quiet) or line.startswith(("RESULT ", "PACKAGE ", "flash_impl", "FAILED"))
        if show:
            print(f"  | {line}", end="" if line.endswith("\n") else "\n", flush=True)
    rc = proc.wait()
    try:
        os.unlink(payload_path)
    except OSError:
        pass
    out = "".join(collected)
    if rc != 0:
        raise RuntimeError(f"worker failed (exit {rc}):\n{out[-3000:]}")
    package = ""
    result = None
    for line in out.splitlines():
        if line.startswith("PACKAGE "):
            package = line[len("PACKAGE ") :]
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT ") :])
    if result is None:
        raise RuntimeError(f"worker produced no RESULT:\n{out[-3000:]}")
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
print("attn", os.environ.get("NANOVLLM_ATTN_BACKEND"), "enforce_eager", p["enforce_eager"], flush=True)
try:
    from nanovllm.layers.attention import flash_attn_source
    print("flash_impl", flash_attn_source(), flush=True)
except Exception as exc:
    print("flash_impl", None, exc, flush=True)
print("loading LLM ...", flush=True)
sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=n) for n in p["max_tokens"]]
llm_kwargs = dict(
    enforce_eager=p["enforce_eager"],
    max_model_len=p["max_model_len"],
    gpu_memory_utilization=p["gpu_memory_utilization"],
    tensor_parallel_size=1,
)
# Align with vLLM worker: no prefix hits on random prompts.
try:
    llm = LLM(p["model"], enable_prefix_caching=False, **llm_kwargs)
except TypeError:
    llm = LLM(p["model"], **llm_kwargs)
print("LLM ready", flush=True)
try:
    if p["warmup"]:
        print("warmup ...", flush=True)
        llm.generate(["warmup"], SamplingParams(max_tokens=4, temperature=0.6), use_tqdm=False)
    print("timed generate ...", flush=True)
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
import json, sys, time, gc, os
# Prefer V0 engine on mixed envs; V1 often pulls leftover flashinfer/torch_c_dlpack from newer vLLM.
os.environ.setdefault("VLLM_USE_V1", "0")
payload_path = sys.argv[1]
with open(payload_path) as f:
    p = json.load(f)
# Match lab attention class when possible.
# vLLM CUDA+V0 rejects TORCH_SDPA; XFORMERS is the supported non-flash GPU backend.
attn = (p.get("attn_backend") or os.environ.get("NANOVLLM_ATTN_BACKEND") or "auto").strip().lower()
if attn in ("torch", "sdpa", "eager"):
    os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
elif attn in ("flash", "auto", "vllm_flash", "vllm-flash", "same"):
    # Pin FLASH_ATTN so vLLM uses its bundled vllm_flash_attn (same .so lab uses with vllm_flash).
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
print("loading vLLM ...", flush=True)
from vllm import LLM, SamplingParams
import vllm
print("PACKAGE", vllm.__file__, flush=True)
flash_impl = None
for name in ("vllm.vllm_flash_attn", "vllm_flash_attn"):
    try:
        mod = __import__(name, fromlist=["*"])
        flash_impl = f"{name} @ {getattr(mod, '__file__', '?')}"
        break
    except Exception:
        pass
print(
    "vllm", getattr(vllm, "__version__", "?"),
    "VLLM_USE_V1=", os.environ.get("VLLM_USE_V1"),
    "VLLM_ATTENTION_BACKEND=", os.environ.get("VLLM_ATTENTION_BACKEND", "<auto>"),
    "flash_impl", flash_impl,
    flush=True,
)
sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=n) for n in p["max_tokens"]]
reqs = [{"prompt_token_ids": x} for x in p["prompts"]]
llm = LLM(
    model=p["model"],
    enforce_eager=p["enforce_eager"],
    max_model_len=p["max_model_len"],
    gpu_memory_utilization=p["gpu_memory_utilization"],
    tensor_parallel_size=1,
    disable_log_stats=True,
    enable_prefix_caching=False,
)
print("LLM ready", flush=True)
try:
    if p["warmup"]:
        print("warmup ...", flush=True)
        llm.generate(["warmup"], SamplingParams(max_tokens=4, temperature=0.6), use_tqdm=False)
    print("timed generate ...", flush=True)
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
print("RESULT", json.dumps({
    "seconds": elapsed,
    "total_tokens": total,
    "version": getattr(vllm, "__version__", "?"),
    "vllm_attn": os.environ.get("VLLM_ATTENTION_BACKEND", "<auto>"),
}), flush=True)
"""


def _can_import(names: list[str]) -> bool:
    for name in names:
        try:
            __import__(name)
            return True
        except Exception:
            continue
    return False


def resolve_attn_backend() -> str:
    """Return vllm_flash|flash|torch. Prefer vLLM's bundled flash for same-kernel compare."""
    env = os.environ.get("NANOVLLM_ATTN_BACKEND", "").strip().lower()
    if env in ("torch", "sdpa", "eager"):
        return "torch"
    if env in ("vllm_flash", "vllm-flash", "same"):
        return "vllm_flash"
    if env == "flash":
        return "flash"
    # auto: same kernel as vLLM when possible
    if _can_import(["vllm.vllm_flash_attn", "vllm_flash_attn"]):
        return "vllm_flash"
    if _can_import(["flash_attn"]):
        return "flash"
    return "torch"


def _payload(model, prompts, max_tokens, args, enforce_eager: bool, attn_backend: str) -> dict:
    return {
        "model": model,
        "prompts": prompts,
        "max_tokens": max_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": enforce_eager,
        "warmup": args.warmup,
        "attn_backend": attn_backend,
    }


def run_lab(model: str, prompts, max_tokens, args: argparse.Namespace) -> dict:
    enforce_eager = resolve_enforce_eager(args)
    attn = resolve_attn_backend()
    print(f"  backend={attn} enforce_eager={enforce_eager}")
    env = {
        "PYTHONPATH": str(ROOT),
        "NANOVLLM_ATTN_BACKEND": attn,
    }
    result = _run_isolated_worker(
        LAB_WORKER,
        _payload(model, prompts, max_tokens, args, enforce_eager, attn),
        env=env,
        quiet=bool(getattr(args, "quiet", False)),
    )
    elapsed = result["seconds"]
    total = result["total_tokens"]
    kind = {
        "vllm_flash": "vllm_flash_attn (same .so as vLLM)",
        "flash": "pip flash-attn",
        "torch": "SDPA/torch",
    }.get(attn, attn)
    return {
        "engine": "lab",
        "label": f"lab {kind} (enforce_eager={enforce_eager})",
        "seconds": round(elapsed, 4),
        "total_tokens": total,
        "tok_per_s": round(total / elapsed, 2) if elapsed else 0.0,
        "package": result.get("package"),
        "attn_backend": attn,
    }


def run_main_subprocess(model: str, prompts, max_tokens, args: argparse.Namespace) -> dict:
    main_path = Path(os.path.expanduser(args.main_path)).resolve()
    if not (main_path / "nanovllm").is_dir():
        raise FileNotFoundError(
            f"main engine not found at {main_path}. Create with:\n"
            f"  git worktree add {main_path} main"
        )
    enforce_eager = resolve_enforce_eager(args)
    attn = resolve_attn_backend()
    env = {"PYTHONPATH": str(main_path)}
    # Clear lab-specific backend so main uses its own flash-attn path.
    env.pop("NANOVLLM_ATTN_BACKEND", None)
    result = _run_isolated_worker(
        MAIN_WORKER,
        _payload(model, prompts, max_tokens, args, enforce_eager, attn),
        env=env,
        quiet=bool(getattr(args, "quiet", False)),
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
    attn = resolve_attn_backend()
    vllm_attn = "XFORMERS" if attn == "torch" else "FLASH_ATTN"
    print(
        f"  attn={attn}→{vllm_attn} enforce_eager={enforce_eager} "
        f"gpu_memory_utilization={args.gpu_memory_utilization} prefix_caching=False"
    )
    result = _run_isolated_worker(
        VLLM_WORKER,
        _payload(model, prompts, max_tokens, args, enforce_eager, attn),
        env=None,
        quiet=bool(getattr(args, "quiet", False)),
    )
    elapsed = result["seconds"]
    total = result["total_tokens"]
    return {
        "engine": "vllm",
        "label": f"vLLM {result.get('version', '?')} {result.get('vllm_attn', vllm_attn)} (eager={enforce_eager})",
        "seconds": round(elapsed, 4),
        "total_tokens": total,
        "tok_per_s": round(total / elapsed, 2) if elapsed else 0.0,
        "package": result.get("package"),
        "attn_backend": attn,
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


def run_comparison(args: argparse.Namespace) -> dict:
    """Run one shared-workload comparison; returns {model, workload, results, config}."""
    model = os.path.expanduser(args.model)
    if not os.path.isdir(model):
        raise FileNotFoundError(f"model dir not found: {model}")

    attn = resolve_attn_backend()
    os.environ["NANOVLLM_ATTN_BACKEND"] = attn

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    for e in engines:
        if e not in RUNNERS:
            raise ValueError(f"unknown engine {e!r}; choose from lab,main,vllm")

    if args.input_len is not None and args.output_len is not None:
        need = args.input_len + args.output_len
        if need > args.max_model_len:
            raise ValueError(
                f"input_len+output_len={need} exceeds max_model_len={args.max_model_len}"
            )

    enforce_eager = resolve_enforce_eager(args)
    # Single shared batch: identical prompt_token_ids + max_tokens for every engine.
    prompts, max_tokens = make_workload(args)
    wl = workload_fingerprint(prompts, max_tokens)
    print(
        f"workload: num_seqs={wl['num_seqs']} prompt_tok={wl['prompt_tokens']} "
        f"output_tok={wl['output_tokens']} seed={args.seed} sha256={wl['sha256']} "
        f"(shared by all engines)"
    )
    if args.input_len is not None or args.output_len is not None:
        print(
            f"  fixed lens: input_len={args.input_len} output_len={args.output_len}"
        )
    print(
        f"config: enforce_eager={enforce_eager} attn={attn} (matched lab↔vLLM) "
        f"gpu_mem_util={args.gpu_memory_utilization} max_model_len={args.max_model_len}"
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
    return {
        "model": model,
        "workload": {
            **wl,
            "seed": args.seed,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "num_seqs": args.num_seqs,
        },
        "config": {
            "enforce_eager": enforce_eager,
            "attn": attn,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "engines": engines,
        },
        "results": rows,
    }


def main():
    args = build_argparser().parse_args()
    try:
        payload = run_comparison(args)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.json_out:
        path = os.path.expanduser(args.json_out)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
