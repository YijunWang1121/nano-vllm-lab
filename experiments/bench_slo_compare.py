#!/usr/bin/env python3
"""Full-config lab vs vLLM SLO / latency / throughput compare.

Default: turn ON the same major optimizations on both sides:
  - CUDA graphs (enforce_eager=False)
  - prefix caching
  - chunked prefill + preemption (lab); vLLM defaults for V0
  - matched FlashAttention (vllm_flash / FLASH_ATTN)

Metrics (offline concurrent batch unless noted):
  - TTFT: admit → first output token
  - TPOT: (finish − first) / (out_tokens − 1)
  - E2E latency per request
  - output tok/s, request/s
  - SLO hit rates (configurable TTFT/TPOT thresholds)
  - batch size mean/max over engine steps (lab) / approx (vLLM)
  - KV pool: total blocks + peak used (lab); vLLM reports #blocks at init

    python experiments/bench_slo_compare.py --warmup --workload shared_prefix
    bash experiments/run_slo_compare.sh
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from random import randint, seed


ROOT = Path(__file__).resolve().parents[1]


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "p99": None, "min": None, "max": None}
    return {
        "n": len(xs),
        "mean": round(statistics.fmean(xs), 6),
        "p50": round(_pct(xs, 50) or 0.0, 6),
        "p90": round(_pct(xs, 90) or 0.0, 6),
        "p99": round(_pct(xs, 99) or 0.0, 6),
        "min": round(min(xs), 6),
        "max": round(max(xs), 6),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model",
        default=os.path.expanduser(os.environ.get("NANOVLLM_TEST_MODEL", "~/huggingface/Qwen3-0.6B/")),
    )
    p.add_argument("--engines", default="lab,vllm")
    p.add_argument(
        "--workload",
        default="shared_prefix",
        choices=("shared_prefix", "independent", "single_stream"),
        help="shared_prefix exercises prefix cache; independent = no shared tokens; "
        "single_stream = one request at a time (latency SLO).",
    )
    p.add_argument("--num-seqs", type=int, default=32)
    p.add_argument("--prefix-len", type=int, default=512)
    p.add_argument("--suffix-len", type=int, default=128)
    p.add_argument("--input-len", type=int, default=256, help="For independent / single_stream")
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", action="store_true")
    p.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Default False = CUDA graphs ON (full config).",
    )
    p.add_argument(
        "--prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default True (full config).",
    )
    p.add_argument("--slo-ttft-ms", type=float, default=500.0)
    p.add_argument("--slo-tpot-ms", type=float, default=50.0)
    p.add_argument("--json-out", default="")
    p.add_argument("--skip-missing", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def make_workload(args: argparse.Namespace) -> dict:
    seed(args.seed)
    vocab = 10000
    if args.workload == "shared_prefix":
        prefix = [randint(0, vocab) for _ in range(args.prefix_len)]
        prompts = [
            prefix + [randint(0, vocab) for _ in range(args.suffix_len)]
            for _ in range(args.num_seqs)
        ]
        return {
            "kind": "shared_prefix",
            "prompts": prompts,
            "max_tokens": [args.output_len] * args.num_seqs,
            "shared_prefix": prefix,
            "num_seqs": args.num_seqs,
        }
    if args.workload == "single_stream":
        prompts = [[randint(0, vocab) for _ in range(args.input_len)] for _ in range(args.num_seqs)]
        return {
            "kind": "single_stream",
            "prompts": prompts,
            "max_tokens": [args.output_len] * args.num_seqs,
            "shared_prefix": None,
            "num_seqs": args.num_seqs,
        }
    prompts = [[randint(0, vocab) for _ in range(args.input_len)] for _ in range(args.num_seqs)]
    return {
        "kind": "independent",
        "prompts": prompts,
        "max_tokens": [args.output_len] * args.num_seqs,
        "shared_prefix": None,
        "num_seqs": args.num_seqs,
    }


def resolve_attn() -> str:
    env = os.environ.get("NANOVLLM_ATTN_BACKEND", "").strip().lower()
    if env in ("torch", "sdpa", "eager", "flash", "vllm_flash"):
        return "torch" if env in ("torch", "sdpa", "eager") else ("flash" if env == "flash" else "vllm_flash")
    for name in ("vllm.vllm_flash_attn", "vllm_flash_attn"):
        try:
            __import__(name)
            return "vllm_flash"
        except Exception:
            pass
    try:
        __import__("flash_attn")
        return "flash"
    except Exception:
        return "torch"


LAB_WORKER = r"""
import json, sys, time, gc, os, statistics
payload_path = sys.argv[1]
with open(payload_path) as f:
    p = json.load(f)
os.environ["NANOVLLM_ATTN_BACKEND"] = p["attn_backend"]
from nanovllm import LLM, SamplingParams
import nanovllm
print("PACKAGE", nanovllm.__file__, flush=True)
try:
    from nanovllm.layers.attention import flash_attn_source
    print("flash_impl", flash_attn_source(), flush=True)
except Exception as e:
    print("flash_impl", None, e, flush=True)

llm = LLM(
    p["model"],
    enforce_eager=p["enforce_eager"],
    enable_prefix_caching=p["enable_prefix_caching"],
    enable_preemption=True,
    enable_chunked_prefill=True,
    max_model_len=p["max_model_len"],
    gpu_memory_utilization=p["gpu_memory_utilization"],
    tensor_parallel_size=1,
)
cfg = llm.model_runner.config
kv_total = int(cfg.num_kvcache_blocks)
kv_block = int(cfg.kvcache_block_size)
print("KV", json.dumps({"num_blocks": kv_total, "block_size": kv_block, "pool_tokens": kv_total * kv_block}), flush=True)
print("LLM ready", "enforce_eager", p["enforce_eager"], "prefix", p["enable_prefix_caching"], flush=True)

def cleanup():
    llm.exit()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass

try:
    if p["warmup"]:
        print("warmup ...", flush=True)
        llm.generate(["warmup"], SamplingParams(max_tokens=8, temperature=0.6), use_tqdm=False)
    shared = p.get("shared_prefix")
    if shared and p["enable_prefix_caching"] and p["prime_prefix"]:
        print("prime prefix ...", flush=True)
        llm.generate([shared], SamplingParams(max_tokens=1, temperature=0.6), use_tqdm=False)

    prompts = p["prompts"]
    max_tokens = p["max_tokens"]
    kind = p["workload_kind"]
    sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=n) for n in max_tokens]

    def run_batch(batch_prompts, batch_sps):
        t_submit, t_first, t_finish, out_lens = {}, {}, {}, {}
        batch_sizes, kv_used, decode_step_s = [], [], []
        prefill_step_s = []
        for prompt, sp in zip(batch_prompts, batch_sps):
            llm.add_request(prompt, sp)
            seq = llm.scheduler.waiting[-1]
            t_submit[seq.seq_id] = time.perf_counter()
            out_lens[seq.seq_id] = 0
        wall0 = time.perf_counter()
        while not llm.is_finished():
            bm = llm.scheduler.block_manager
            kv_used.append(len(bm.used_block_ids))
            seqs, is_prefill = llm.scheduler.schedule()
            batch_sizes.append(len(seqs))
            t0 = time.perf_counter()
            token_ids = llm.model_runner.call("run", seqs, is_prefill)
            t1 = time.perf_counter()
            llm.scheduler.postprocess(seqs, token_ids, is_prefill)
            dt = t1 - t0
            if is_prefill:
                prefill_step_s.append(dt)
            else:
                decode_step_s.append(dt)
            for seq in seqs:
                sid = seq.seq_id
                n = seq.num_completion_tokens
                if sid not in t_first and n >= 1:
                    t_first[sid] = t1
                out_lens[sid] = n
                if seq.is_finished:
                    t_finish[sid] = t1
        wall = time.perf_counter() - wall0
        return {
            "wall": wall,
            "t_submit": t_submit,
            "t_first": t_first,
            "t_finish": t_finish,
            "out_lens": out_lens,
            "batch_sizes": batch_sizes,
            "kv_used": kv_used,
            "decode_step_s": decode_step_s,
            "prefill_step_s": prefill_step_s,
        }

    if kind == "single_stream":
        parts = []
        for prompt, sp in zip(prompts, sps):
            parts.append(run_batch([prompt], [sp]))
        # merge
        wall = sum(x["wall"] for x in parts)
        t_submit, t_first, t_finish, out_lens = {}, {}, {}, {}
        batch_sizes, kv_used, decode_step_s, prefill_step_s = [], [], [], []
        for x in parts:
            t_submit.update(x["t_submit"])
            t_first.update(x["t_first"])
            t_finish.update(x["t_finish"])
            out_lens.update(x["out_lens"])
            batch_sizes.extend(x["batch_sizes"])
            kv_used.extend(x["kv_used"])
            decode_step_s.extend(x["decode_step_s"])
            prefill_step_s.extend(x["prefill_step_s"])
        raw = {
            "wall": wall,
            "t_submit": t_submit,
            "t_first": t_first,
            "t_finish": t_finish,
            "out_lens": out_lens,
            "batch_sizes": batch_sizes,
            "kv_used": kv_used,
            "decode_step_s": decode_step_s,
            "prefill_step_s": prefill_step_s,
        }
    else:
        print("timed generate ...", flush=True)
        raw = run_batch(prompts, sps)

    ttft, tpot, e2e = [], [], []
    total_out = 0
    for sid, t_sub in raw["t_submit"].items():
        tf = raw["t_first"].get(sid)
        te = raw["t_finish"].get(sid)
        n = int(raw["out_lens"].get(sid, 0))
        total_out += n
        if tf is not None:
            ttft.append(tf - t_sub)
        if te is not None:
            e2e.append(te - t_sub)
            if tf is not None and n > 1:
                tpot.append((te - tf) / (n - 1))
    peak_kv = max(raw["kv_used"]) if raw["kv_used"] else 0
    result = {
        "engine": "lab",
        "wall_s": raw["wall"],
        "num_requests": len(raw["t_submit"]),
        "total_output_tokens": total_out,
        "output_tok_per_s": (total_out / raw["wall"]) if raw["wall"] else 0.0,
        "request_per_s": (len(raw["t_submit"]) / raw["wall"]) if raw["wall"] else 0.0,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "e2e_s": e2e,
        "batch_size": raw["batch_sizes"],
        "decode_step_s": raw["decode_step_s"],
        "prefill_step_s": raw["prefill_step_s"],
        "kv": {
            "num_blocks": kv_total,
            "block_size": kv_block,
            "pool_tokens": kv_total * kv_block,
            "peak_used_blocks": peak_kv,
            "peak_used_frac": (peak_kv / kv_total) if kv_total else None,
        },
        "config": {
            "enforce_eager": p["enforce_eager"],
            "enable_prefix_caching": p["enable_prefix_caching"],
            "attn": p["attn_backend"],
        },
    }
    print("RESULT", json.dumps(result), flush=True)
finally:
    cleanup()
"""


VLLM_WORKER = r"""
import json, sys, time, gc, os
os.environ.setdefault("VLLM_USE_V1", "0")
payload_path = sys.argv[1]
with open(payload_path) as f:
    p = json.load(f)
attn = p["attn_backend"]
if attn in ("torch", "sdpa", "eager"):
    os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
else:
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
print("loading vLLM ...", flush=True)
from vllm import LLM, SamplingParams
import vllm
print("PACKAGE", vllm.__file__, flush=True)
print("vllm", getattr(vllm, "__version__", "?"),
      "VLLM_ATTENTION_BACKEND", os.environ.get("VLLM_ATTENTION_BACKEND"), flush=True)

llm = None
llm = LLM(
    model=p["model"],
    enforce_eager=p["enforce_eager"],
    max_model_len=p["max_model_len"],
    gpu_memory_utilization=p["gpu_memory_utilization"],
    tensor_parallel_size=1,
    disable_log_stats=False,
    enable_prefix_caching=p["enable_prefix_caching"],
)
print("LLM ready", flush=True)

def cleanup():
    global llm
    if llm is not None:
        del llm
        llm = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass

try:
    if p["warmup"]:
        print("warmup ...", flush=True)
        llm.generate(["warmup"], SamplingParams(max_tokens=8, temperature=0.6), use_tqdm=False)
    shared = p.get("shared_prefix")
    if shared and p["enable_prefix_caching"] and p["prime_prefix"]:
        print("prime prefix ...", flush=True)
        llm.generate([{"prompt_token_ids": shared}], SamplingParams(max_tokens=1, temperature=0.6), use_tqdm=False)

    prompts = p["prompts"]
    max_tokens = p["max_tokens"]
    kind = p["workload_kind"]
    sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=n) for n in max_tokens]

    def one_batch(batch_prompts, batch_sps):
        reqs = [{"prompt_token_ids": x} for x in batch_prompts]
        # Wall clock around generate; per-request metrics from RequestOutput.metrics when present.
        t_wall0 = time.perf_counter()
        t_admit = t_wall0
        outs = llm.generate(reqs, batch_sps, use_tqdm=False)
        wall = time.perf_counter() - t_wall0
        ttft, tpot, e2e, out_lens = [], [], [], []
        for o in outs:
            n = len(o.outputs[0].token_ids) if o.outputs else 0
            out_lens.append(n)
            m = getattr(o, "metrics", None)
            if m is not None and getattr(m, "first_token_time", None) and getattr(m, "arrival_time", None):
                # vLLM timestamps are absolute time.time()-style in some versions; prefer diffs.
                arr = m.arrival_time
                first = m.first_token_time
                fin = getattr(m, "finished_time", None) or time.time()
                if first and arr:
                    ttft.append(first - arr)
                if fin and first and n > 1:
                    tpot.append((fin - first) / (n - 1))
                if fin and arr:
                    e2e.append(fin - arr)
            else:
                # Fallback: only wall-level aggregates (no true TTFT). Mark empty ttft.
                pass
        return {
            "wall": wall,
            "ttft": ttft,
            "tpot": tpot,
            "e2e": e2e,
            "out_lens": out_lens,
            "admit": t_admit,
        }

    if kind == "single_stream":
        walls, ttft, tpot, e2e, out_lens = 0.0, [], [], [], []
        for prompt, sp in zip(prompts, sps):
            r = one_batch([prompt], [sp])
            walls += r["wall"]
            ttft.extend(r["ttft"])
            tpot.extend(r["tpot"])
            e2e.extend(r["e2e"])
            out_lens.extend(r["out_lens"])
        wall = walls
    else:
        print("timed generate ...", flush=True)
        r = one_batch(prompts, sps)
        wall, ttft, tpot, e2e, out_lens = r["wall"], r["ttft"], r["tpot"], r["e2e"], r["out_lens"]
        # If metrics missing, approximate TTFT/E2E from wall (same for all) — mark approximate.
        if not ttft and wall > 0:
            # Conservative: cannot recover per-request TTFT; leave empty and report wall only.
            pass

    total_out = sum(out_lens)
    result = {
        "engine": "vllm",
        "version": getattr(vllm, "__version__", "?"),
        "wall_s": wall,
        "num_requests": len(prompts),
        "total_output_tokens": total_out,
        "output_tok_per_s": (total_out / wall) if wall else 0.0,
        "request_per_s": (len(prompts) / wall) if wall else 0.0,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "e2e_s": e2e,
        "batch_size": [],  # not exposed in offline LLM API
        "decode_step_s": [],
        "prefill_step_s": [],
        "kv": {
            "note": "see vLLM init logs for # cuda blocks; peak used not exposed here",
        },
        "metrics_source": "RequestOutput.metrics" if ttft else "wall_only_partial",
        "config": {
            "enforce_eager": p["enforce_eager"],
            "enable_prefix_caching": p["enable_prefix_caching"],
            "attn": os.environ.get("VLLM_ATTENTION_BACKEND"),
        },
    }
    print("RESULT", json.dumps(result), flush=True)
finally:
    cleanup()
"""


def _run_worker(src: str, payload: dict, quiet: bool) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        path = f.name
    env = os.environ.copy()
    if payload.get("engine") == "lab":
        env["PYTHONPATH"] = str(ROOT)
        env["NANOVLLM_ATTN_BACKEND"] = payload["attn_backend"]
    proc = subprocess.Popen(
        [sys.executable, "-c", src, path],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)
        if (not quiet) or line.startswith(("RESULT ", "PACKAGE ", "KV ", "flash_impl", "LLM ready", "vllm ")):
            print(f"  | {line}", end="" if line.endswith("\n") else "\n", flush=True)
    rc = proc.wait()
    try:
        os.unlink(path)
    except OSError:
        pass
    out = "".join(lines)
    if rc != 0:
        raise RuntimeError(f"worker exit {rc}:\n{out[-4000:]}")
    for line in out.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    raise RuntimeError(f"no RESULT:\n{out[-2000:]}")


def summarize(raw: dict, slo_ttft_ms: float, slo_tpot_ms: float) -> dict:
    ttft = [x * 1000 for x in raw.get("ttft_s") or []]
    tpot = [x * 1000 for x in raw.get("tpot_s") or []]
    e2e = [x * 1000 for x in raw.get("e2e_s") or []]
    bs = raw.get("batch_size") or []
    return {
        "engine": raw.get("engine"),
        "version": raw.get("version"),
        "config": raw.get("config"),
        "wall_s": round(raw["wall_s"], 4),
        "num_requests": raw["num_requests"],
        "total_output_tokens": raw["total_output_tokens"],
        "output_tok_per_s": round(raw["output_tok_per_s"], 2),
        "request_per_s": round(raw["request_per_s"], 3),
        "ttft_ms": _stats(ttft),
        "tpot_ms": _stats(tpot),
        "e2e_ms": _stats(e2e),
        "slo": {
            "ttft_lt_ms": slo_ttft_ms,
            "tpot_lt_ms": slo_tpot_ms,
            "ttft_attainment": round(sum(1 for x in ttft if x <= slo_ttft_ms) / len(ttft), 4) if ttft else None,
            "tpot_attainment": round(sum(1 for x in tpot if x <= slo_tpot_ms) / len(tpot), 4) if tpot else None,
        },
        "batch": {
            "mean": round(statistics.fmean(bs), 3) if bs else None,
            "max": max(bs) if bs else None,
            "n_steps": len(bs),
        },
        "kv": raw.get("kv"),
        "metrics_source": raw.get("metrics_source", "lab_step_loop"),
        "decode_step_ms": _stats([x * 1000 for x in (raw.get("decode_step_s") or [])]),
        "prefill_step_ms": _stats([x * 1000 for x in (raw.get("prefill_step_s") or [])]),
    }


def print_summary(rows: list[dict]) -> None:
    print("\n======== full-config SLO compare ========")
    for r in rows:
        if r.get("error"):
            print(f"{r['engine']:6} FAIL {r['error'][:80]}")
            continue
        print(f"\n--- {r['engine']} ---")
        print(f"  config: {r.get('config')}")
        print(f"  wall: {r['wall_s']}s  out_tok/s: {r['output_tok_per_s']}  req/s: {r['request_per_s']}")
        print(f"  TTFT ms: {r['ttft_ms']}")
        print(f"  TPOT ms: {r['tpot_ms']}")
        print(f"  E2E  ms: {r['e2e_ms']}")
        print(f"  SLO: {r['slo']}")
        print(f"  batch: {r['batch']}")
        print(f"  kv: {r['kv']}")


def main():
    args = build_argparser().parse_args()
    model = os.path.expanduser(args.model)
    if not os.path.isdir(model):
        raise SystemExit(f"model dir not found: {model}")
    attn = resolve_attn()
    os.environ["NANOVLLM_ATTN_BACKEND"] = attn
    wl = make_workload(args)
    print(
        f"full-config: enforce_eager={args.enforce_eager} prefix_caching={args.prefix_caching} "
        f"attn={attn} workload={args.workload} num_seqs={wl['num_seqs']} out={args.output_len}"
    )
    print(
        f"SLO thresholds: TTFT<{args.slo_ttft_ms}ms TPOT<{args.slo_tpot_ms}ms"
    )

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    rows = []
    for name in engines:
        print(f"\n=== {name} ===")
        payload = {
            "engine": name,
            "model": model,
            "prompts": wl["prompts"],
            "max_tokens": wl["max_tokens"],
            "shared_prefix": wl.get("shared_prefix"),
            "workload_kind": wl["kind"],
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "enable_prefix_caching": args.prefix_caching,
            "attn_backend": attn,
            "warmup": args.warmup,
            "prime_prefix": bool(wl.get("shared_prefix")) and args.prefix_caching,
        }
        try:
            if name == "lab":
                raw = _run_worker(LAB_WORKER, payload, args.quiet)
            elif name == "vllm":
                raw = _run_worker(VLLM_WORKER, payload, args.quiet)
            else:
                raise ValueError(f"unknown engine {name}")
            rows.append(summarize(raw, args.slo_ttft_ms, args.slo_tpot_ms))
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"FAILED: {msg}")
            if not args.skip_missing:
                raise
            rows.append({"engine": name, "error": msg})

    print_summary(rows)
    if args.json_out:
        path = os.path.expanduser(args.json_out)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": model,
                    "workload": {
                        "kind": wl["kind"],
                        "num_seqs": wl["num_seqs"],
                        "output_len": args.output_len,
                        "prefix_len": args.prefix_len if args.workload == "shared_prefix" else None,
                        "suffix_len": args.suffix_len if args.workload == "shared_prefix" else None,
                        "input_len": args.input_len if args.workload != "shared_prefix" else None,
                        "seed": args.seed,
                    },
                    "full_config": {
                        "enforce_eager": args.enforce_eager,
                        "enable_prefix_caching": args.prefix_caching,
                        "attn": attn,
                        "gpu_memory_utilization": args.gpu_memory_utilization,
                    },
                    "slo_thresholds_ms": {"ttft": args.slo_ttft_ms, "tpot": args.slo_tpot_ms},
                    "results": rows,
                },
                f,
                indent=2,
            )
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
