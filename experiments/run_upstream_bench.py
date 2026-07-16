#!/usr/bin/env python3
"""Upstream-style bench: mirror GeeeekExplorer/nano-vllm bench.py for lab vs vLLM.

Ref: https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py

Env (set by run_upstream_bench.sh):
  NANOVLLM_TEST_MODEL, NANOVLLM_ATTN_BACKEND
  UPSTREAM_BENCH_REPEATS, UPSTREAM_BENCH_GPU_UTIL, UPSTREAM_BENCH_BASE_SEED
  UPSTREAM_BENCH_JSON_OUT
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, stdev

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compare_engines import build_argparser, run_comparison  # noqa: E402


def main() -> None:
    repeats = int(os.environ.get("UPSTREAM_BENCH_REPEATS", "1"))
    base_seed = int(os.environ.get("UPSTREAM_BENCH_BASE_SEED", "0"))
    util = float(os.environ.get("UPSTREAM_BENCH_GPU_UTIL", "0.9"))
    model = os.path.expanduser(
        os.environ.get("NANOVLLM_TEST_MODEL", "~/huggingface/Qwen3-0.6B")
    )
    out = os.path.expanduser(
        os.environ.get(
            "UPSTREAM_BENCH_JSON_OUT",
            "~/engine_compare_results/upstream_bench.json",
        )
    )

    runs = []
    t0 = time.perf_counter()
    for r in range(repeats):
        seed = base_seed + r
        print(f"\n######## upstream bench repeat {r+1}/{repeats} seed={seed}", flush=True)
        p = build_argparser().parse_args([])
        p.model = model
        p.engines = "lab,vllm"
        p.num_seqs = 256
        p.min_input_len = 100
        p.max_input_len = 1024
        p.min_output_len = 100
        p.max_output_len = 1024
        p.input_len = None
        p.output_len = None
        p.max_model_len = 4096
        p.gpu_memory_utilization = util
        p.seed = seed
        p.warmup = True
        p.enforce_eager = False  # match upstream bench.py
        p.json_out = ""
        p.skip_missing = True
        p.quiet = True
        payload = run_comparison(p)
        runs.append({"repeat": r, "seed": seed, **payload})

    print("\n======== upstream-style bench summary ========")
    print(f"{'engine':8} {'tok/s':>12} {'sec':>10} {'vs_lab':>8}")
    last = runs[-1].get("results", [])
    lab_tps = next(
        (row.get("tok_per_s") for row in last if row.get("engine") == "lab" and not row.get("error")),
        None,
    )
    for row in last:
        eng = row.get("engine")
        if row.get("error"):
            print(f"{eng:8} FAIL {str(row['error'])[:60]}")
            continue
        tps = float(row.get("tok_per_s") or 0)
        sec = float(row.get("seconds") or 0)
        vs = (tps / lab_tps) if lab_tps else None
        vs_s = f"{vs:.2f}x" if vs else "-"
        print(f"{eng:8} {tps:12.2f} {sec:10.2f} {vs_s:>8}")

    means: dict[str, list[float]] = {}
    for run in runs:
        for row in run.get("results", []):
            if row.get("error"):
                continue
            means.setdefault(row["engine"], []).append(float(row["tok_per_s"]))
    if repeats > 1 and means:
        print("\nmean±std over repeats:")
        lab_m = fmean(means["lab"]) if means.get("lab") else None
        for eng, xs in means.items():
            m = fmean(xs)
            s = stdev(xs) if len(xs) > 1 else 0.0
            vs = (m / lab_m) if lab_m else None
            extra = f"  vs_lab={vs:.3f}x" if vs else ""
            print(f"  {eng}: {m:.1f}±{s:.1f} tok/s{extra}")

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study": "upstream_bench",
        "ref": "https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py",
        "config": {
            "num_seqs": 256,
            "min_input_len": 100,
            "max_input_len": 1024,
            "min_output_len": 100,
            "max_output_len": 1024,
            "enforce_eager": False,
            "max_model_len": 4096,
            "gpu_memory_utilization": util,
            "prefix_caching": False,
            "attn": os.environ.get("NANOVLLM_ATTN_BACKEND"),
            "repeats": repeats,
            "base_seed": base_seed,
            "model": model,
        },
        "upstream_readme_ref": {
            "hardware_claimed": "RTX 4070 Laptop 8GB",
            "vllm_tok_s": 1361.84,
            "nano_tok_s": 1434.13,
            "note": "Different GPU; compare method/ratio, not absolute tok/s to README.",
        },
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "runs": runs,
        "aggregates": {
            eng: {
                "tok_per_s_mean": round(fmean(xs), 2),
                "tok_per_s_std": round(stdev(xs), 2) if len(xs) > 1 else 0.0,
                "n": len(xs),
            }
            for eng, xs in means.items()
        },
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
