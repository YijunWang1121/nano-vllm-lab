#!/usr/bin/env python3
"""Multi-config engine sweep: several shapes × repeats → mean±std tok/s.

Uses the same shared-token / same-kernel settings as compare_engines.py.

    python experiments/sweep_engine_compare.py --suite default --repeats 3 --warmup
    python experiments/sweep_engine_compare.py --suite quick --repeats 1
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compare_engines import (  # noqa: E402
    build_argparser as build_compare_argparser,
    run_comparison,
)


@dataclass(frozen=True)
class Case:
    name: str
    num_seqs: int
    input_len: int
    output_len: int
    note: str = ""


# Fixed lengths (not random) so shapes are apples-to-apples across engines/repeats.
SUITES: dict[str, list[Case]] = {
    "quick": [
        Case("bs8_in128_out64", 8, 128, 64, "smoke"),
        Case("bs32_in256_out128", 32, 256, 128, "balanced"),
    ],
    "default": [
        Case("bs1_in128_out256", 1, 128, 256, "single-stream decode-heavy"),
        Case("bs8_in128_out256", 8, 128, 256, "small-batch decode-heavy"),
        Case("bs8_in1024_out32", 8, 1024, 32, "prefill-heavy"),
        Case("bs32_in64_out64", 32, 64, 64, "short prompts"),
        Case("bs32_in256_out128", 32, 256, 128, "balanced mid"),
        Case("bs64_in256_out128", 64, 256, 128, "larger batch"),
        Case("bs16_in2048_out64", 16, 2048, 64, "long context"),
        Case("bs64_in512_out32", 64, 512, 32, "prefill-ish large batch"),
    ],
    "stress": [
        Case("bs1_in128_out512", 1, 128, 512, "long decode"),
        Case("bs4_in2048_out128", 4, 2048, 128, "long ctx + decode"),
        Case("bs16_in1024_out128", 16, 1024, 128, "mid/long"),
        Case("bs32_in512_out128", 32, 512, 128, "classic"),
        Case("bs64_in128_out128", 64, 128, 128, "wide batch short ctx"),
        Case("bs64_in512_out128", 64, 512, 128, "wide + longer prompt"),
        Case("bs128_in128_out64", 128, 128, 64, "very wide (may OOM)"),
    ],
}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="default", choices=sorted(SUITES.keys()))
    p.add_argument("--repeats", type=int, default=3, help="Repeats per case with different seeds")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument(
        "--model",
        default=os.path.expanduser(os.environ.get("NANOVLLM_TEST_MODEL", "~/huggingface/Qwen3-0.6B/")),
    )
    p.add_argument("--engines", default="lab,vllm")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default True for fair kernel-matched compare",
    )
    p.add_argument("--warmup", action="store_true", default=True)
    p.add_argument("--no-warmup", action="store_false", dest="warmup")
    p.add_argument("--skip-missing", action="store_true", default=True)
    p.add_argument("--quiet", action="store_true", default=True)
    p.add_argument("--verbose", action="store_false", dest="quiet")
    p.add_argument(
        "--json-out",
        default="",
        help="Summary JSON path (default under ~/engine_compare_results/)",
    )
    p.add_argument(
        "--cases",
        default="",
        help="Comma subset of case names (default: all in suite)",
    )
    return p


def _mean_std(xs: list[float]) -> tuple[float | None, float | None]:
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.fmean(xs), statistics.stdev(xs)


def _aggregate_case(case: Case, runs: list[dict]) -> dict:
    by_engine: dict[str, list[float]] = {}
    errors: dict[str, list[str]] = {}
    for run in runs:
        for row in run.get("results", []):
            eng = row["engine"]
            if row.get("error"):
                errors.setdefault(eng, []).append(row["error"])
                continue
            by_engine.setdefault(eng, []).append(float(row["tok_per_s"]))

    engines = {}
    for eng, vals in by_engine.items():
        mean, std = _mean_std(vals)
        engines[eng] = {
            "tok_per_s_mean": None if mean is None else round(mean, 2),
            "tok_per_s_std": None if std is None else round(std, 2),
            "n": len(vals),
            "samples": [round(v, 2) for v in vals],
        }
    for eng, errs in errors.items():
        engines.setdefault(eng, {"tok_per_s_mean": None, "tok_per_s_std": None, "n": 0, "samples": []})
        engines[eng]["errors"] = errs[:3]

    lab = engines.get("lab", {}).get("tok_per_s_mean")
    for eng, stats in engines.items():
        m = stats.get("tok_per_s_mean")
        if lab and m:
            stats["vs_lab"] = round(m / lab, 3)
        else:
            stats["vs_lab"] = None

    return {
        "case": asdict(case),
        "engines": engines,
        "runs": runs,
    }


def print_summary(agg_rows: list[dict]) -> None:
    print("\n======== sweep summary (mean ± std tok/s) ========")
    hdr = f"{'case':28} {'bs':>4} {'in':>5} {'out':>5} {'lab':>14} {'vllm':>14} {'vs_lab':>8}"
    print(hdr)
    print("-" * len(hdr))
    for row in agg_rows:
        c = row["case"]
        lab = row["engines"].get("lab", {})
        vllm = row["engines"].get("vllm", {})
        lab_s = (
            f"{lab['tok_per_s_mean']:.1f}±{lab['tok_per_s_std']:.1f}"
            if lab.get("tok_per_s_mean") is not None
            else "FAIL"
        )
        vllm_s = (
            f"{vllm['tok_per_s_mean']:.1f}±{vllm['tok_per_s_std']:.1f}"
            if vllm.get("tok_per_s_mean") is not None
            else "FAIL"
        )
        vs = vllm.get("vs_lab")
        vs_s = f"{vs:.2f}x" if vs is not None else "-"
        print(
            f"{c['name']:28} {c['num_seqs']:4d} {c['input_len']:5d} {c['output_len']:5d} "
            f"{lab_s:>14} {vllm_s:>14} {vs_s:>8}"
        )


def main():
    args = build_argparser().parse_args()
    cases = SUITES[args.suite]
    if args.cases.strip():
        want = {x.strip() for x in args.cases.split(",") if x.strip()}
        cases = [c for c in cases if c.name in want]
        missing = want - {c.name for c in cases}
        if missing:
            raise SystemExit(f"unknown cases: {sorted(missing)}")

    out = args.json_out or os.path.expanduser(
        f"~/engine_compare_results/{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_sweep.json"
    )
    Path(os.path.expanduser(out)).parent.mkdir(parents=True, exist_ok=True)

    # Base Namespace from compare_engines defaults, then override.
    cmp_p = build_compare_argparser()
    cmp_args = cmp_p.parse_args([])
    cmp_args.model = args.model
    cmp_args.engines = args.engines
    cmp_args.max_model_len = args.max_model_len
    cmp_args.gpu_memory_utilization = args.gpu_memory_utilization
    cmp_args.enforce_eager = args.enforce_eager
    cmp_args.warmup = args.warmup
    cmp_args.skip_missing = args.skip_missing
    cmp_args.quiet = args.quiet
    cmp_args.json_out = ""

    print(
        f"sweep suite={args.suite} cases={len(cases)} repeats={args.repeats} "
        f"engines={args.engines} enforce_eager={args.enforce_eager} warmup={args.warmup}"
    )
    t0 = time.perf_counter()
    aggregated: list[dict] = []

    for i, case in enumerate(cases, 1):
        print(f"\n######## [{i}/{len(cases)}] {case.name}  "
              f"bs={case.num_seqs} in={case.input_len} out={case.output_len}  ({case.note})")
        runs = []
        for r in range(args.repeats):
            seed = args.base_seed + r
            print(f"\n---- repeat {r+1}/{args.repeats} seed={seed} ----")
            cmp_args.num_seqs = case.num_seqs
            cmp_args.input_len = case.input_len
            cmp_args.output_len = case.output_len
            cmp_args.max_input_len = case.input_len
            cmp_args.max_output_len = case.output_len
            cmp_args.seed = seed
            try:
                payload = run_comparison(cmp_args)
            except Exception as exc:
                print(f"FAILED case run: {exc}")
                payload = {
                    "results": [
                        {"engine": e, "error": str(exc), "tok_per_s": 0.0}
                        for e in args.engines.split(",")
                    ],
                    "workload": {"seed": seed},
                }
            runs.append(
                {
                    "repeat": r,
                    "seed": seed,
                    "workload": payload.get("workload"),
                    "results": payload.get("results"),
                    "config": payload.get("config"),
                }
            )
        aggregated.append(_aggregate_case(case, runs))

    print_summary(aggregated)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite,
        "repeats": args.repeats,
        "base_seed": args.base_seed,
        "model": os.path.expanduser(args.model),
        "engines": args.engines,
        "enforce_eager": args.enforce_eager,
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "cases": aggregated,
    }
    path = os.path.expanduser(out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {path}  ({summary['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
