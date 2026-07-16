#!/usr/bin/env python3
"""Merge eager vs CUDA-graph sweep JSONs and print a delta table.

    python experiments/merge_graph_ablation.py \\
      --eager ~/engine_compare_results/..._eager.json \\
      --graphs ~/engine_compare_results/..._graphs.json \\
      --json-out ~/engine_compare_results/..._graph_ablation.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _eng(row: dict, name: str) -> dict:
    return row.get("engines", {}).get(name, {})


def _mean(row: dict, name: str) -> float | None:
    v = _eng(row, name).get("tok_per_s_mean")
    return None if v is None else float(v)


def _std(row: dict, name: str) -> float | None:
    v = _eng(row, name).get("tok_per_s_std")
    return None if v is None else float(v)


def _fmt(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "FAIL"
    if std is None:
        return f"{mean:.1f}"
    return f"{mean:.1f}±{std:.1f}"


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return round(a / b, 3)


def merge(eager: dict, graphs: dict) -> dict:
    by_name_e = {r["case"]["name"]: r for r in eager.get("cases", [])}
    by_name_g = {r["case"]["name"]: r for r in graphs.get("cases", [])}
    names = [r["case"]["name"] for r in eager.get("cases", [])]
    for n in by_name_g:
        if n not in by_name_e:
            names.append(n)

    rows = []
    for name in names:
        e = by_name_e.get(name)
        g = by_name_g.get(name)
        case = (e or g)["case"]
        lab_e, lab_g = _mean(e or {}, "lab"), _mean(g or {}, "lab")
        vllm_e, vllm_g = _mean(e or {}, "vllm"), _mean(g or {}, "vllm")
        row = {
            "case": case,
            "eager": {
                "lab": _mean(e or {}, "lab"),
                "lab_std": _std(e or {}, "lab"),
                "vllm": _mean(e or {}, "vllm"),
                "vllm_std": _std(e or {}, "vllm"),
                "vs_lab": _eng(e or {}, "vllm").get("vs_lab"),
            },
            "graphs": {
                "lab": _mean(g or {}, "lab"),
                "lab_std": _std(g or {}, "lab"),
                "vllm": _mean(g or {}, "vllm"),
                "vllm_std": _std(g or {}, "vllm"),
                "vs_lab": _eng(g or {}, "vllm").get("vs_lab"),
            },
            "lab_speedup_graphs_over_eager": _ratio(lab_g, lab_e),
            "vllm_speedup_graphs_over_eager": _ratio(vllm_g, vllm_e),
            "vs_lab_eager": _eng(e or {}, "vllm").get("vs_lab"),
            "vs_lab_graphs": _eng(g or {}, "vllm").get("vs_lab"),
        }
        rows.append(row)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study": "graph_ablation",
        "note": "Same shapes/flash/prefix=OFF; only enforce_eager toggled.",
        "eager_source": eager.get("created_utc"),
        "graphs_source": graphs.get("created_utc"),
        "repeats": eager.get("repeats"),
        "model": eager.get("model") or graphs.get("model"),
        "cases": rows,
    }


def print_table(merged: dict) -> None:
    print("\n======== graph ablation (prefix OFF; only graphs ON/OFF) ========")
    hdr = (
        f"{'case':28} "
        f"{'lab eager':>12} {'lab graph':>12} {'lab×':>6} "
        f"{'vllm eager':>12} {'vllm graph':>12} {'vllm×':>6} "
        f"{'vsE':>6} {'vsG':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for row in merged["cases"]:
        c = row["case"]
        e, g = row["eager"], row["graphs"]
        lab_x = row["lab_speedup_graphs_over_eager"]
        vllm_x = row["vllm_speedup_graphs_over_eager"]
        vs_e = row["vs_lab_eager"]
        vs_g = row["vs_lab_graphs"]
        print(
            f"{c['name']:28} "
            f"{_fmt(e['lab'], e['lab_std']):>12} {_fmt(g['lab'], g['lab_std']):>12} "
            f"{(f'{lab_x:.2f}x' if lab_x else '-'):>6} "
            f"{_fmt(e['vllm'], e['vllm_std']):>12} {_fmt(g['vllm'], g['vllm_std']):>12} "
            f"{(f'{vllm_x:.2f}x' if vllm_x else '-'):>6} "
            f"{(f'{vs_e:.2f}x' if vs_e else '-'):>6} "
            f"{(f'{vs_g:.2f}x' if vs_g else '-'):>6}"
        )
    print("\nvsE / vsG = vLLM / lab under eager and under graphs.")
    print("lab× / vllm× = tok/s(graphs) / tok/s(eager) for that engine.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eager", required=True, help="Sweep JSON with enforce_eager=True")
    p.add_argument("--graphs", required=True, help="Sweep JSON with enforce_eager=False")
    p.add_argument("--json-out", required=True)
    args = p.parse_args()

    with open(os.path.expanduser(args.eager), encoding="utf-8") as f:
        eager = json.load(f)
    with open(os.path.expanduser(args.graphs), encoding="utf-8") as f:
        graphs = json.load(f)

    if eager.get("enforce_eager") is False:
        print("warning: --eager file has enforce_eager=False", flush=True)
    if graphs.get("enforce_eager") is True:
        print("warning: --graphs file has enforce_eager=True", flush=True)

    merged = merge(eager, graphs)
    print_table(merged)

    out = os.path.expanduser(args.json_out)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
