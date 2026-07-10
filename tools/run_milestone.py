#!/usr/bin/env python3
"""Run a single course milestone: show objective, TODOs, docs, and tests."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML is required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(name: str):
    with open(ROOT / "course" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int, help="Milestone number, e.g. 4")
    parser.add_argument("--no-test", action="store_true")
    args = parser.parse_args()

    milestones = {m["number"]: m for m in load_yaml("milestones.yaml")["milestones"]}
    todos = {t["id"]: t for t in load_yaml("todos.yaml")["todos"]}
    if args.number not in milestones:
        print(f"Unknown milestone {args.number}. Valid: {sorted(milestones)}")
        sys.exit(2)
    ms = milestones[args.number]

    print(f"=== Milestone {ms['number']}: {ms['title']} ===")
    print(ms["objective"])
    print()
    print("Source files:")
    for f in ms["source_files"]:
        print(f"  - {f}")
    print()
    print("TODO IDs:")
    for tid in ms.get("todo_ids", []):
        t = todos.get(tid, {})
        print(f"  - {tid}: {t.get('title', '')}")
    print()
    print("Documentation:")
    for d in ms.get("docs", []):
        print(f"  - {d}")
    print()
    print("Tests:")
    for t in ms["tests"]:
        print(f"  - {t}")
    print()

    if args.no_test:
        return

    marker = [] if not ms.get("cpu_only", True) else ["-m", "not gpu"]
    # GPU milestones still run under pytest; skips happen inside tests.
    if not ms.get("cpu_only", True):
        marker = ["-m", "gpu"]
    cmd = [sys.executable, "-m", "pytest", "-vv", *marker, *ms["tests"]]
    print("Running:", " ".join(cmd))
    print()
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        print()
        print("Failure categories:")
        print("  - CourseNotImplementedError: implement the listed TODO")
        print("  - AssertionError: logic bug; re-read the tutorial invariants")
        print("  - ImportError/CUDA: use CPU milestones or install GPU deps")
        print(f"  - Docs: {', '.join(ms.get('docs', []))}")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
