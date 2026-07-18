#!/usr/bin/env python3
"""Run a single course milestone: show objective, TODOs, docs, and tests.

When tests pass, records the milestone as done and removes TODO / raise scaffolding
from the milestone source files (disable with --no-cleanup).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from course_lib import (  # noqa: E402
    cleanup_milestone,
    format_milestone_meta,
    load_yaml,
    mark_milestone_passed,
)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int, help="Milestone number, e.g. 4")
    parser.add_argument("--no-test", action="store_true")
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Do not delete TODO comments / dead raises after a passing run",
    )
    args = parser.parse_args()

    milestones = {m["number"]: m for m in load_yaml("milestones.yaml")["milestones"]}
    todos = {t["id"]: t for t in load_yaml("todos.yaml")["todos"]}
    if args.number not in milestones:
        print(f"Unknown milestone {args.number}. Valid: {sorted(milestones)}")
        sys.exit(2)
    ms = milestones[args.number]

    def emit(*args, **kwargs):
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)

    emit(f"=== Milestone {ms['number']}: {ms['title']} ===")
    emit(ms["objective"])
    emit(f"Effort: {format_milestone_meta(ms)}")
    emit()
    emit("Source files:")
    for f in ms["source_files"]:
        emit(f"  - {f}")
    emit()
    emit("TODO IDs:")
    for tid in ms.get("todo_ids", []):
        t = todos.get(tid, {})
        emit(f"  - {tid}: {t.get('title', '')}")
    emit()
    emit("Documentation:")
    for d in ms.get("docs", []):
        emit(f"  - {d}")
    emit()
    emit("Tests:")
    for t in ms["tests"]:
        emit(f"  - {t}")
    emit()

    if args.no_test:
        return

    marker = ["-m", "not gpu"] if ms.get("cpu_only", True) else ["-m", "gpu"]
    cmd = [sys.executable, "-m", "pytest", "-vv", *marker, *ms["tests"]]
    emit("Running:", " ".join(cmd))
    emit()
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        print()
        print("Failure categories:")
        print("  - CourseNotImplementedError: implement the listed TODO")
        print("  - AssertionError: logic bug; re-read the tutorial invariants")
        print("  - ImportError/CUDA: use CPU milestones or install GPU deps")
        print(f"  - Docs: {', '.join(ms.get('docs', []))}")
        sys.exit(proc.returncode)

    print()
    print(f"Milestone {ms['number']} PASSED.")
    cleaned_ids: list[str] = []
    if not args.no_cleanup:
        print("Updating progress and cleaning TODO scaffolding:")
        cleaned_ids = cleanup_milestone(ms, todos)
    else:
        print("Recording progress (--no-cleanup: left TODO scaffolding in place).")
        cleaned_ids = list(ms.get("todo_ids", []))

    mark_milestone_passed(ms["number"], cleaned_ids)
    print(f"Recorded in .course_progress.json")
    print("Next: python tools/course_status.py")
    sys.exit(0)


if __name__ == "__main__":
    main()
