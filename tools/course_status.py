#!/usr/bin/env python3
"""Print course progress: TODOs, milestones, and next recommended work.

Milestone completion is driven by:
  1. Entries in .course_progress.json (written when milestone tests pass), or
  2. Absence of TODO / CourseNotImplementedError scaffolding in source, or
  3. With --run-tests: live pytest results (pass => record + optional cleanup).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from course_lib import (  # noqa: E402
    cleanup_milestone,
    load_progress,
    load_yaml,
    mark_milestone_passed,
    milestone_passed,
    run_pytest,
    todo_implemented,
)


def git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Execute milestone tests; on pass, record progress and clean scaffolding",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="With --run-tests, record passes but do not delete TODO/raise scaffolding",
    )
    args = parser.parse_args()

    todos = load_yaml("todos.yaml")["todos"]
    milestones = load_yaml("milestones.yaml")["milestones"]
    progress = load_progress()
    branch = git_branch()

    print(f"Branch: {branch}")
    if branch == "course/reference-solutions":
        print("Note: this is the reference branch. Implement on course/student.")
    if progress.get("milestones_passed"):
        print(f"Progress file: passed M{', M'.join(str(n) for n in progress['milestones_passed'])}")
    print()

    print("Milestones:")
    todos_by_id = {t["id"]: t for t in todos}
    first_pending_ms = None

    for ms in milestones:
        cpu_only = ms.get("cpu_only", True)
        cpu = "CPU" if cpu_only else "GPU"
        ms_todos = [todos_by_id[tid] for tid in ms.get("todo_ids", []) if tid in todos_by_id]

        if args.run_tests:
            ok, out = run_pytest(ms["tests"], cpu_only=cpu_only)
            if ok:
                if not args.no_cleanup:
                    print(f"  cleaning M{ms['number']:02d} scaffolding...")
                    cleaned = cleanup_milestone(ms, todos_by_id)
                else:
                    cleaned = list(ms.get("todo_ids", []))
                mark_milestone_passed(ms["number"], cleaned)
                progress = load_progress()
                status = "done"
            else:
                status = "pending"
            print(f"  [{status:7}] M{ms['number']:02d} ({cpu}) {ms['title']}")
            print(f"           tests: {'PASS' if ok else 'FAIL'}")
            if not ok:
                for line in out.splitlines():
                    if "FAILED" in line or "CourseNotImplementedError" in line or "TODO-L" in line:
                        print(f"           {line}")
        else:
            # Intuition: a milestone is done only after its tests have passed
            # (recorded in .course_progress.json by run_milestone / --run-tests).
            ms_done = milestone_passed(ms["number"], progress)
            status = "done" if ms_done else "pending"
            print(f"  [{status:7}] M{ms['number']:02d} ({cpu}) {ms['title']}")

        if status == "pending" and first_pending_ms is None:
            first_pending_ms = ms

    progress = load_progress()
    done = [t for t in todos if todo_implemented(t, progress)]
    remaining = [t for t in todos if not todo_implemented(t, progress)]
    print()
    print(f"TODOs completed: {len(done)}/{len(todos)}")
    print(f"TODOs remaining: {len(remaining)}")
    print()

    if first_pending_ms is None and not remaining:
        print("All registered milestones/TODOs look complete. Run full suite:")
        print('  pytest -m "not gpu"')
        print("  pytest -m gpu   # when CUDA is available")
        return

    if remaining:
        next_todo = remaining[0]
        print("Next recommended TODO:")
        print(f"  {next_todo['id']}: {next_todo['title']}")
        print(f"  File: {next_todo['source_file']} :: {next_todo['symbol']}")
        print(f"  Docs: {next_todo['documentation']}")
        print(f"  Tests: {' '.join(next_todo['tests'])}")
    elif first_pending_ms is not None:
        print("Next recommended milestone:")
        print(f"  M{first_pending_ms['number']:02d}: {first_pending_ms['title']}")
        print(f"  Run: python tools/run_milestone.py {first_pending_ms['number']}")
    print()
    print("Tip: after implementing, run:")
    print(f"  python tools/run_milestone.py {first_pending_ms['number'] if first_pending_ms else 1}")
    print("  (passing tests auto-updates progress and deletes TODO/raise scaffolding)")


if __name__ == "__main__":
    main()
