#!/usr/bin/env python3
"""Print course progress: TODOs, milestones, and next recommended work."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML is required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]


def git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def load_yaml(name: str):
    with open(ROOT / "course" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def todo_implemented(todo: dict) -> bool:
    """A TODO is complete if its marker comment is absent from the source file.

    Student branch keeps `# TODO-L...` markers; reference solutions remove them.
    """
    path = ROOT / todo["source_file"]
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    # Marker present => not done. Also treat CourseNotImplementedError raise for this id as not done.
    if re.search(rf"TODO-{re.escape(todo['id'].removeprefix('TODO-'))}\b|{re.escape(todo['id'])}", text):
        # Distinguish registry mention in comments vs active TODO block / raise.
        if f"raise CourseNotImplementedError(\"{todo['id']}\"" in text or f"raise CourseNotImplementedError('{todo['id']}'" in text:
            return False
        if f"# {todo['id']}" in text or f"# TODO-{todo['id'].removeprefix('TODO-')}" in text:
            return False
        # id may appear only in docs strings on reference — treat as done if no raise and no TODO header
    if f"# {todo['id']}:" in text or f"# {todo['id']} " in text:
        return False
    if "CourseNotImplementedError" in text and todo["id"] in text:
        return False
    return True


def run_pytest(paths: list[str]) -> tuple[bool, str]:
    cmd = [sys.executable, "-m", "pytest", "-q", "-m", "not gpu", *paths]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tests", action="store_true", help="Execute milestone tests (CPU)")
    args = parser.parse_args()

    todos = load_yaml("todos.yaml")["todos"]
    milestones = load_yaml("milestones.yaml")["milestones"]
    branch = git_branch()

    print(f"Branch: {branch}")
    if branch == "course/reference-solutions":
        print("Note: this is the reference branch. Implement on course/student.")
    print()

    done = [t for t in todos if todo_implemented(t)]
    remaining = [t for t in todos if not todo_implemented(t)]
    print(f"TODOs completed: {len(done)}/{len(todos)}")
    print(f"TODOs remaining: {len(remaining)}")
    print()

    print("Milestones:")
    next_todo = remaining[0] if remaining else None
    for ms in milestones:
        ms_todos = [t for t in todos if t["id"] in ms.get("todo_ids", [])]
        ms_done = all(todo_implemented(t) for t in ms_todos) if ms_todos else True
        status = "done" if ms_done else "pending"
        cpu = "CPU" if ms.get("cpu_only", True) else "GPU"
        print(f"  [{status:7}] M{ms['number']:02d} ({cpu}) {ms['title']}")
        if args.run_tests:
            ok, out = run_pytest(ms["tests"])
            print(f"           tests: {'PASS' if ok else 'FAIL'}")
            if not ok:
                # compact failure summary
                for line in out.splitlines():
                    if "FAILED" in line or "CourseNotImplementedError" in line or "TODO-L" in line:
                        print(f"           {line}")
    print()
    if next_todo:
        print("Next recommended TODO:")
        print(f"  {next_todo['id']}: {next_todo['title']}")
        print(f"  File: {next_todo['source_file']} :: {next_todo['symbol']}")
        print(f"  Docs: {next_todo['documentation']}")
        print(f"  Tests: {' '.join(next_todo['tests'])}")
    else:
        print("All registered TODOs look complete. Run full suite:")
        print('  pytest -m "not gpu"')
        print("  pytest -m gpu   # when CUDA is available")


if __name__ == "__main__":
    main()
