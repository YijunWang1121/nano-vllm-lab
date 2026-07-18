"""Shared helpers for course progress tracking and TODO scaffolding cleanup."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
PROGRESS_PATH = ROOT / ".course_progress.json"

DIFFICULTY_ORDER = ("easy", "medium", "hard", "very_hard")


def format_milestone_meta(ms: dict) -> str:
    """Short difficulty + time tag for status / run_milestone banners."""
    difficulty = ms.get("difficulty", "unknown")
    hours = ms.get("estimated_hours")
    if hours is None:
        return f"difficulty={difficulty}"
    hours_s = f"{hours:g}" if isinstance(hours, (int, float)) else str(hours)
    return f"difficulty={difficulty}, ~{hours_s}h"


def load_yaml(name: str):
    if yaml is None:
        print("PyYAML is required: pip install pyyaml", file=sys.stderr)
        sys.exit(1)
    with open(ROOT / "course" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_progress() -> dict:
    if not PROGRESS_PATH.exists():
        return {"milestones_passed": [], "cleaned_todos": []}
    try:
        data = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"milestones_passed": [], "cleaned_todos": []}
    data.setdefault("milestones_passed", [])
    data.setdefault("cleaned_todos", [])
    return data


def save_progress(data: dict) -> None:
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    PROGRESS_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def mark_milestone_passed(number: int, todo_ids: list[str] | None = None) -> dict:
    data = load_progress()
    passed = set(data["milestones_passed"])
    passed.add(number)
    data["milestones_passed"] = sorted(passed)
    if todo_ids:
        cleaned = set(data["cleaned_todos"])
        cleaned.update(todo_ids)
        data["cleaned_todos"] = sorted(cleaned)
    save_progress(data)
    return data


def milestone_passed(number: int, progress: dict | None = None) -> bool:
    progress = progress if progress is not None else load_progress()
    return number in progress.get("milestones_passed", [])


def todo_marker_present(text: str, todo_id: str) -> bool:
    short = todo_id.removeprefix("TODO-")
    if re.search(rf"#\s*(?:TODO-{re.escape(short)}|{re.escape(todo_id)})\b", text):
        return True
    if f'raise CourseNotImplementedError("{todo_id}"' in text:
        return True
    if f"raise CourseNotImplementedError('{todo_id}'" in text:
        return True
    if "CourseNotImplementedError" in text and todo_id in text:
        # Multi-line raise where id is on a following argument line.
        if re.search(
            rf"raise\s+CourseNotImplementedError\s*\((?:.|\n)*?{re.escape(todo_id)}",
            text,
        ):
            return True
    return False


def todo_implemented(todo: dict, progress: dict | None = None) -> bool:
    """Complete if milestone recorded, cleaned in progress, or scaffolding gone."""
    progress = progress if progress is not None else load_progress()
    todo_id = todo["id"]
    if todo_id in progress.get("cleaned_todos", []):
        return True
    if todo.get("milestone") in progress.get("milestones_passed", []):
        return True
    path = ROOT / todo["source_file"]
    if not path.exists():
        return False
    return not todo_marker_present(path.read_text(encoding="utf-8"), todo_id)


def run_pytest(paths: list[str], *, cpu_only: bool = True) -> tuple[bool, str]:
    marker = ["-m", "not gpu"] if cpu_only else ["-m", "gpu"]
    cmd = [sys.executable, "-m", "pytest", "-q", *marker, *paths]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out


def _remove_todo_comment_blocks(text: str, todo_id: str) -> str:
    short = todo_id.removeprefix("TODO-")
    # From `# TODO-...` / `# TODO-L... (continued):` through following `#` comment lines.
    pattern = re.compile(
        rf"^[ \t]*#\s*(?:TODO-{re.escape(short)}|{re.escape(todo_id)})\b.*\n"
        rf"(?:[ \t]*#.*\n)*",
        re.MULTILINE,
    )
    return pattern.sub("", text)


def _remove_raises_for_todo(text: str, todo_id: str) -> str:
    """Remove multi-line raise CourseNotImplementedError(...) blocks for todo_id.

    Closing paren is expected on its own line (course scaffold style), so we do
    not stop early on ')' characters inside hint strings.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    start_re = re.compile(r"^[ \t]*raise\s+CourseNotImplementedError\s*\(")
    end_re = re.compile(r"^[ \t]*\)\s*$")
    while i < len(lines):
        if start_re.match(lines[i]):
            j = i + 1
            while j < len(lines) and not end_re.match(lines[j]):
                j += 1
            if j < len(lines):
                j += 1  # include closing ')' line
            block = "".join(lines[i:j])
            if todo_id in block:
                i = j
                continue
            out.extend(lines[i:j])
            i = j
            continue
        out.append(lines[i])
        i += 1
    return "".join(out)


def _remove_unused_course_import(text: str) -> str:
    import_re = re.compile(
        r"^[ \t]*from nanovllm\.course\.exceptions import CourseNotImplementedError\n",
        re.MULTILINE,
    )
    without = import_re.sub("", text)
    if "CourseNotImplementedError" not in without:
        return without
    return text


def _collapse_blank_lines(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def cleanup_todo_scaffolding(todo: dict) -> bool:
    """Remove TODO comment blocks and CourseNotImplementedError raises for one TODO.

    Returns True if the source file changed.
    """
    path = ROOT / todo["source_file"]
    if not path.exists():
        return False
    original = path.read_text(encoding="utf-8")
    text = original
    todo_id = todo["id"]
    text = _remove_todo_comment_blocks(text, todo_id)
    text = _remove_raises_for_todo(text, todo_id)
    text = _remove_unused_course_import(text)
    text = _collapse_blank_lines(text)
    if text == original:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def cleanup_milestone(ms: dict, todos_by_id: dict[str, dict]) -> list[str]:
    """Clean scaffolding for all TODOs in a milestone. Returns cleaned todo ids."""
    cleaned: list[str] = []
    for tid in ms.get("todo_ids", []):
        todo = todos_by_id.get(tid)
        if not todo:
            continue
        changed = cleanup_todo_scaffolding(todo)
        # Record even if already clean, so status stays green after a pass.
        cleaned.append(tid)
        if changed:
            print(f"  cleaned scaffolding: {tid} ({todo['source_file']})")
        else:
            print(f"  already clean: {tid}")
    return cleaned
