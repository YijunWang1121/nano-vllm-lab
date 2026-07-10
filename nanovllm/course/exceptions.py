"""Pedagogical exceptions for unfinished course TODOs."""

from __future__ import annotations


class CourseNotImplementedError(NotImplementedError):
    """Raised when a student TODO has not been implemented yet."""

    def __init__(
        self,
        todo_id: str,
        *,
        subsystem: str,
        tutorial_path: str,
        milestone_test: str,
        hint: str | None = None,
    ):
        self.todo_id = todo_id
        self.subsystem = subsystem
        self.tutorial_path = tutorial_path
        self.milestone_test = milestone_test
        self.hint = hint
        parts = [
            f"{todo_id} is not implemented.",
            f"Subsystem: {subsystem}.",
            f"Read {tutorial_path}.",
            f"Verify with: {milestone_test}",
        ]
        if hint:
            parts.append(f"Hint: {hint}")
        super().__init__(" ".join(parts))
