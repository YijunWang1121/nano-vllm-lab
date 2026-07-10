"""Opt-in debug logging for nano-vLLM subsystems.

Enable with environment variables (disabled by default):

    NANOVLLM_DEBUG_ENGINE=1
    NANOVLLM_DEBUG_SCHEDULER=1
    NANOVLLM_DEBUG_KVCACHE=1
    NANOVLLM_DEBUG_RUNNER=1
    NANOVLLM_DEBUG_ATTENTION=1
    NANOVLLM_DEBUG_SAMPLING=1
"""

from __future__ import annotations

import os
from typing import Any


_FLAGS = {
    "engine": "NANOVLLM_DEBUG_ENGINE",
    "scheduler": "NANOVLLM_DEBUG_SCHEDULER",
    "kvcache": "NANOVLLM_DEBUG_KVCACHE",
    "runner": "NANOVLLM_DEBUG_RUNNER",
    "attention": "NANOVLLM_DEBUG_ATTENTION",
    "sampling": "NANOVLLM_DEBUG_SAMPLING",
}


def debug_enabled(subsystem: str) -> bool:
    env = _FLAGS.get(subsystem)
    if env is None:
        return False
    return os.environ.get(env, "").strip() not in ("", "0", "false", "False")


def debug_log(subsystem: str, message: str, **fields: Any) -> None:
    if not debug_enabled(subsystem):
        return
    if fields:
        extras = " ".join(f"{k}={v!r}" for k, v in fields.items())
        print(f"[nanovllm:{subsystem}] {message} | {extras}", flush=True)
    else:
        print(f"[nanovllm:{subsystem}] {message}", flush=True)
