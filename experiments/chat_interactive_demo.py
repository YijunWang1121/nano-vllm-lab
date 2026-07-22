#!/usr/bin/env python3
"""Interactive multi-turn chat: each assistant reply is fed into the next prompt.

Unlike ``WORKLOAD=chat`` in run_compare.sh (pre-written fake assistant history for
throughput), this script actually generates each turn and appends the model output
before the next user message.

Example (RunPod):

    source /workspace/venv-nanovllm/bin/activate
    export NANOVLLM_TEST_MODEL=/workspace/huggingface/Qwen3-0.6B
    python experiments/chat_interactive_demo.py
    python experiments/chat_interactive_demo.py --turns 4 --max-tokens 128
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common_workload import default_model_path  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=default_model_path())
    p.add_argument(
        "--user-turns",
        nargs="+",
        default=[
            "What is paged KV cache in LLM serving?",
            "Why does that help continuous batching?",
            "Can you give a concrete example?",
        ],
        help="User messages, one per round (assistant replies come from the model).",
    )
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    return p


def run_interactive_chat(
    llm,
    tokenizer,
    user_turns: list[str],
    *,
    max_tokens: int,
    temperature: float,
) -> list[dict[str, str]]:
    """One conversation: user -> model -> append -> user -> model -> ..."""
    from nanovllm import SamplingParams

    messages: list[dict[str, str]] = []
    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    for turn_idx, user_text in enumerate(user_turns, start=1):
        messages.append({"role": "user", "content": user_text})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        print(f"\n{'=' * 60}")
        print(f"Turn {turn_idx} — user")
        print(user_text)
        print(f"\n--- prompt sent to model ({len(prompt)} chars) ---")
        print(prompt[:500] + ("..." if len(prompt) > 500 else ""))

        out = llm.generate([prompt], sp, use_tqdm=False)[0]
        assistant_text = out["text"]
        messages.append({"role": "assistant", "content": assistant_text})

        print(f"\nTurn {turn_idx} — assistant")
        print(assistant_text)

    return messages


def main() -> None:
    args = build_parser().parse_args()
    if not os.path.isdir(args.model):
        raise SystemExit(f"Model directory not found: {args.model}")

    from nanovllm import LLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(args.model, enforce_eager=args.enforce_eager, tensor_parallel_size=1)
    try:
        print(f"model={args.model}")
        print(f"rounds={len(args.user_turns)} max_tokens={args.max_tokens}")
        messages = run_interactive_chat(
            llm,
            tokenizer,
            args.user_turns,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"\n{'=' * 60}")
        print("Final message history (what you would keep in a chat session):")
        for msg in messages:
            role = msg["role"]
            preview = msg["content"][:120].replace("\n", " ")
            print(f"  [{role}] {preview}{'...' if len(msg['content']) > 120 else ''}")
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
