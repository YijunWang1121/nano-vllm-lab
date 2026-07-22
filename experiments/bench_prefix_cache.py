#!/usr/bin/env python3
"""Prefix-cache stress benchmarks with cache hit-rate reporting.

Two modes designed so the paged-KV prefix cache actually matters:

  shared_prefix  N requests share one long prompt prefix (like a big system
                 prompt / few-shot header) and differ only in a short suffix.
                 Runs a cold batch, then a warm batch with NEW suffixes but the
                 same prefix -> warm prefill should hit cached blocks.

  multiturn      One real conversation. Each turn re-sends the full history
                 through the chat template (stateless-server style). With the
                 prefix cache, turn N only prefills the new tail; a stateless
                 HF baseline (--hf-baseline) recomputes the whole history.

Note: cache hits are block-granular (256 tokens) and the last partial block is
never shared, so the shared prefix should be >= 512 tokens to see hits.

Examples:

    python experiments/bench_prefix_cache.py --mode shared_prefix
    python experiments/bench_prefix_cache.py --mode multiturn --turns 6
    python experiments/bench_prefix_cache.py --mode multiturn --hf-baseline
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from random import randint, seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common_workload import default_model_path  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=default_model_path())
    p.add_argument("--mode", choices=("shared_prefix", "multiturn"), default="shared_prefix")
    p.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=4096)
    # shared_prefix knobs
    p.add_argument("--num-seqs", type=int, default=8)
    p.add_argument("--prefix-len", type=int, default=1792, help="Shared prefix tokens (multiple of 256 recommended)")
    p.add_argument("--suffix-len", type=int, default=64, help="Unique per-request suffix tokens")
    p.add_argument("--max-tokens", type=int, default=64, help="Output tokens per request")
    # multiturn knobs
    p.add_argument("--turns", type=int, default=5)
    p.add_argument("--system-len", type=int, default=1024, help="Approx tokens of synthetic system prompt")
    p.add_argument("--hf-baseline", action="store_true", help="Also run stateless HF generate for the same conversation")
    p.add_argument("--hf-only", action="store_true", help="Run only the stateless HF baseline (no nano engine)")
    p.add_argument("--seed", type=int, default=0)
    return p


def cache_stats(llm) -> dict:
    return llm.scheduler.block_manager.prefix_cache_stats()


def reset_cache_stats(llm):
    llm.scheduler.block_manager.reset_prefix_cache_stats()


def fmt_stats(s: dict) -> str:
    return (
        f"prompt_tokens={s['prompt_tokens']} cached_tokens={s['cached_tokens']} "
        f"hit_rate={s['hit_rate']:.1%}"
    )


# ---------------------------------------------------------------- shared_prefix

def run_shared_prefix(llm, args) -> None:
    from nanovllm import SamplingParams

    seed(args.seed)
    prefix = [randint(0, 10000) for _ in range(args.prefix_len)]

    def make_batch():
        return [prefix + [randint(0, 10000) for _ in range(args.suffix_len)] for _ in range(args.num_seqs)]

    sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.max_tokens)
    total_out = args.num_seqs * args.max_tokens

    print(
        f"mode=shared_prefix num_seqs={args.num_seqs} prefix={args.prefix_len} "
        f"suffix={args.suffix_len} out={args.max_tokens}"
    )

    reset_cache_stats(llm)
    t0 = time.perf_counter()
    llm.generate(make_batch(), sp, use_tqdm=False)
    cold = time.perf_counter() - t0
    cold_stats = cache_stats(llm)
    print(f"[cold] time={cold:.3f}s throughput={total_out / cold:.1f}tok/s  {fmt_stats(cold_stats)}")

    reset_cache_stats(llm)
    t0 = time.perf_counter()
    llm.generate(make_batch(), sp, use_tqdm=False)  # new suffixes, same prefix
    warm = time.perf_counter() - t0
    warm_stats = cache_stats(llm)
    print(f"[warm] time={warm:.3f}s throughput={total_out / warm:.1f}tok/s  {fmt_stats(warm_stats)}")
    print(f"[warm/cold] speedup={cold / warm:.2f}x")


# ------------------------------------------------------------------- multiturn

SYSTEM_SENTENCE = (
    "You are a meticulous engineering assistant for an LLM serving codebase; "
    "always reason about scheduling, paged KV cache blocks, prefix reuse, CUDA "
    "graphs, and tensor parallel layouts before you answer. "
)

USER_TURNS = [
    "What is a paged KV cache and why do serving engines use it?",
    "How does prefix caching interact with those KV blocks?",
    "When does a multi-turn conversation benefit from that cache?",
    "What metadata does the attention kernel need for a cached prefill?",
    "Why do CUDA graphs mostly help the decode phase?",
    "How would tensor parallelism change the KV block layout?",
    "Summarize everything we discussed in three bullet points.",
]


def build_system_prompt(tokenizer, target_tokens: int) -> str:
    text = SYSTEM_SENTENCE
    while len(tokenizer.encode(text)) < target_tokens:
        text += SYSTEM_SENTENCE
    return text


def run_multiturn(llm, tokenizer, args) -> list[dict]:
    from nanovllm import SamplingParams

    system_prompt = build_system_prompt(tokenizer, args.system_len)
    sp = SamplingParams(temperature=0.6, max_tokens=args.max_tokens)
    messages = [{"role": "system", "content": system_prompt}]
    rows = []

    print(f"mode=multiturn turns={args.turns} system~{args.system_len}tok out<={args.max_tokens}")

    for t in range(args.turns):
        messages.append({"role": "user", "content": USER_TURNS[t % len(USER_TURNS)]})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        n_prompt = len(tokenizer.encode(prompt))

        reset_cache_stats(llm)
        t0 = time.perf_counter()
        out = llm.generate([prompt], sp, use_tqdm=False)[0]
        wall = time.perf_counter() - t0
        stats = cache_stats(llm)
        messages.append({"role": "assistant", "content": out["text"]})

        rows.append(dict(turn=t + 1, prompt=n_prompt, wall=wall, **stats))
        print(
            f"[turn {t + 1}] prompt={n_prompt}tok wall={wall:.3f}s "
            f"cached={stats['cached_tokens']}tok hit_rate={stats['hit_rate']:.1%}"
        )

    return rows


def run_multiturn_hf(args) -> None:
    """Stateless HF baseline: re-prefill full history every turn."""
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Reclaim GPU memory left by the nano engine (module cycles need a gc pass
    # before the caching allocator can actually release the weights).
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True
        ).cuda()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True
        ).cuda()
    model.eval()

    system_prompt = build_system_prompt(tokenizer, args.system_len)
    messages = [{"role": "system", "content": system_prompt}]

    print(f"mode=multiturn engine=hf_stateless turns={args.turns}")
    with torch.inference_mode():
        for t in range(args.turns):
            messages.append({"role": "user", "content": USER_TURNS[t % len(USER_TURNS)]})
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(prompt, return_tensors="pt")
            input_ids = enc["input_ids"].cuda()
            attn = enc["attention_mask"].cuda()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0

            reply = tokenizer.decode(out[0][input_ids.size(1):], skip_special_tokens=True)
            messages.append({"role": "assistant", "content": reply})
            print(f"[turn {t + 1}] prompt={input_ids.size(1)}tok wall={wall:.3f}s (full re-prefill)")

    del model
    torch.cuda.empty_cache()


def main() -> None:
    args = build_parser().parse_args()
    if not os.path.isdir(args.model):
        raise SystemExit(f"Model directory not found: {args.model}")

    if args.hf_only:
        run_multiturn_hf(args)
        return

    from nanovllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
    )
    try:
        llm.generate(["warmup"], SamplingParams(max_tokens=4), use_tqdm=False)
        if args.mode == "shared_prefix":
            run_shared_prefix(llm, args)
        else:
            run_multiturn(llm, tokenizer, args)
    finally:
        llm.exit()

    if args.mode == "multiturn" and args.hf_baseline:
        run_multiturn_hf(args)


if __name__ == "__main__":
    main()
