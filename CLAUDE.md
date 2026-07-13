# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

nano-vllm is a lightweight vLLM-style LLM inference engine built from scratch (single package `nanovllm/`). It supports offline batch inference with prefix caching, tensor parallelism, `torch.compile`, and CUDA graphs. Only the Qwen3 architecture is implemented (`nanovllm/models/qwen3.py`).

`origin` is a personal fork (`YijunWang1121/nano-vllm`); `upstream` is `GeeeekExplorer/nano-vllm`. This is a personal working copy, not an active upstream-contribution workflow.

## No tests, no CI, no lint on this branch

This branch (`read`) has no test suite, CI config, or lint/formatter config — `tests/` and `tools/` only contain leftover bytecode from other branches (a separate teaching-course fork has real tests under `tests/milestones/`, but that code isn't present here). There is also no GPU in this environment. Since changes can't be executed or auto-verified here, review code changes carefully by reading rather than running `example.py` / `bench.py`, and say so explicitly rather than claiming a change was tested.

## Install gotcha

`flash-attn` (a hard dependency) requires a working CUDA/nvcc toolchain to build — this is the most common install failure, not just a `pip install` issue.

## Code style

- Dataclasses use `@dataclass(slots=True)` deliberately (e.g. `Config`) — keep this when adding new dataclasses.
- The codebase is intentionally terse and dense: minimal docstrings/comments, short variable names (`q`, `k`, `v`, `h`), multi-statement one-liners. Match this style rather than expanding it with verbose comments or defensive code.
- Type hints are used inconsistently — don't treat their absence as a bug to fix incidentally.

## Architecture notes

- Attention delegates to `flash_attn_varlen_func` / `flash_attn_with_kvcache`; there's no hand-rolled attention kernel, but `layers/attention.py` has a custom Triton kernel (`store_kvcache_kernel`) for writing into the paged KV cache.
- Prefix caching uses block-hashing (`xxhash`) in `engine/block_manager.py`; KV cache block size must be a multiple of 256 (`Sequence.block_size = 256`).
- Tensor parallelism is hand-rolled in `engine/model_runner.py`: rank 0 drives other ranks via `multiprocessing.shared_memory.SharedMemory` + `Event`-based RPC (pickled method calls), not `torch.multiprocessing.spawn`. `Sequence` implements custom `__getstate__`/`__setstate__` for cheap pickling across this IPC boundary. The NCCL rendezvous address is hardcoded to `tcp://localhost:2333`.
