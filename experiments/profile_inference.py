#!/usr/bin/env python3
"""Profile nano-vLLM inference with torch.profiler and/or Nsight Systems (nsys).

This script drives the engine at the add_request()/step() level (same
pattern as bench_cuda_graph_decode.py) so prefill and decode steps can be
timed separately, and wraps NVTX ranges around each step. ModelRunner.run()
itself pushes finer-grained NVTX ranges ("prepare_input", "run_model[eager
or graph]", "sample") so both torch.profiler and nsys timelines break down
into the same phases.

Two workload modes:
  --workload burst    (default) Every request is queued up front, like
                       bench_cuda_graph_decode.py -- an offline batch job.
  --workload poisson   Requests arrive over time as a Poisson process at
                       --request-rate req/s (same convention as vLLM's
                       benchmark_serving.py and this repo's
                       experiments/bench_step_modes.py), so new prefills can
                       land while other sequences are mid-decode. Use this
                       to see head-of-line blocking: with the default
                       Config.step_mode="separate", the scheduler always
                       prefers prefill over decode (Scheduler.schedule()),
                       so an arriving request's prefill step runs to
                       completion before any queued decode work resumes --
                       every in-flight sequence's inter-token latency spikes
                       by roughly that prefill step's duration. Pass
                       --step-mode combined or unified (see
                       Scheduler._schedule_prefill_chunked /
                       _schedule_unified) to interleave them instead, and
                       compare the per-step latency spikes this script
                       prints at the end.

                       Caveats found while building this:
                       (1) --step-mode combined used to double-process a
                       request that finished its whole prefill within one
                       step: LLMEngine._step_combined called
                       Scheduler._schedule_decode() (which also runs
                       whenever such a request just joined self.running
                       before the `if self.scheduler.running` check) before
                       computing `prefill_tokens = sum(s.num_scheduled_tokens
                       for s in prefill_seqs)`; since Sequence objects are
                       shared, _schedule_decode() overwrote that seq's
                       num_scheduled_tokens (e.g. 400 -> 1) before the sum
                       ran, corrupting both the step's returned token count
                       and (via postprocess()) that sequence's
                       num_cached_tokens bookkeeping, plus wasting an extra
                       decode-shaped forward pass. Fixed by reordering
                       _step_combined to schedule decode before prefill each
                       step (nanovllm/engine/llm_engine.py) -- confirmed via
                       the repro above and the full test suite (71 passed).
                       (2) --step-mode unified: every step is reported as
                       "prefill" (Scheduler._schedule_unified always returns
                       is_prefill=True), so this script's decode-step-gap
                       metric below is always empty for unified -- comparing
                       its tail latency needs per-sequence token-arrival
                       timestamps instead of the step-level prefill/decode
                       sign convention used here.

torch.profiler mode (self-contained, no external tools needed):
    python experiments/profile_inference.py --torch-profile \
        --num-seqs 8 --input-len 256 --output-len 128 --no-enforce-eager
    # writes a chrome trace to --trace-dir (open at chrome://tracing or
    # https://ui.perfetto.dev) and prints top ops by CUDA/CPU time.

    python experiments/profile_inference.py --torch-profile --workload poisson \
        --num-seqs 24 --request-rate 3.0 --step-mode separate --no-enforce-eager

nsys mode (needs the nsys CLI). Two copies are installed on this machine but
neither is on PATH:
    - /usr/lib/nsight-systems/bin/nsys (2021.3.3.2) -- crashes on this
      Ubuntu/glibc with "undefined symbol: __libc_dlclose" in its injection
      library. Don't use it here.
    - /home/*/nsight-compute-2025.2.0.11/host/target-linux-x64/nsys
      (2025.2.0.0, bundled with Nsight Compute) -- works. Use this one.

    NSYS=~/nsight-compute-2025.2.0.11/host/target-linux-x64/nsys
    $NSYS profile -t cuda,nvtx,osrt -o experiments/artifacts/nsys_report \
        --force-overwrite=true --stats=true \
        python experiments/profile_inference.py \
        --num-seqs 8 --input-len 256 --output-len 128 --no-enforce-eager

    To exclude model load/warmup from the capture (recommended -- weight
    loading and CUDA graph capture dominate wall time and are not the
    "steady state inference" you usually want in the trace), add
    --cuda-profiler-range to this script and bound nsys to the
    cudaProfilerStart/Stop calls it makes around the generation loop:
    $NSYS profile -t cuda,nvtx,osrt -c cudaProfilerApi --capture-range-end=stop \
        -o experiments/artifacts/nsys_gen_only --force-overwrite=true --stats=true \
        python experiments/profile_inference.py --cuda-profiler-range ...
    Open the resulting .nsys-rep in the Nsight Systems UI (nsys-ui) and look
    at the NVTX row to jump straight to prefill/decode/prepare_input/
    run_model/sample ranges.

    Known limitation on this box: under WSL2, nsys's system-level GPU kernel
    trace comes back empty ("does not contain CUDA kernel data") even though
    the CPU-side CUDA API trace (cudaMemcpyAsync/cudaLaunchKernel/etc, the
    'cuda_api_sum' table from --stats=true) and the NVTX range summary both
    work fine -- likely a CUPTI/driver limitation of the virtualized GPU.
    For actual per-kernel GPU time (flash-attn, GEMMs, elementwise, ...) use
    --torch-profile above instead; its in-process CUPTI capture is unaffected.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common_workload import default_model_path, make_poisson_serving_workload, make_workload  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=default_model_path())
    p.add_argument("--workload", choices=["burst", "poisson"], default="burst")
    p.add_argument("--num-seqs", type=int, default=8)
    p.add_argument("--input-len", type=int, default=256, help="[burst] fixed prompt length")
    p.add_argument("--output-len", type=int, default=128, help="[burst] decode length; ignore_eos so every run is identical")
    p.add_argument("--mean-input-len", type=int, default=150, help="[poisson] ShareGPT-ish mean prompt length")
    p.add_argument("--mean-output-len", type=int, default=100, help="[poisson] ShareGPT-ish mean decode length")
    p.add_argument("--request-rate", type=float, default=3.0, help="[poisson] requests/sec, Poisson arrival")
    p.add_argument("--step-mode", choices=["separate", "combined", "unified"], default="separate")
    p.add_argument("--chunk-prefill-tokens", type=int, default=-1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--torch-profile", action="store_true",
        help="wrap the generation loop in torch.profiler and dump a chrome trace + op tables",
    )
    p.add_argument(
        "--trace-dir", default=os.path.join("/tmp", "nanovllm_profile"),
        help="where to write the torch.profiler chrome trace (default: /tmp/nanovllm_profile)",
    )
    p.add_argument(
        "--cuda-profiler-range", action="store_true",
        help="bound the profiled region with cudaProfilerStart/Stop so `nsys profile -c cudaProfilerApi` "
             "only captures the generation loop, not model load/warmup/graph capture",
    )
    p.add_argument("--top-k", type=int, default=20, help="how many ops to print per table")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not os.path.isdir(args.model):
        raise SystemExit(f"Model directory not found: {args.model}")

    from nanovllm import LLM, SamplingParams

    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        step_mode=args.step_mode,
        chunk_prefill_tokens=args.chunk_prefill_tokens,
    )
    try:
        torch.cuda.nvtx.range_push("warmup")
        llm.generate(["hi"], SamplingParams(max_tokens=4), use_tqdm=False)

        if args.workload == "burst":
            # torch.compile/Triton autotune keys on tensor shape, so a bs=1
            # warmup alone leaves the actual profiled batch/seq-len shapes
            # uncompiled -- without this, the trace's top entries are
            # CachingAutotuner.benchmark_all_configs (compile-time
            # autotuning), not real inference work. Run the exact same
            # workload once, unprofiled, so every shape below is cached.
            prompts, _ = make_workload(args.num_seqs, args.input_len, args.input_len, args.output_len, args.output_len, args.seed)
            sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.output_len) for _ in prompts]
            llm.generate(
                list(prompts),
                [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=sp.max_tokens) for sp in sps],
                use_tqdm=False,
            )
        else:
            # Poisson mode has variable request lengths by construction, so
            # there's no single shape to pre-warm -- recompilation for new
            # shapes is an inherent (and worth-reporting) cost of serving
            # variable-length real traffic, not a benchmark artifact to hide.
            requests = make_poisson_serving_workload(args.num_seqs, args.mean_input_len, args.mean_output_len, args.request_rate, args.seed)
        torch.cuda.nvtx.range_pop()

        prof = None
        if args.torch_profile:
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            prof.__enter__()

        if args.cuda_profiler_range:
            torch.cuda.cudart().cudaProfilerStart()

        prefill_time = decode_time = 0.0
        prefill_tokens = decode_tokens = 0
        prefill_steps = decode_steps = 0
        decode_step_times: list[float] = []  # per-decode-step wall time, in order -- used to spot latency spikes

        if args.workload == "burst":
            for prompt, sp in zip(prompts, sps):
                llm.add_request(prompt, sp)
            while not llm.is_finished():
                t0 = time.perf_counter()
                torch.cuda.nvtx.range_push("engine.step")
                _, num_tokens = llm.step()
                torch.cuda.nvtx.range_pop()
                dt = time.perf_counter() - t0
                if num_tokens > 0:
                    prefill_time += dt
                    prefill_tokens += num_tokens
                    prefill_steps += 1
                else:
                    decode_time += dt
                    decode_tokens += -num_tokens
                    decode_steps += 1
                    decode_step_times.append(dt)
        else:
            # Same arrival-loop pattern as experiments/bench_step_modes.py:
            # add requests as their scheduled arrival time is reached, sleep
            # until the next arrival if nothing is in flight, else keep
            # stepping the engine.
            next_idx = 0
            t_wall0 = time.perf_counter()
            t_last_decode_end = None  # gap since the previous decode step *finished* -- this is what an
            # in-flight sequence's inter-token latency actually feels, since a step in "separate" mode is
            # either all-prefill or all-decode: an intervening prefill step shows up here as a gap, not in
            # the decode step's own (short) duration.
            while next_idx < len(requests) or not llm.is_finished():
                now = time.perf_counter() - t_wall0
                while next_idx < len(requests) and requests[next_idx][2] <= now:
                    prompt, max_tokens, _ = requests[next_idx]
                    sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_tokens)
                    llm.add_request(prompt, sp)
                    next_idx += 1
                if llm.is_finished():
                    if next_idx < len(requests):
                        sleep_s = requests[next_idx][2] - (time.perf_counter() - t_wall0)
                        if sleep_s > 0:
                            time.sleep(sleep_s)
                    continue
                t0 = time.perf_counter()
                torch.cuda.nvtx.range_push("engine.step")
                _, num_tokens = llm.step()
                torch.cuda.nvtx.range_pop()
                t1 = time.perf_counter()
                dt = t1 - t0
                if num_tokens > 0:
                    prefill_time += dt
                    prefill_tokens += num_tokens
                    prefill_steps += 1
                else:
                    decode_time += dt
                    decode_tokens += -num_tokens
                    decode_steps += 1
                    if t_last_decode_end is not None:
                        decode_step_times.append(t1 - t_last_decode_end)
                    t_last_decode_end = t1

        if args.cuda_profiler_range:
            torch.cuda.cudart().cudaProfilerStop()

        if prof is not None:
            prof.__exit__(None, None, None)
    finally:
        llm.exit()

    print(
        f"\nmodel={args.model} workload={args.workload} step_mode={args.step_mode} "
        f"enforce_eager={args.enforce_eager} num_seqs={args.num_seqs}",
        flush=True,
    )
    print(
        f"prefill: {prefill_tokens} tok in {prefill_steps} steps, "
        f"{prefill_time:.3f}s, {prefill_tokens / prefill_time if prefill_time else 0:.1f} tok/s",
        flush=True,
    )
    print(
        f"decode:  {decode_tokens} tok in {decode_steps} steps, "
        f"{decode_time:.3f}s, {decode_tokens / decode_time if decode_time else 0:.1f} tok/s "
        f"({1000 * decode_time / decode_steps if decode_steps else 0:.2f} ms/step avg)",
        flush=True,
    )

    if decode_step_times:
        metric = "inter-token gap (time since the previous decode step)" if args.workload == "poisson" else "decode step duration"
        med = statistics.median(decode_step_times)
        p99 = sorted(decode_step_times)[int(0.99 * (len(decode_step_times) - 1))]
        mx = max(decode_step_times)
        spikes = [dt for dt in decode_step_times if dt > 3 * med]
        print(
            f"{metric}: median={1000 * med:.2f}ms p99={1000 * p99:.2f}ms max={1000 * mx:.2f}ms "
            f"| {len(spikes)}/{len(decode_step_times)} steps >3x median "
            f"(likely a concurrent prefill blocking decode -- see step_mode)",
            flush=True,
        )

    if prof is not None:
        os.makedirs(args.trace_dir, exist_ok=True)
        trace_path = os.path.join(args.trace_dir, f"trace_{args.workload}_{args.step_mode}_eager{args.enforce_eager}.json")
        prof.export_chrome_trace(trace_path)
        print(f"\nchrome trace written to {trace_path} (open at chrome://tracing or https://ui.perfetto.dev)")

        print(f"\n--- top {args.top_k} ops by self CUDA time ---")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=args.top_k))

        print(f"\n--- top {args.top_k} ops by self CPU time (Python/launch overhead) ---")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=args.top_k))


if __name__ == "__main__":
    main()
