import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.exceptions import EngineOOMError
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.utils.debug import debug_log


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.step_mode = config.step_mode
        self._exited = False
        atexit.register(self.exit)

    def exit(self):
        if getattr(self, "_exited", False):
            return
        self._exited = True
        if hasattr(self, "model_runner"):
            self.model_runner.call("exit")
            del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        debug_log("engine", "add_request", seq_id=seq.seq_id, prompt_len=len(prompt))
        self.scheduler.add(seq)

    def step(self):
        if self.step_mode == "combined":
            return self._step_combined()
        if self.step_mode == "unified":
            return self._step_unified()
        return self._step_separate()

    def _drain_swaps(self):
        # Execute any KV-cache swap copies the scheduler just decided on
        # (preempt() -> swap-out, _schedule_swap_ins() -> swap-in) before the
        # next model forward pass, so freshly-restored/evicted blocks are
        # never read half-copied. No-ops when kv_swap_enabled=False.
        for gpu_block_ids, cpu_block_ids in self.scheduler.drain_pending_swap_outs():
            self.model_runner.call("swap_out", gpu_block_ids, cpu_block_ids)
        for cpu_block_ids, gpu_block_ids in self.scheduler.drain_pending_swap_ins():
            self.model_runner.call("swap_in", cpu_block_ids, gpu_block_ids)

    def _run_batch(self, scheduler_outputs: list, seqs: list, is_prefill: bool) -> None:
        # Shared by all three step methods: run a batch, and on a real CUDA
        # OOM abort just this batch (freeing its blocks, marking it FINISHED
        # with aborted=True) instead of letting the exception take down the
        # whole engine. Appends (seq_id, completion_token_ids, aborted)
        # tuples onto `scheduler_outputs` in place.
        try:
            token_ids = self.model_runner.call("run", seqs, is_prefill)
        except EngineOOMError:
            self.scheduler.abort_batch(seqs, reason="forward_oom")
            scheduler_outputs += [(s.seq_id, s.completion_token_ids, True) for s in seqs]
            return
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        scheduler_outputs += [(s.seq_id, s.completion_token_ids, False) for s in seqs if s.is_finished]

    def _step_separate(self):
        # Config.step_mode == "separate" (default): a step is either all
        # prefill or all decode, never both -- see Scheduler.schedule().
        seqs, is_prefill = self.scheduler.schedule()
        self._drain_swaps()
        aborted = self.scheduler.drain_pending_aborts()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        debug_log(
            "engine",
            "step",
            is_prefill=is_prefill,
            seq_ids=[s.seq_id for s in seqs],
            num_tokens=num_tokens,
        )
        outputs = [(s.seq_id, s.completion_token_ids, True) for s in aborted]
        # seqs can legitimately be empty here (e.g. the only running
        # sequence was just aborted for lack of KV-cache room) -- calling
        # model_runner.call("run", [], ...) would crash downstream in
        # build_block_tables's unconditional max() over an empty list.
        if seqs:
            self._run_batch(outputs, seqs, is_prefill)
        return outputs, num_tokens

    def _step_combined(self):
        # Config.step_mode == "combined": both prefill and decode admitted
        # every step, as two separate kernel calls (prefill stays on the
        # varlen path, decode keeps using with_kvcache/CUDA graph unchanged).
        self.scheduler._schedule_swap_ins()
        # Decode is scheduled *before* prefill deliberately: _schedule_decode()
        # reads self.scheduler.running, and _schedule_prefill_classic/chunked
        # appends a sequence straight into self.running the moment it finishes
        # its whole prompt. Scheduling prefill first used to let such a
        # sequence get immediately re-selected by _schedule_decode() in that
        # same step -- wasting a forward pass (the prefill sample was thrown
        # away) and corrupting that sequence's num_cached_tokens bookkeeping,
        # since Sequence objects are shared and _schedule_decode() overwrites
        # num_scheduled_tokens (full prompt length -> 1) before postprocess()
        # for the prefill call ever reads it. Decode-first means self.running
        # only reflects sequences that were already running *before* this
        # step, so a freshly admitted sequence's first step is prefill-only.
        decode_seqs: list = []
        if self.scheduler.running:
            decode_seqs, _ = self.scheduler._schedule_decode()
        prefill_seqs, _ = (
            self.scheduler._schedule_prefill_chunked() if self.scheduler.chunk_prefill_tokens > 0
            else self.scheduler._schedule_prefill_classic()
        )
        self._drain_swaps()
        aborted = self.scheduler.drain_pending_aborts()

        outputs = [(s.seq_id, s.completion_token_ids, True) for s in aborted]
        prefill_tokens = 0
        if prefill_seqs:
            prefill_tokens = sum(s.num_scheduled_tokens for s in prefill_seqs)
            self._run_batch(outputs, prefill_seqs, True)
        if decode_seqs:
            self._run_batch(outputs, decode_seqs, False)
        # Both kinds of work can happen in the same step here, so the usual
        # single-signed-int "prefill xor decode" convention can't fully
        # represent it -- report prefill tokens when any occurred (matches
        # existing callers' ">0 means prefill" check), else decode count.
        return outputs, prefill_tokens if prefill_seqs else -len(decode_seqs)

    def _step_unified(self):
        # Config.step_mode == "unified": both admitted into ONE batch driven
        # by a single flash_attn_varlen_func call -- see
        # Scheduler._schedule_unified. Always eager (never CUDA graph).
        seqs = self.scheduler._schedule_unified()
        self._drain_swaps()
        # Drain aborts before the early return below -- otherwise a
        # sequence aborted inside _schedule_unified's decode loop with no
        # remaining prefill work this step would never be reported.
        aborted = self.scheduler.drain_pending_aborts()
        outputs = [(s.seq_id, s.completion_token_ids, True) for s in aborted]
        if not seqs:
            return outputs, 0
        num_tokens = sum(s.num_scheduled_tokens for s in seqs)
        self._run_batch(outputs, seqs, True)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids, aborted in output:
                outputs[seq_id] = (token_ids, aborted)
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids, "aborted": aborted}
            for token_ids, aborted in outputs
        ]
        return outputs
