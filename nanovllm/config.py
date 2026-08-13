import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # Prefix caching (on by default): hash full KV blocks and reuse them
    # across requests sharing an identical token prefix. Disable to measure
    # its impact (every prefill recomputes from scratch; block reuse via
    # ref_count sharing never happens) -- see BlockManager.
    enable_prefix_caching: bool = True
    chunk_prefill_tokens: int = -1
    # CPU KV-cache swap: on preemption, copy a sequence's private KV blocks
    # to a CPU pool instead of discarding them, so re-admission skips the
    # recompute forward pass (see Scheduler.preempt/_should_swap). Verified
    # correct (nanovllm/engine/cpu_block_manager.py, tests/unit/test_kv_swap.py,
    # 593 real swaps on Qwen3-0.6B/RTX 3060 with matching output vs. no
    # preemption) -- but measured *6% SLOWER* wall-clock than plain recompute
    # at the same load (experiments/bench_preemption_cost.py --compare-swap):
    # it cuts wasted recompute tokens by ~79%, but the synchronous D2H/H2D
    # copy + torch.cuda.synchronize() overhead (see ModelRunner.swap_out/in)
    # outweighs that for a model this small, where recompute is already
    # cheap (~10k+ tok/s prefill). Off by default; may pay off for larger
    # models / longer contexts where recompute FLOPs dominate, unverified.
    kv_swap_enabled: bool = False
    kv_swap_min_tokens: int = 0
    cpu_kvcache_gib: float = 1.0
    num_cpu_kvcache_blocks: int = -1
    # Experimental: how prefill and decode share an engine step.
    #   "separate" (default) -- current behavior: a step is either all
    #     prefill (flash_attn_varlen_func) or all decode
    #     (flash_attn_with_kvcache / CUDA graph), never both.
    #   "combined" -- both admitted every step, but as two separate kernel
    #     calls (prefill via varlen, decode via with_kvcache/CUDA graph).
    #   "unified" -- both admitted into ONE batch through ONE varlen call
    #     (decode is just seqlen_q=1 prefill against cached context). Always
    #     eager -- CUDA graph never applies to this mode.
    # See Scheduler._schedule_unified, LLMEngine._step_combined/_step_unified.
    step_mode: str = "separate"

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.chunk_prefill_tokens == -1 or 0 < self.chunk_prefill_tokens <= self.max_num_batched_tokens
        assert not self.kv_swap_enabled or self.cpu_kvcache_gib > 0
        assert self.kv_swap_min_tokens >= 0
        assert self.step_mode in ("separate", "combined", "unified")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
