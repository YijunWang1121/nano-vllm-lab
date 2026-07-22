# 15. Adding a Model Family: Llama

## Learning Objectives

- Walk through the exact steps used to port Llama (incl. Llama 3.1) onto this engine.
- Know which parts of a new model are architecture code vs. plumbing (registry, loader, RoPE).
- Estimate KV-cache size for any HF model: what you look up in `config.json` vs. what you compute by hand.
- Understand how the engine turns "free VRAM" into a concrete block count at startup.

## Why Exists

The engine originally hardcoded `Qwen3ForCausalLM`. Supporting a second family forces the
codebase to separate model-specific code (attention/MLP layout, RoPE variant, weight names)
from engine code (scheduler, paged KV cache, CUDA graphs) — the same split vLLM uses with
its model registry.

## The Porting Workflow (5 steps)

Everything below was done in this order. Each step lists the file and the functions touched.

### Step 1 — Read the target `config.json` and diff against a supported model

Before writing code, compare the HF configs. For `Meta-Llama-3.1-8B-Instruct` vs `Qwen3-0.6B`
the load-bearing differences were:

| config field | Qwen3-0.6B | Llama-3.1-8B | consequence |
|---|---|---|---|
| `model_type` | `qwen3` | `llama` | registry key (Step 4) |
| per-head Q/K RMSNorm | yes (`q_norm`/`k_norm` weights) | **no** | attention module differs (Step 2) |
| `attention_bias` | absent (no bias) | `false` | pass through as `qkv_bias` |
| `head_dim` | explicit `128` | absent → `hidden_size / num_attention_heads` | use `getattr` fallback |
| `rope_scaling` | `null` | `{"rope_type": "llama3", factor=8, ...}` | new RoPE init path (Step 3) |
| `tie_word_embeddings` | `true` | `false` (8B) | conditional weight tying |
| `num_key_value_heads` | 8 | 8 (GQA 32:8) | affects KV sizing only |

Rule of thumb: anything that changes **weight names or shapes** needs model code; anything that
only changes **numbers** should flow through `config` untouched.

### Step 2 — Write the model definition

File: `nanovllm/models/llama.py` (new). It mirrors `nanovllm/models/qwen3.py` and reuses all
existing parallel layers — no new kernels were written:

- `LlamaAttention` — same skeleton as `Qwen3Attention` but **without** `q_norm`/`k_norm`.
  Composes `QKVParallelLinear` → `get_rope` → `Attention` (paged KV + FlashAttention) →
  `RowParallelLinear`. Head counts are divided by `dist.get_world_size()` for TP.
- `LlamaMLP` — identical to Qwen3's: `MergedColumnParallelLinear` (gate+up fused) →
  `SiluAndMul` → `RowParallelLinear`. Asserts `hidden_act == "silu"`.
- `LlamaDecoderLayer` — pre-norm residual layout using the fused `RMSNorm(x, residual)` path.
  This is where config fields are unpacked (`attention_bias`, `head_dim`, `rope_theta`,
  `rope_scaling`) with `getattr` fallbacks so older Llama configs (Llama 2, TinyLlama) also work.
- `LlamaModel` — embedding + N layers + final norm.
- `LlamaForCausalLM` — adds `lm_head` (`ParallelLMHead`), ties it to `embed_tokens` only when
  `config.tie_word_embeddings` (true for Llama 3.2 1B/3B, false for 8B), and declares
  `packed_modules_mapping` (see Step 5).

The engine-facing contract is small: `forward(input_ids, positions) -> hidden_states` plus
`compute_logits(hidden_states)`. Everything about KV cache, block tables, and attention metadata
is inherited from the shared `Attention` layer, so the model file contains zero paging logic.

### Step 3 — Implement the Llama 3.1 RoPE variant

File: `nanovllm/layers/rotary_embedding.py`.

Llama 3.1 extends context 8k → 128k by rescaling RoPE frequencies, declared in
`config.rope_scaling` with `rope_type: "llama3"`. Without this the model loads fine but degrades
sharply past ~8k tokens — a silent correctness bug, which is why it must be ported, not skipped.

- `_apply_llama3_scaling(inv_freq, rope_scaling)` (new) — reimplements HF's
  `ROPE_INIT_FUNCTIONS["llama3"]`: long-wavelength frequencies are divided by `factor` (8),
  short ones kept, and a smooth interpolation is applied in between using
  `low_freq_factor` / `high_freq_factor` / `original_max_position_embeddings`.
- `RotaryEmbedding.__init__` — applies the scaling to `inv_freq` **before** building
  `cos_sin_cache`, so the hot path (`forward`, `@torch.compile`d) is unchanged.
- `get_rope(...)` — gained a `rope_scaling: dict | None` parameter. Since dicts aren't hashable,
  it converts the dict to a sorted tuple for the `lru_cache` key (`_get_rope_cached`).

Verification: compared `cos_sin_cache` against HF's `LlamaRotaryEmbedding` output for the same
config — max abs diff at float32 resolution.

### Step 4 — Registry + engine wiring

Files: `nanovllm/models/__init__.py` (new), `nanovllm/engine/model_runner.py`.

- `_MODEL_REGISTRY: dict[str, type]` maps `config.model_type` → model class
  (`{"qwen3": Qwen3ForCausalLM, "llama": LlamaForCausalLM}`).
- `create_causal_lm(config)` looks up `config.model_type` and raises a `ValueError` listing
  supported types if unknown.
- `ModelRunner.__init__` replaced the hardcoded `Qwen3ForCausalLM(hf_config)` with
  `create_causal_lm(hf_config)`. This is the **only** engine change needed for a new family —
  scheduler, block manager, CUDA-graph capture are all model-agnostic.

Adding the next family (e.g. Mistral) is therefore: write `nanovllm/models/mistral.py`,
add one registry entry.

### Step 5 — Weight loading (no changes needed, but must be understood)

File: `nanovllm/utils/loader.py`, `load_model()`.

The loader walks every tensor in the checkpoint's `*.safetensors` files. Checkpoints store
`q_proj`/`k_proj`/`v_proj` and `gate_proj`/`up_proj` as separate tensors, but our modules fuse
them (`qkv_proj`, `gate_up_proj`). The bridge is the class attribute on `LlamaForCausalLM`:

```python
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}
```

`load_model()` rewrites the checkpoint name to the fused parameter name and calls that
parameter's `weight_loader(param, tensor, shard_id)`, which copies into the right row-slice
(and the right TP shard). Because Llama's weight names match Qwen3's pattern, the mapping is
identical and the loader needed zero changes. A new family with different names (e.g. fused
checkpoints, MoE experts) would need its own mapping.

### Verification checklist

1. **Smoke test**: `example.py` with the chat template — model self-identifies as Llama and
   answers a factual question correctly (prime list).
2. **RoPE parity**: compare scaled `inv_freq`/`cos_sin_cache` against HF for the same config.
3. **Benchmarks**: `experiments/run_compare.sh --engines nanovllm,pytorch` — nano output token
   counts match the HF baseline for the same sampling setup; 8B eager hit ~357 tok/s vs
   ~51 tok/s HF-sequential on a 24 GB card.

## KV-Cache Sizing: What You Query vs. What You Compute

File: `nanovllm/utils/kvcache.py` (helpers), `nanovllm/engine/model_runner.py`
(`allocate_kv_cache`, the authoritative runtime version).

### Queried directly from `config.json` (no math)

| value | config field | Llama-3.1-8B |
|---|---|---|
| layers | `num_hidden_layers` | 32 |
| KV heads | `num_key_value_heads` (GQA — **not** `num_attention_heads`) | 8 |
| head dim | `head_dim`, else `hidden_size / num_attention_heads` | 4096/32 = 128 |
| dtype size | `torch_dtype` → `hf_config.dtype.itemsize` | bf16 → 2 B |

Using `num_attention_heads` (32) instead of `num_key_value_heads` (8) is the classic mistake —
it overestimates KV memory by the GQA ratio (4x here).

### Computed by hand

**Per-token KV bytes** (the number worth memorizing):

```
kv_bytes/token = 2 (K and V) x layers x kv_heads x head_dim x dtype_size / tp_size
Llama-3.1-8B   = 2 x 32 x 8 x 128 x 2 = 131,072 B = 128 KiB/token
```

**Per-block bytes** (block = 256 tokens, `Config.kvcache_block_size`) — implemented in
`kvcache.block_bytes()` and duplicated inline in `allocate_kv_cache()`:

```
block_bytes = 128 KiB x 256 = 32 MiB/block
```

**Capacity sanity check for a 24 GB card:**

```
weights (bf16)            ~ 16.0 GiB
activations + CUDA ctx    ~  2-3 GiB   (measured, not guessed — see below)
usable @ util=0.90        ~ 21.2 GiB
leftover for KV           ~  2.5-3.7 GiB  ->  81-119 blocks -> 20k-30k token slots
                          -> ~5-7 concurrent seqs at max_model_len=4096
```

This matches the startup banner printed by `format_kvcache_capacity()`:
`KV cache: 81 blocks x 256 tokens = 20736 total slots (2.53 GiB), ~5 seqs at max_model_len=4096`.

### How the engine computes it at runtime (measured, not estimated)

`ModelRunner.allocate_kv_cache()` does not trust hand math for the activation term. Sequence:

1. `warmup_model()` runs a max-size dummy batch first, so PyTorch's peak-memory stats reflect
   real activation pressure.
2. Query the allocator: `torch.cuda.mem_get_info()` for `(free, total)`, plus
   `memory_stats()["allocated_bytes.all.peak"]` and `...all.current"`.
3. Solve for the block budget:

```
num_blocks = (total * gpu_memory_utilization - used - peak + current) // block_bytes
```

`- peak + current` reserves headroom equal to the worst activation spike observed during
warmup, which is why the same model yields fewer blocks when other processes hold VRAM
(81 vs 119 blocks in our runs — another process was using ~1.2 GiB).

4. Allocate one big tensor `kv_cache[2, layers, num_blocks, block_size, kv_heads, head_dim]`
   and hand each `Attention` module a view (`kv_cache[0, layer_id]` / `kv_cache[1, layer_id]`).

### Practical implications observed on 24 GB / Llama-3.1-8B

- 20736 slots is tiny: 8 concurrent seqs x ~1900 tokens (shared-prefix bench) ≈ 15k slots
  already fits only because prefix caching dedupes the shared 7 blocks.
- Raising concurrency past capacity triggers recompute-based preemption
  (`Scheduler.preempt`), visible via the `Preemptions=` counter in `bench_nanovllm.py`.
- Per-token cost scales with `layers x kv_heads x head_dim`, **not** with parameter count:
  Qwen3-0.6B (2 x 28 x 8 x 128 x 2 = 112 KiB/token) costs almost as much per token as
  Llama-8B (128 KiB/token) despite being 13x smaller — GQA keeps KV width constant while
  weights shrink, so small models get proportionally *more* KV blocks only because their
  weights leave more VRAM free.

## Files/functions

- `nanovllm/models/llama.py`: `LlamaAttention`, `LlamaMLP`, `LlamaDecoderLayer`, `LlamaModel`,
  `LlamaForCausalLM`, `packed_modules_mapping`.
- `nanovllm/models/__init__.py`: `_MODEL_REGISTRY`, `create_causal_lm`, `supported_model_types`.
- `nanovllm/layers/rotary_embedding.py`: `_apply_llama3_scaling`, `RotaryEmbedding.__init__`,
  `get_rope`, `_get_rope_cached`.
- `nanovllm/engine/model_runner.py`: `__init__` (`create_causal_lm`), `allocate_kv_cache`
  (block budget + capacity banner).
- `nanovllm/utils/kvcache.py`: `block_bytes`, `kvcache_capacity`, `format_kvcache_capacity`.
- `nanovllm/utils/loader.py`: `load_model` (consumes `packed_modules_mapping`).

## Common Failure Modes

- **Missing `rope_scaling` handling**: generations look fine at short context, degrade at long
  context. Always diff RoPE config fields when porting.
- **`num_attention_heads` vs `num_key_value_heads`** in KV sizing: 4x overestimate under GQA.
- **Unconditional weight tying**: `lm_head.weight` must only alias `embed_tokens.weight` when
  `tie_word_embeddings` is true, otherwise the 8B's real `lm_head` tensor is ignored.
- **Unhashable `rope_scaling` dict** passed into an `lru_cache`d factory: convert to a sorted
  tuple key.

## Exercises

1. Port Llama-3.2-1B (`tie_word_embeddings=true`) and confirm the tying branch is exercised.
2. Compute KV bytes/token for Qwen3-0.6B by hand, then verify against the startup banner.
3. Set `gpu_memory_utilization=0.95` and explain the change in block count using the
   `allocate_kv_cache` formula.
