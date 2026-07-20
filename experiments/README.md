# Experiments

## PyTorch / HuggingFace baseline

Stock Transformers `generate` (plain `nn.Linear` + SDPA/eager attention).  
No nano-vLLM scheduler, paged KV, or CUDA-graph decode.

```bash
# Needs CUDA + local model (same default path as bench.py)
python experiments/bench_pytorch_baseline.py

# Smaller smoke
python experiments/bench_pytorch_baseline.py --num-seqs 8 --max-input-len 128 --max-output-len 64

# Compare to nano-vLLM
python bench.py
```

| Flag | Meaning |
|------|---------|
| `--mode sequential` | One request at a time (default; honest no-batching baseline) |
| `--mode batched` | Single padded batch (still not continuous batching) |
| `--attn-implementation sdpa\|eager\|flash_attention_2` | HF attention backend |

For lab vs vLLM sweeps / graph ablation, use branch `experiments/engine-compare` (`ENGINE_COMPARE.md`).
