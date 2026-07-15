<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Course project fork

This repository includes an educational course conversion:

- Start here: [`COURSE.md`](COURSE.md)
- Student branch: `course/student` (implement TODOs)
- Reference branch: `course/reference-solutions` (complete solutions — do not work here)
- Conversion report: [`COURSE_CONVERSION_REPORT.md`](COURSE_CONVERSION_REPORT.md)

```bash
git checkout course/student
python tools/course_status.py
python tools/run_milestone.py 1
```

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Ablation / A-B experiments

Config switches (defaults match upstream behavior):

- `enforce_eager` — disable CUDA graphs
- `enable_prefix_caching` — disable automatic prefix KV reuse
- `enable_preemption` — disable decode preemption under KV pressure
- `enable_chunked_prefill` — disable partial-prefill scheduling
- also: `max_num_seqs`, `max_num_batched_tokens`, `gpu_memory_utilization`, `tensor_parallel_size`

```bash
export PYTHONPATH=.
# KV pool size on this GPU (dynamic)
python experiments/print_kv_capacity.py

# CUDA graph ablation (random prompts)
python experiments/run_ablation.py --preset baseline,no_cudagraph --warmup

# Prefix-cache ablation (shared system prompt — required to see a gap)
python experiments/run_ablation.py --preset baseline,no_prefix \
  --workload shared_prefix --shared-prefix-len 512 --warmup

# Multi-turn chat simulation
python experiments/run_ablation.py --preset baseline,no_prefix \
  --workload multi_turn --num-sessions 16 --num-turns 4 \
  --shared-prefix-len 512 --warmup
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)