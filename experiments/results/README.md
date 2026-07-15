# Engine-compare result artifacts

Copy JSON from the GPU pod into this directory, then commit on `experiments/engine-compare`.

## From the pod

```bash
# on pod — list
ls -lh ~/engine_compare_results/

# on your Mac (adjust host / path)
mkdir -p experiments/results
scp 'runpod:/root/engine_compare_results/20260715_223901_sweep_default.json' experiments/results/
scp 'runpod:/root/engine_compare_results/20260715_223801_sweep_quick.json' experiments/results/
# optional single-shot compares
scp 'runpod:/root/engine_compare_results/*_compare.json' experiments/results/
```

If results live under `/workspace`, use that path instead of `/root`.

## What is worth committing

| Keep | Skip |
|------|------|
| `*_sweep_*.json` (mean/std + per-repeat) | venv, models, pip cache |
| Optional: a few `*_compare.json` | Huge logs / terminal dumps |
| This folder + `ENGINE_COMPARE_REPORT.md` | |

Do **not** commit `~/venv-nanovllm` or Hugging Face weights.
