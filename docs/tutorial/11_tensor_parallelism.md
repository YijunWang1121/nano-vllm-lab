# 11. Tensor Parallelism

## Learning Objectives

- Explain column-parallel and row-parallel linear layers.
- Load rank-local weight shards correctly.
- Understand why `RowParallelLinear.forward()` needs `dist.all_reduce()`.

## Why Exists

Tensor parallelism splits model weights across GPUs so a larger model or larger batch can run than would fit on one device.

## Call Path

`ModelRunner.__init__()` initializes an NCCL process group and creates one `Qwen3ForCausalLM` per rank. Weight loading calls each parameter's `weight_loader`. Forward passes use parallel linear layers in `nanovllm/layers/linear.py`.

## Files/functions

- `nanovllm/engine/model_runner.py`: NCCL setup, spawned ranks, `SharedMemory`, worker `loop`.
- `nanovllm/layers/linear.py`: `divide`, `LinearBase`, `ColumnParallelLinear`, `MergedColumnParallelLinear`, `QKVParallelLinear`, `RowParallelLinear`.
- `nanovllm/utils/loader.py`: model weight loading path.
- `tests/milestones/test_11_tensor_parallel.py`.

## Data Structures

Each parallel layer knows `tp_rank` and `tp_size` from `torch.distributed`. Weight shards are stored as normal parameters, but only a slice of the full checkpoint tensor is loaded per rank.

## State Transitions

Tensor-parallel setup is initialized once per `ModelRunner`. Rank 0 sends method calls to worker ranks through shared memory. Worker ranks execute the same method and wait for the next call.

## Tensor Shapes

For `ColumnParallelLinear(input_size, output_size)`, each rank owns `output_size / tp_size` rows. For `RowParallelLinear(input_size, output_size)`, each rank owns `input_size / tp_size` columns and all-reduces the partial output.

## Pseudocode

```text
ColumnParallelLinear.weight_loader:
    shard_size = local weight size along output dim
    start = rank * shard_size
    copy loaded_weight.narrow(output_dim, start, shard_size)
RowParallelLinear.forward:
    y = linear(local_x, local_weight, bias only on rank 0)
    all_reduce(y) if tp_size > 1
```

## TODO IDs

- `TODO-L2-TP-01`: column-parallel weight shard loading.
- `TODO-L3-TP-02`: row-parallel all-reduce.

## Hints

`QKVParallelLinear` is a special column-parallel layer. It loads Q, K, and V shards into different offsets of one packed local parameter.

## Common Bugs

- Sharding along the wrong dimension.
- Applying bias on every row-parallel rank before all-reduce.
- Forgetting that `dist.all_reduce()` modifies the tensor in place.
- Allowing hidden sizes that are not divisible by `tp_size`.

## Tests

Run:

```bash
python tools/run_milestone.py 11
```

The milestone is marked CPU-capable in the course registry, but distributed setup still depends on the test harness.

## Debugging

Print local parameter shapes, `tp_rank`, and `tp_size`. For row parallelism, compare the all-reduced output to a single full linear layer in a small test.

## Expected Intermediate Behavior

With `tp_size=2`, each column-parallel rank holds half the output rows. Each row-parallel rank computes a partial output with the same final output shape, and all-reduce sums the partials.

## Reflection

Tensor parallelism is a shape discipline. Most bugs are not mysterious distributed failures; they are wrong slices.

## Connection to Production vLLM

Production vLLM supports many more parallel modes and deployment topologies, but column and row tensor parallel linear layers are still foundational building blocks.
