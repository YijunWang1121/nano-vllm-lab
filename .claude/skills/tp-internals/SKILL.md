---
name: tp-internals
description: Reference for nano-vllm's hand-rolled tensor-parallel RPC mechanism in nanovllm/engine/model_runner.py. Use before reading, modifying, or debugging ModelRunner, its multi-process setup, or anything involving rank>0 workers, SharedMemory, or the .call()/.loop() dispatch pattern.
---

# Tensor-parallel internals (`nanovllm/engine/model_runner.py`)

nano-vllm does not use `torch.multiprocessing.spawn` or a framework RPC layer for tensor
parallelism. It's a hand-rolled rank-0-drives-the-rest design:

- One `ModelRunner` process per rank. All ranks join the same NCCL process group at
  `tcp://localhost:2333` (hardcoded — a second concurrent run on the same machine will
  collide on this port).
- If `world_size > 1`: rank 0 creates a `multiprocessing.shared_memory.SharedMemory`
  block named `"nanovllm"` (2**20 bytes) and returns from `__init__` to the caller
  (`LLMEngine`). Every other rank attaches to that same shared-memory block and then
  blocks forever inside `self.loop()` — it never returns from `__init__`.
- **Dispatch**: rank 0 calls `self.call(method_name, *args)` for things like `run(...)`.
  `call()` first does `write_shm(method_name, *args)` (pickles `[method_name, *args]`
  into the shared-memory buffer, prefixed with a 4-byte little-endian length, then sets
  every worker's `Event`), and *then* invokes the method locally via
  `getattr(self, method_name)(*args)`. Worker ranks are woken by their `Event`, read
  and unpickle the same call out of shared memory in `read_shm()`, and invoke it via
  `self.call(...)` too — so all ranks end up running the identical method with identical
  args, rank 0 by direct call, others by RPC replay.
- Because of this, **any new method that needs to run on all ranks must be invoked via
  `self.call("method_name", ...)` from rank 0**, not called directly as
  `self.method_name(...)` — a direct call only executes locally on rank 0 and workers
  will never see it.
- Shutdown: `call("exit", ...)` is how the loop on worker ranks breaks — `loop()` checks
  `if method_name == "exit": break` after invoking it.
- `Sequence` (in `engine/sequence.py`) implements custom `__getstate__`/`__setstate__`
  specifically so instances pickle cheaply across this shared-memory boundary — don't
  add large/expensive fields to `Sequence` without checking that pickling path.
- Only rank 0 has real sampling output: `token_ids = self.sampler(...).tolist() if
  self.rank == 0 else None` in `run()` — worker ranks return `None`.

When touching this file: preserve the rank-0-drives-workers invariant, keep new
cross-rank calls going through `self.call(...)`, and remember worker ranks are running
inside `loop()`/`read_shm()`, not in normal call/return flow.
