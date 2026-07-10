# 06. Model Runner Inputs

## Learning Objectives

- Build CPU metadata lists for prefill and decode.
- Convert `block_table` into padded block tables and physical slot mappings.
- Understand what `ModelRunner.prepare_prefill()` and `prepare_decode()` send to CUDA.

## Why Exists

The scheduler works with sequences and counters. Attention kernels need flat tensors, cumulative sequence lengths, context lengths, and physical KV-cache slots.

## Call Path

`ModelRunner.run()` calls `prepare_prefill(seqs)` when `is_prefill=True` and `prepare_decode(seqs)` otherwise. These call helpers in `nanovllm/engine/input_metadata.py`, convert lists to CUDA tensors, and set `nanovllm.utils.context`.

## Files/functions

- `nanovllm/engine/input_metadata.py`: `build_block_tables`, `build_prefill_metadata`, `build_decode_metadata`.
- `nanovllm/engine/model_runner.py`: `prepare_block_tables`, `prepare_prefill`, `prepare_decode`, `prepare_sample`.
- `tests/milestones/test_05_model_inputs.py`.
- `tests/milestones/test_06_attention_metadata.py`.

## Data Structures

The builders return dictionaries with list values. `build_prefill_metadata()` returns `input_ids`, `positions`, `cu_seqlens_q`, `cu_seqlens_k`, `max_seqlen_q`, `max_seqlen_k`, `slot_mapping`, and `need_block_tables`.

## State Transitions

Metadata builders should not mutate sequence state. They read `num_cached_tokens`, `num_scheduled_tokens`, `token_ids`, and `block_table` to describe the already-scheduled work.

## Tensor Shapes

For A `block_table=[7,2]`, B `block_table=[5]`, block size 4:

- `build_block_tables([A,B])` returns `[[7,2],[5,-1]]`.
- Full prefill input has 8 tokens.
- Full prefill `positions` are `[0,1,2,3,4,0,1,2]`.
- Decode `slot_mapping` is `[8,22]`.

## Pseudocode

```text
build_prefill_metadata:
    for seq in seqs:
        start = seq.num_cached_tokens
        end = start + seq.num_scheduled_tokens
        append seq[start:end] to input_ids
        append range(start, end) to positions
        update cumulative q and k lengths
        map each scheduled token to physical cache slot
```

## TODO IDs

- `TODO-L2-RUNNER-01`: build padded block tables.
- `TODO-L2-RUNNER-02`: build prefill input metadata and slot mapping.
- `TODO-L2-RUNNER-03`: build decode input metadata.

## Hints

The slot formula is `physical_block_id * block_size + offset_within_block`. For decode, the offset is `seq.last_block_num_tokens - 1` because decode feeds the current last token.

## Common Bugs

- Padding block tables with `0` instead of `-1`.
- Using total prompt length for `cu_seqlens_q` during prefix-cache prefill.
- Forgetting that prefix-cache prefill has `cu_seqlens_k > cu_seqlens_q`.
- Writing decode slot mapping as the next free slot instead of the current last-token slot.

## Tests

Run:

```bash
python tools/run_milestone.py 5
```

Then run attention metadata checks:

```bash
python tools/run_milestone.py 6 --no-test
python -m pytest -q tests/milestones/test_06_attention_metadata.py
```

## Debugging

Call the pure Python builders directly from a REPL. Because they do not touch CUDA, they are the best place to debug shape and slot errors.

## Expected Intermediate Behavior

For prefix-cache prefill where A has `num_cached_tokens=4` and `num_scheduled_tokens=1`, metadata should schedule only `[14]`, position `[4]`, slot `[8]`, and `need_block_tables=True`.

## Reflection

Metadata is the bridge from scheduler semantics to kernel semantics. A single off-by-one here usually looks like an attention bug later.

## Connection to Production vLLM

Production vLLM builds richer attention metadata objects, but they carry the same information: sequence lengths, block tables, cache slots, and whether the batch is prefill or decode.
