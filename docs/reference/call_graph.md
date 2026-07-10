# Call Graph Reference

This page is a compact call graph for the educational implementation.

## Public API

```text
nanovllm/__init__.py
    exports LLM, SamplingParams

nanovllm/llm.py
    class LLM(LLMEngine)
```

## Engine Construction

```text
LLMEngine.__init__(model, **kwargs)
    Config(model, **config_kwargs)
    Sequence.block_size = config.kvcache_block_size
    spawn ModelRunner worker ranks when tensor_parallel_size > 1
    ModelRunner(config, rank=0, events)
    AutoTokenizer.from_pretrained(config.model)
    Scheduler(config)
```

## Add Request

```text
LLMEngine.add_request(prompt, sampling_params)
    tokenizer.encode(prompt) if prompt is str
    Sequence(prompt, sampling_params)
    Scheduler.add(seq)
```

## Generate Loop

```text
LLMEngine.generate(prompts, sampling_params, use_tqdm=True)
    add_request(...) for each prompt
    while not is_finished():
        outputs, num_tokens = step()
        update prefill/decode throughput
    decode completion token IDs with tokenizer
```

## One Step

```text
LLMEngine.step()
    seqs, is_prefill = Scheduler.schedule()
    token_ids = ModelRunner.call("run", seqs, is_prefill)
    Scheduler.postprocess(seqs, token_ids, is_prefill)
    return finished outputs and throughput token count
```

## Scheduler

```text
Scheduler.schedule()
    prefill branch:
        BlockManager.can_allocate(seq)
        BlockManager.allocate(seq, num_cached_blocks)
        set seq.num_scheduled_tokens
        move complete prefill WAITING -> RUNNING
        return seqs, True
    decode branch:
        pop running seq
        BlockManager.can_append(seq)
        maybe preempt(...)
        BlockManager.may_append(seq)
        return seqs, False
```

```text
Scheduler.postprocess(seqs, token_ids, is_prefill)
    BlockManager.hash_blocks(seq)
    advance num_cached_tokens
    skip append if incomplete chunked prefill
    Sequence.append_token(token_id)
    finish on EOS or max_tokens
    BlockManager.deallocate(seq) for finished sequences
```

## Block Manager

```text
BlockManager.can_allocate(seq)
    compute chained prefix hashes over full blocks
    count reusable blocks
    verify enough free blocks

BlockManager.allocate(seq, num_cached_blocks)
    append reused cached physical IDs
    allocate new physical IDs
    set seq.num_cached_tokens

BlockManager.hash_blocks(seq)
    hash newly completed full blocks
    update hash_to_block_id
```

## Model Runner

```text
ModelRunner.run(seqs, is_prefill)
    prepare_prefill(seqs) or prepare_decode(seqs)
    prepare_sample(seqs) on rank 0
    logits = run_model(input_ids, positions, is_prefill)
    token_ids = Sampler(logits, temperatures) on rank 0
    reset_context()
```

```text
ModelRunner.run_model(input_ids, positions, is_prefill)
    if prefill or enforce_eager or batch_size > 512:
        model(input_ids, positions)
        compute_logits(...)
    else:
        copy inputs/context into graph_vars
        graph.replay()
        compute_logits(outputs[:batch_size])
```

## Attention

```text
Attention.forward(q, k, v)
    context = get_context()
    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
    if context.is_prefill:
        flash_attn_varlen_func(...)
    else:
        flash_attn_with_kvcache(...)
```

## Tensor Parallel Worker Calls

```text
ModelRunner.call(method_name, *args) on rank 0
    write method call to SharedMemory
    signal worker events
    run local method

ModelRunner.loop() on worker ranks
    wait for event
    read method call from SharedMemory
    call method
```
