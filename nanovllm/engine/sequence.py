from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams
from nanovllm.course.exceptions import CourseNotImplementedError


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # TODO-L1-SEQ-01: Initialize sequence tracking fields.
        #
        # Goal:
        # Create the per-request state object that the scheduler, block manager,
        # and model runner all share for the lifetime of one generation.
        #
        # Called from:
        # - LLMEngine.add_request -> Sequence(prompt_token_ids, sampling_params)
        #
        # Calls next:
        # - Scheduler.add places this object into the waiting queue
        #
        # Inputs:
        # - token_ids: prompt token ids (already encoded)
        # - sampling_params: temperature, max_tokens, ignore_eos
        #
        # Required fields to set:
        # - seq_id from Sequence.counter
        # - status = WAITING
        # - token_ids (copy), last_token, num_tokens, num_prompt_tokens
        # - num_cached_tokens = 0, num_scheduled_tokens = 0
        # - is_prefill = True, block_table = []
        # - temperature / max_tokens / ignore_eos from sampling_params
        #
        # Invariants:
        # - num_tokens == len(token_ids) at construction
        # - num_prompt_tokens is fixed for the sequence lifetime
        # - block_table starts empty until BlockManager.allocate
        #
        # Read: docs/tutorial/02_request_and_sequence.md
        # Tests: pytest tests/milestones/test_01_sequence.py
        # Hints: use copy(token_ids); last_token is token_ids[-1]
        raise CourseNotImplementedError(
            "TODO-L1-SEQ-01",
            subsystem="sequence",
            tutorial_path="docs/tutorial/02_request_and_sequence.md",
            milestone_test="pytest tests/milestones/test_01_sequence.py",
            hint="Copy prompt tokens and initialize cache/schedule counters to 0.",
        )

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        # TODO-L1-SEQ-03: Compute how many logical KV blocks this sequence needs.
        #
        # Goal:
        # Map token length to the number of block_size pages required.
        #
        # Formula:
        #   ceil(num_tokens / block_size)
        #
        # Example (block_size=4, prompt len=5): num_blocks == 2
        #
        # Read: docs/tutorial/02_request_and_sequence.md
        # Tests: pytest tests/milestones/test_01_sequence.py
        raise CourseNotImplementedError(
            "TODO-L1-SEQ-03",
            subsystem="sequence",
            tutorial_path="docs/tutorial/02_request_and_sequence.md",
            milestone_test="pytest tests/milestones/test_01_sequence.py",
            hint="(num_tokens + block_size - 1) // block_size",
        )

    @property
    def last_block_num_tokens(self):
        # TODO-L1-SEQ-03 (continued): tokens residing in the final logical block.
        #
        # Example (len=5, block_size=4): last_block_num_tokens == 1
        # Used by decode slot_mapping:
        #   slot = block_table[-1] * block_size + last_block_num_tokens - 1
        raise CourseNotImplementedError(
            "TODO-L1-SEQ-03",
            subsystem="sequence",
            tutorial_path="docs/tutorial/02_request_and_sequence.md",
            milestone_test="pytest tests/milestones/test_01_sequence.py",
            hint="num_tokens - (num_blocks - 1) * block_size",
        )

    def block(self, i):
        # TODO-L1-SEQ-03 (continued): return token_ids for logical block i.
        #
        # Slice token_ids[i*block_size : (i+1)*block_size]
        # Assert 0 <= i < num_blocks
        # Used by BlockManager hashing / prefix cache matching.
        raise CourseNotImplementedError(
            "TODO-L1-SEQ-03",
            subsystem="sequence",
            tutorial_path="docs/tutorial/02_request_and_sequence.md",
            milestone_test="pytest tests/milestones/test_01_sequence.py",
            hint="Slice the token_ids list for logical block i.",
        )

    def append_token(self, token_id: int):
        # TODO-L1-SEQ-02: Append a newly sampled token.
        #
        # Goal:
        # Extend the sequence after the sampler returns a token id.
        #
        # Called from:
        # - Scheduler.postprocess (after a completed prefill chunk or decode step)
        #
        # Required behavior:
        # 1. Append token_id to token_ids
        # 2. Update last_token
        # 3. Increment num_tokens
        #
        # Do NOT change num_prompt_tokens.
        # Do NOT modify block_table here (BlockManager.may_append handles pages).
        #
        # Read: docs/tutorial/02_request_and_sequence.md
        # Tests: pytest tests/milestones/test_01_sequence.py
        raise CourseNotImplementedError(
            "TODO-L1-SEQ-02",
            subsystem="sequence",
            tutorial_path="docs/tutorial/02_request_and_sequence.md",
            milestone_test="pytest tests/milestones/test_01_sequence.py",
            hint="Append to token_ids and keep last_token / num_tokens in sync.",
        )

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
