import torch
from torch import nn

from nanovllm.utils.debug import debug_log
from nanovllm.course.exceptions import CourseNotImplementedError


class Sampler(nn.Module):

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # TODO-L1-SAMPLE-01: Temperature sampling via the Gumbel-max trick.
        #
        # Goal:
        # Given logits [N, vocab] and temperatures [N], return token ids [N].
        #
        # Algorithm (matches nano-vLLM):
        # 1. logits = logits.float() / temperatures.unsqueeze(1)
        # 2. probs = softmax(logits, dim=-1)
        # 3. gumbel = exponential_(1) noise, clamp_min(1e-10)
        # 4. return (probs / gumbel).argmax(dim=-1)
        #
        # Notes:
        # - SamplingParams forbids temperature ~= 0 (no pure argmax path)
        # - Rank 0 only samples in TP mode
        #
        # Read: docs/tutorial/08_sampling.md
        # Tests: pytest tests/milestones/test_07_sampling.py
        # Oracle: tests/oracles/sampling_oracle.py (do not call from here)
        raise CourseNotImplementedError(
            "TODO-L1-SAMPLE-01",
            subsystem="sampling",
            tutorial_path="docs/tutorial/08_sampling.md",
            milestone_test="pytest tests/milestones/test_07_sampling.py",
            hint="Scale by temperature, softmax, then Gumbel-max via exponential noise.",
        )
