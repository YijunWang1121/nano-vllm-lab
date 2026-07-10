import torch
from torch import nn

from nanovllm.utils.debug import debug_log


class Sampler(nn.Module):

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # Keep sampling logic uncompiled so CPU milestone tests can run without
        # torch.compile inductor issues on non-CUDA machines.
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        debug_log("sampling", "forward", tokens=sample_tokens.tolist(), temperatures=temperatures.tolist())
        return sample_tokens
