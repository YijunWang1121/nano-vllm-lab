import torch
from torch import nn

from nanovllm.utils.debug import debug_log

class Sampler(nn.Module):

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        N, vocab = logits.shape
        logits = logits.float() / temperatures.unsqueeze(1)
        probs= torch.softmax(logits, dim=-1)
        noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        return (probs / noise).argmax(dim=-1)
