"""Deterministic sampling reference (oracle).

Students must implement Sampler themselves; do not call this from production code.
"""

from __future__ import annotations

import torch


def reference_sample(logits: torch.Tensor, temperatures: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Gumbel-max sampling matching nano-vLLM's Sampler algorithm."""
    scaled = logits.float() / temperatures.unsqueeze(1)
    probs = torch.softmax(scaled, dim=-1)
    gumbel = torch.empty_like(probs).exponential_(1, generator=generator).clamp_min_(1e-10)
    return (probs / gumbel).argmax(dim=-1)
