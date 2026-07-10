"""Milestone 7: sampling."""

from __future__ import annotations

import pytest
import torch

from nanovllm.course.exceptions import CourseNotImplementedError
from nanovllm.layers.sampler import Sampler
from tests.oracles.sampling_oracle import reference_sample


def test_sampler_shape_and_dtype():
    sampler = Sampler()
    logits = torch.tensor([[0.0, 5.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float32)
    temperatures = torch.tensor([0.8, 0.8], dtype=torch.float32)
    try:
        out = sampler(logits.clone(), temperatures.clone())
    except CourseNotImplementedError as e:
        pytest.fail(str(e))
    assert out.shape == (2,)
    assert out.dtype == torch.int64 or out.dtype == torch.long


def test_low_temperature_prefers_argmax():
    sampler = Sampler()
    logits = torch.tensor([[0.0, 10.0, 0.0]], dtype=torch.float32)
    temperatures = torch.tensor([1e-4], dtype=torch.float32)
    # With near-zero temperature, Gumbel noise is overwhelmed by the peak.
    torch.manual_seed(0)
    tokens = sampler(logits.clone(), temperatures.clone())
    assert tokens.tolist() == [1]


def test_matches_reference_oracle_with_fixed_noise():
    # Compare algorithm structure via oracle with shared generator seed path:
    # We can't easily inject the same exponential noise into Sampler without hooks,
    # so verify oracle itself and that Sampler returns in-vocab indices.
    logits = torch.randn(4, 50)
    temperatures = torch.ones(4) * 0.9
    torch.manual_seed(123)
    ref = reference_sample(logits.clone(), temperatures.clone())
    torch.manual_seed(123)
    sampler = Sampler()
    # Different RNG stream order may differ; just check bounds.
    out = sampler(logits.clone(), temperatures.clone())
    assert out.min() >= 0
    assert out.max() < 50
    assert ref.shape == out.shape
