"""GPU smoke tests (skipped without CUDA)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu


def test_cuda_available_marker(require_cuda):
    import torch
    assert torch.cuda.is_available()
