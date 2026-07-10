"""Pytest configuration for nano-vLLM course tests."""

from __future__ import annotations

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires NVIDIA CUDA GPU")


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.fixture
def require_cuda(cuda_available):
    if not cuda_available:
        pytest.skip("CUDA GPU not available")
