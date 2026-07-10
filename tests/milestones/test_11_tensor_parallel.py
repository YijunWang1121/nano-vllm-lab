"""Milestone 11: tensor-parallel weight sharding and all-reduce (CPU-mockable)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from nanovllm.course.exceptions import CourseNotImplementedError


def test_column_parallel_weight_loader_shards():
    import torch.distributed as dist
    from nanovllm.layers.linear import ColumnParallelLinear

    with patch.object(dist, "get_rank", return_value=1), patch.object(dist, "get_world_size", return_value=2):
        layer = ColumnParallelLinear(4, 8, bias=False)
        # Full weight [out=8, in=4]; rank1 should get rows 4:8
        full = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        try:
            layer.weight_loader(layer.weight, full.clone())
        except CourseNotImplementedError as e:
            pytest.fail(str(e))
        assert torch.equal(layer.weight.data, full[4:8])


def test_row_parallel_all_reduce_called():
    import torch.distributed as dist
    from nanovllm.layers.linear import RowParallelLinear

    calls = []

    def fake_all_reduce(tensor):
        calls.append(tensor.clone())

    with patch.object(dist, "get_rank", return_value=0), patch.object(dist, "get_world_size", return_value=2), patch.object(dist, "all_reduce", side_effect=fake_all_reduce):
        layer = RowParallelLinear(4, 3, bias=False)
        layer.weight.data.fill_(0.5)
        x = torch.ones(2, 2)  # input_size per rank = 4/2 = 2
        try:
            y = layer(x)
        except CourseNotImplementedError as e:
            pytest.fail(str(e))
        assert y.shape == (2, 3)
        assert len(calls) == 1


def test_row_parallel_tp1_no_all_reduce():
    import torch.distributed as dist
    from nanovllm.layers.linear import RowParallelLinear

    with patch.object(dist, "get_rank", return_value=0), patch.object(dist, "get_world_size", return_value=1), patch.object(dist, "all_reduce") as ar:
        layer = RowParallelLinear(4, 3, bias=False)
        layer.weight.data.zero_()
        y = layer(torch.ones(2, 4))
        assert y.shape == (2, 3)
        ar.assert_not_called()
