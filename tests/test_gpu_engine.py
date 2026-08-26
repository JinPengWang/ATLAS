"""Unit tests for GPU acceleration engine."""

import numpy as np
import pytest

from atlas.gpu import (
    TorchBenchmarkProblem,
    get_default_device,
    is_torch_available,
)


def test_is_torch_available_returns_bool():
    avail = is_torch_available()
    assert isinstance(avail, bool)


def test_get_default_device():
    dev = get_default_device()
    assert dev in ("cuda", "mps", "cpu")


@pytest.mark.skipif(not is_torch_available(), reason="PyTorch is not installed")
def test_torch_benchmark_sphere():
    prob = TorchBenchmarkProblem("sphere", dim=10)
    X = np.ones((5, 10), dtype=np.float32) * 2.0
    # Sphere of 2.0 on 10 dims = 10 * 4.0 = 40.0
    res = prob.evaluate_numpy(X)
    assert len(res) == 5
    assert np.allclose(res, 40.0)


@pytest.mark.skipif(not is_torch_available(), reason="PyTorch is not installed")
def test_torch_benchmark_rastrigin():
    prob = TorchBenchmarkProblem("rastrigin", dim=5)
    X = np.zeros((3, 5), dtype=np.float32)
    # At 0.0, Rastrigin is exactly 0.0
    res = prob.evaluate_numpy(X)
    assert np.allclose(res, 0.0)
