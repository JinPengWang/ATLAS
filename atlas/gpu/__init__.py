"""GPU acceleration module."""

from __future__ import annotations

from atlas.gpu.torch_engine import (
    TorchBenchmarkProblem,
    get_default_device,
    is_torch_available,
)

__all__ = [
    "is_torch_available",
    "get_default_device",
    "TorchBenchmarkProblem",
]
