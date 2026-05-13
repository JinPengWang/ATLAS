"""Benchmark problems sub-package (unimodal and multimodal)."""

from atlas.problems.benchmark.multimodal import (
    Ackley,
    Griewank,
    Levy,
    Michalewicz,
    Rastrigin,
    Schwefel,
)
from atlas.problems.benchmark.unimodal import (
    Quartic,
    Rosenbrock,
    Schwefel222,
    Sphere,
)

__all__ = [
    # Unimodal
    "Sphere",
    "Rosenbrock",
    "Schwefel222",
    "Quartic",
    # Multimodal
    "Rastrigin",
    "Ackley",
    "Griewank",
    "Levy",
    "Schwefel",
    "Michalewicz",
]
