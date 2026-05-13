"""Problems sub-package.

Importing this package automatically registers all built-in benchmark and
custom problems via the :mod:`atlas.utils.registry` mechanism.
"""

from atlas.problems.benchmark import (
    Ackley,
    Griewank,
    Levy,
    Michalewicz,
    Quartic,
    Rastrigin,
    Rosenbrock,
    Schwefel,
    Schwefel222,
    Sphere,
)
from atlas.problems.custom import PressureVessel

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
    # Custom / engineering
    "PressureVessel",
]
