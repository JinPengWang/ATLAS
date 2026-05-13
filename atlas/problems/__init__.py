"""Problems sub-package.

Importing this package automatically registers all built-in benchmark and
custom problems via the :mod:`atlas.utils.registry` mechanism.

CEC benchmark functions (2005-2022) are registered when ``opfunu`` is
installed (``pip install opfunu``).
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

# ---------------------------------------------------------------------------
# CEC benchmarks (optional dependency: opfunu)
# ---------------------------------------------------------------------------
try:
    from atlas.problems.cec import register_all_cec

    register_all_cec()
except ImportError:
    pass  # opfunu not installed; CEC problems unavailable
