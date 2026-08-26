"""Problems sub-package.

Importing this package automatically registers all built-in benchmark,
custom, engineering, and pure-Python CEC problems via the :mod:`atlas.utils.registry` mechanism.

CEC benchmark functions (2005-2022) with opfunu C extensions are also registered
when ``opfunu`` is installed (``pip install opfunu``).
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
import atlas.problems.cec_pure  # registers pure Python CEC problems
from atlas.problems.custom import PressureVessel
from atlas.problems.engineering import (
    PressureVesselEng,
    SpeedReducer,
    SpringDesign,
    Truss3Bar,
    WeldedBeam,
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
    # Custom / engineering
    "PressureVessel",
    "WeldedBeam",
    "SpringDesign",
    "SpeedReducer",
    "Truss3Bar",
    "PressureVesselEng",
]

# ---------------------------------------------------------------------------
# CEC benchmarks (optional dependency: opfunu)
# ---------------------------------------------------------------------------
try:
    from atlas.problems.cec import register_all_cec

    register_all_cec()
except ImportError:
    pass  # opfunu not installed; CEC problems unavailable
