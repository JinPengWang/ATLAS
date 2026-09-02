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
_has_opfunu = False
try:
    from atlas.problems.cec import register_all_cec
    register_all_cec()
    _has_opfunu = True
except ImportError:
    pass  # opfunu not installed; CEC problems unavailable via opfunu
except Exception as _exc:  # e.g. ValueError from registry conflict
    import warnings
    warnings.warn(f"CEC registration failed: {_exc}")

# If opfunu is not available, expose pure-Python CEC2017 under the standard names
if not _has_opfunu:
    from atlas.utils.registry import ProblemRegistry
    _pure_map = {
        "cec2017_f1": "cec2017_f1_pure",
        "cec2017_f3": "cec2017_f3_pure",
        "cec2017_f4": "cec2017_f4_pure",
        "cec2017_f5": "cec2017_f5_pure",
        "cec2017_f6": "cec2017_f6_pure",
        "cec2017_f7": "cec2017_f7_pure",
        "cec2017_f8": "cec2017_f8_pure",
        "cec2017_f9": "cec2017_f9_pure",
        "cec2017_f10": "cec2017_f10_pure",
    }
    _prob_reg = ProblemRegistry._registry
    for _std, _pure in _pure_map.items():
        if _pure in _prob_reg and _std not in _prob_reg:
            _prob_reg[_std] = _prob_reg[_pure]
