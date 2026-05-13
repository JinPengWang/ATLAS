"""CEC 2017 benchmark suite registration (F1-F29).

Note: Dimensions must be even (2, 10, 20, 30, 50, 100).
"""

from __future__ import annotations

from atlas.problems.cec.base_cec import make_cec_class
from atlas.utils.registry import register_problem

_CEC2017_FUNCTIONS = list(range(1, 30))  # F1-F29
_DEFAULT_DIM = 30  # supports 2, 10, 20, 30, 50, 100 (must be even)


def register_cec2017() -> None:
    """Register all CEC 2017 benchmark functions."""
    import opfunu.cec_based.cec2017 as cec2017

    for fid in _CEC2017_FUNCTIONS:
        cls_name = f"F{fid}2017"
        func_cls = getattr(cec2017, cls_name, None)
        if func_cls is None:
            continue
        problem_cls = make_cec_class(func_cls, fid, "2017", _DEFAULT_DIM)
        register_problem(f"cec2017_f{fid}")(problem_cls)
