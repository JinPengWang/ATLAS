"""CEC 2013 benchmark suite registration (F1-F28)."""

from __future__ import annotations

from atlas.problems.cec.base_cec import make_cec_class
from atlas.utils.registry import register_problem

_CEC2013_FUNCTIONS = list(range(1, 29))  # F1-F28
_DEFAULT_DIM = 30  # supports 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100


def register_cec2013() -> None:
    """Register all CEC 2013 benchmark functions."""
    import opfunu.cec_based.cec2013 as cec2013

    for fid in _CEC2013_FUNCTIONS:
        cls_name = f"F{fid}2013"
        func_cls = getattr(cec2013, cls_name, None)
        if func_cls is None:
            continue
        problem_cls = make_cec_class(func_cls, fid, "2013", _DEFAULT_DIM)
        register_problem(f"cec2013_f{fid}")(problem_cls)
