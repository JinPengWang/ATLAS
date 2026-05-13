"""CEC 2020 benchmark suite registration (F1-F10)."""

from __future__ import annotations

from atlas.problems.cec.base_cec import make_cec_class
from atlas.utils.registry import register_problem

_CEC2020_FUNCTIONS = list(range(1, 11))  # F1-F10
_DEFAULT_DIM = 30  # supports 2, 5, 10, 15, 20, 30, 50, 100


def register_cec2020() -> None:
    """Register all CEC 2020 benchmark functions."""
    import opfunu.cec_based.cec2020 as cec2020

    for fid in _CEC2020_FUNCTIONS:
        cls_name = f"F{fid}2020"
        func_cls = getattr(cec2020, cls_name, None)
        if func_cls is None:
            continue
        problem_cls = make_cec_class(func_cls, fid, "2020", _DEFAULT_DIM)
        register_problem(f"cec2020_f{fid}")(problem_cls)
