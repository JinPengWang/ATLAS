"""CEC 2014 benchmark suite registration (F1-F30)."""

from __future__ import annotations

from atlas.problems.cec.base_cec import make_cec_class
from atlas.utils.registry import register_problem

_CEC2014_FUNCTIONS = list(range(1, 31))  # F1-F30
_DEFAULT_DIM = 30  # supports 10, 20, 30, 50, 100


def register_cec2014() -> None:
    """Register all CEC 2014 benchmark functions."""
    import opfunu.cec_based.cec2014 as cec2014

    for fid in _CEC2014_FUNCTIONS:
        cls_name = f"F{fid}2014"
        func_cls = getattr(cec2014, cls_name, None)
        if func_cls is None:
            continue
        problem_cls = make_cec_class(func_cls, fid, "2014", _DEFAULT_DIM)
        register_problem(f"cec2014_f{fid}")(problem_cls)
