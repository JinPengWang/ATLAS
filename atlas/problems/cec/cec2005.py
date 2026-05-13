"""CEC 2005 benchmark suite registration (F1-F25)."""

from __future__ import annotations

from atlas.problems.cec.base_cec import make_cec_class
from atlas.utils.registry import register_problem

_CEC2005_FUNCTIONS = list(range(1, 26))  # F1-F25
_DEFAULT_DIM = 30  # supports 10, 30, 50


def register_cec2005() -> None:
    """Register all CEC 2005 benchmark functions."""
    import opfunu.cec_based.cec2005 as cec2005

    for fid in _CEC2005_FUNCTIONS:
        cls_name = f"F{fid}2005"
        func_cls = getattr(cec2005, cls_name, None)
        if func_cls is None:
            continue
        problem_cls = make_cec_class(func_cls, fid, "2005", _DEFAULT_DIM)
        register_problem(f"cec2005_f{fid}")(problem_cls)
