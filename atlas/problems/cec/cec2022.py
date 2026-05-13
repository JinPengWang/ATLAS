"""CEC 2022 benchmark suite registration (F1-F12)."""

from __future__ import annotations

from atlas.problems.cec.base_cec import make_cec_class
from atlas.utils.registry import register_problem

_CEC2022_FUNCTIONS = list(range(1, 13))  # F1-F12
_DEFAULT_DIM = 10


def register_cec2022() -> None:
    """Register all CEC 2022 benchmark functions."""
    import opfunu.cec_based.cec2022 as cec2022

    for fid in _CEC2022_FUNCTIONS:
        cls_name = f"F{fid}2022"
        func_cls = getattr(cec2022, cls_name, None)
        if func_cls is None:
            continue
        problem_cls = make_cec_class(func_cls, fid, "2022", _DEFAULT_DIM)
        register_problem(f"cec2022_f{fid}")(problem_cls)
