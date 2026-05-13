"""CEC 2019 benchmark suite registration (F1-F10).

Note: CEC 2019 functions have **fixed** dimensions per function.
      F1=9, F2=16, F3=18, F4-F10=10 (scalable within a limited range).
"""

from __future__ import annotations

from atlas.problems.cec.base_cec import CECProblem
from atlas.utils.registry import register_problem

# Fixed default dimensions for each CEC 2019 function
_CEC2019_DEFAULT_DIMS = {
    1: 9,
    2: 16,
    3: 18,
    4: 10,
    5: 10,
    6: 10,
    7: 10,
    8: 10,
    9: 10,
    10: 10,
}


def register_cec2019() -> None:
    """Register all CEC 2019 benchmark functions."""
    import opfunu.cec_based.cec2019 as cec2019

    for fid, default_dim in _CEC2019_DEFAULT_DIMS.items():
        cls_name = f"F{fid}2019"
        func_cls = getattr(cec2019, cls_name, None)
        if func_cls is None:
            continue

        # Create a class with the correct default dimension
        def _make_cls(fc, fn, dd):
            class _CEC2019Func(CECProblem):
                def __init__(self, dim: int = dd, **kwargs) -> None:
                    func_obj = fc(ndim=dim)
                    super().__init__(func_obj, **kwargs)

                def get_name(self) -> str:
                    return f"cec2019_f{fn}"

            _CEC2019Func.__name__ = f"CEC2019F{fn}"
            _CEC2019Func.__qualname__ = _CEC2019Func.__name__
            return _CEC2019Func

        problem_cls = _make_cls(func_cls, fid, default_dim)
        register_problem(f"cec2019_f{fid}")(problem_cls)
