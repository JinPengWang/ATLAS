"""Generic wrapper adapting ``opfunu`` CEC functions to ``BaseProblem``."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from atlas.core.base_problem import BaseProblem


class CECProblem(BaseProblem):
    """Wraps any ``opfunu`` function object as a :class:`BaseProblem`.

    Args:
        func_obj: An instantiated ``opfunu`` function object (e.g.
            ``opfunu.cec_based.cec2017.F12017(ndim=10)``).
        **kwargs: Forwarded to :class:`BaseProblem`.
    """

    def __init__(self, func_obj, **kwargs) -> None:
        self._func = func_obj
        super().__init__(dim=func_obj.ndim, **kwargs)

    def evaluate(self, x: np.ndarray) -> float:
        val = self._func.evaluate(x)
        if not np.isfinite(val):
            return 1e300
        return float(val)

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self._func.lb, dtype=float), np.array(self._func.ub, dtype=float)

    def get_name(self) -> str:
        return self._func.name

    def get_optimum(self) -> Optional[float]:
        return float(self._func.f_bias) if hasattr(self._func, "f_bias") else None


def make_cec_class(func_cls, func_id: int, suite_name: str, default_dim: int = 10):
    """Dynamically create a :class:`BaseProblem` subclass for one CEC function.

    Args:
        func_cls: The ``opfunu`` function **class** (e.g. ``F12017``).
        func_id: Function number within the suite (e.g. 1).
        suite_name: Suite identifier (e.g. ``"2017"``).
        default_dim: Default dimensionality.

    Returns:
        A new class suitable for registration.
    """

    class _CECFunc(CECProblem):
        def __init__(self, dim: int = default_dim, **kwargs) -> None:
            # Validate dimension by checking opfunu's supported dimensions
            _tmp = func_cls()
            if hasattr(_tmp, 'dim_supported') and _tmp.dim_supported is not None:
                if dim not in _tmp.dim_supported:
                    raise ValueError(
                        f"cec{suite_name}_f{func_id} does not support dim={dim}. "
                        f"Supported dimensions: {_tmp.dim_supported}"
                    )
            elif hasattr(_tmp, 'dim_max') and dim > _tmp.dim_max:
                raise ValueError(
                    f"cec{suite_name}_f{func_id} maximum dimension is {_tmp.dim_max}, "
                    f"got dim={dim}"
                )
            func_obj = func_cls(ndim=dim)
            super().__init__(func_obj, **kwargs)

        def get_name(self) -> str:
            return f"cec{suite_name}_f{func_id}"

    _CECFunc.__name__ = f"CEC{suite_name}F{func_id}"
    _CECFunc.__qualname__ = _CECFunc.__name__
    return _CECFunc
