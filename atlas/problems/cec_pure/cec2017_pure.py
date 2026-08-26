"""Pure-Python vectorized implementation of CEC 2017 benchmark functions.

Zero external C compiler dependency. Deterministic shifts and orthogonal rotation
transformations are generated via reproducible QR factorizations.

Search space: [-100, 100]^D (supports arbitrary dimensions, typically D=10, 30, 50, 100).
Optimum fitness: f*(x*) = 100 * func_id (bias).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from atlas.core.base_problem import BaseProblem
from atlas.problems.cec_pure.cec_transformations import (
    get_deterministic_shift_and_rotation,
    transform_x,
)
from atlas.utils.registry import register_problem


class PureCEC2017Problem(BaseProblem):
    """Base class for pure-Python CEC 2017 benchmark functions."""

    def __init__(
        self,
        func_id: int,
        dim: int = 10,
        boundary_strategy: str = "clip",
        penalty_weight: float = 1e6,
        **kwargs,
    ) -> None:
        super().__init__(dim=dim, boundary_strategy=boundary_strategy, penalty_weight=penalty_weight, **kwargs)
        self.func_id = func_id
        self.bias = float(func_id * 100.0)
        self._lb = np.full(dim, -100.0, dtype=float)
        self._ub = np.full(dim, 100.0, dtype=float)
        self.shift, self.M = get_deterministic_shift_and_rotation(func_id, dim)

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lb.copy(), self._ub.copy()

    def get_optimum(self) -> Optional[float]:
        return self.bias

    def get_optimum_location(self) -> Optional[np.ndarray]:
        return self.shift.copy()

    def get_name(self) -> str:
        return f"cec2017_f{self.func_id}_pure"


# ---------------------------------------------------------------------------
# F1: Shifted and Rotated Bent Cigar Function
# ---------------------------------------------------------------------------
@register_problem("cec2017_f1_pure", aliases=["cec2017_f1"])
class CEC2017F1(PureCEC2017Problem):
    """F1: Shifted and Rotated Bent Cigar Function (Unimodal)."""

    def __init__(self, dim: int = 10, **kwargs) -> None:
        super().__init__(func_id=1, dim=dim, **kwargs)

    def evaluate(self, x: np.ndarray) -> float:
        z = transform_x(x, self.shift, self.M)
        val = z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2)
        return float(val + self.bias)


# ---------------------------------------------------------------------------
# F3: Shifted and Rotated Rosenbrock's Function
# ---------------------------------------------------------------------------
@register_problem("cec2017_f3_pure", aliases=["cec2017_f3"])
class CEC2017F3(PureCEC2017Problem):
    """F3: Shifted and Rotated Rosenbrock's Function (Unimodal)."""

    def __init__(self, dim: int = 10, **kwargs) -> None:
        super().__init__(func_id=3, dim=dim, **kwargs)

    def evaluate(self, x: np.ndarray) -> float:
        # Scale to standard domain [-2.048, 2.048]
        z = transform_x(x, self.shift, self.M) * 0.02048 + 1.0
        val = np.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (z[:-1] - 1.0) ** 2)
        return float(val + self.bias)


# ---------------------------------------------------------------------------
# F4: Shifted and Rotated Rastrigin's Function
# ---------------------------------------------------------------------------
@register_problem("cec2017_f4_pure", aliases=["cec2017_f4"])
class CEC2017F4(PureCEC2017Problem):
    """F4: Shifted and Rotated Rastrigin's Function (Multimodal)."""

    def __init__(self, dim: int = 10, **kwargs) -> None:
        super().__init__(func_id=4, dim=dim, **kwargs)

    def evaluate(self, x: np.ndarray) -> float:
        z = transform_x(x, self.shift, self.M) * 0.0512  # scale to [-5.12, 5.12]
        val = np.sum(z**2 - 10.0 * np.cos(2.0 * np.pi * z) + 10.0)
        return float(val + self.bias)


# ---------------------------------------------------------------------------
# F5: Shifted and Rotated Expanded Griewank's plus Rosenbrock's Function
# ---------------------------------------------------------------------------
@register_problem("cec2017_f5_pure", aliases=["cec2017_f5"])
class CEC2017F5(PureCEC2017Problem):
    """F5: Shifted and Rotated Expanded Griewank's plus Rosenbrock's Function."""

    def __init__(self, dim: int = 10, **kwargs) -> None:
        super().__init__(func_id=5, dim=dim, **kwargs)

    def evaluate(self, x: np.ndarray) -> float:
        z = transform_x(x, self.shift, self.M) * 0.05 + 1.0
        # Expanded Rosenbrock
        ros = 100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (z[:-1] - 1.0) ** 2
        # Wrap with Griewank
        val = np.sum(ros**2 / 4000.0 - np.cos(ros) + 1.0)
        return float(val + self.bias)


# ---------------------------------------------------------------------------
# F6: Shifted and Rotated Ackley's Function
# ---------------------------------------------------------------------------
@register_problem("cec2017_f6_pure", aliases=["cec2017_f6"])
class CEC2017F6(PureCEC2017Problem):
    """F6: Shifted and Rotated Ackley's Function (Multimodal)."""

    def __init__(self, dim: int = 10, **kwargs) -> None:
        super().__init__(func_id=6, dim=dim, **kwargs)

    def evaluate(self, x: np.ndarray) -> float:
        z = transform_x(x, self.shift, self.M) * 0.32  # scale to [-32, 32]
        d = len(z)
        term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(z**2) / d))
        term2 = -np.exp(np.sum(np.cos(2.0 * np.pi * z)) / d)
        val = term1 + term2 + 20.0 + np.e
        return float(val + self.bias)


# ---------------------------------------------------------------------------
# F7: Shifted and Rotated Modified Schwefel's Function
# ---------------------------------------------------------------------------
@register_problem("cec2017_f7_pure", aliases=["cec2017_f7"])
class CEC2017F7(PureCEC2017Problem):
    """F7: Shifted and Rotated Modified Schwefel's Function (Multimodal)."""

    def __init__(self, dim: int = 10, **kwargs) -> None:
        super().__init__(func_id=7, dim=dim, **kwargs)

    def evaluate(self, x: np.ndarray) -> float:
        z = transform_x(x, self.shift, self.M) * 10.0 + 420.9687462275036  # scale to [-500, 500]
        val = 418.9828872724338 * len(z) - np.sum(z * np.sin(np.sqrt(np.abs(z))))
        return float(val + self.bias)


# ---------------------------------------------------------------------------
# F8: Shifted and Rotated Griewank's Function
# ---------------------------------------------------------------------------
@register_problem("cec2017_f8_pure", aliases=["cec2017_f8"])
class CEC2017F8(PureCEC2017Problem):
    """F8: Shifted and Rotated Griewank's Function (Multimodal)."""

    def __init__(self, dim: int = 10, **kwargs) -> None:
        super().__init__(func_id=8, dim=dim, **kwargs)

    def evaluate(self, x: np.ndarray) -> float:
        z = transform_x(x, self.shift, self.M) * 6.0  # scale to [-600, 600]
        i = np.arange(1, len(z) + 1)
        val = np.sum(z**2) / 4000.0 - np.prod(np.cos(z / np.sqrt(i))) + 1.0
        return float(val + self.bias)


# ---------------------------------------------------------------------------
# F9: Shifted and Rotated Weierstrass Function
# ---------------------------------------------------------------------------
@register_problem("cec2017_f9_pure", aliases=["cec2017_f9"])
class CEC2017F9(PureCEC2017Problem):
    """F9: Shifted and Rotated Weierstrass Function (Multimodal)."""

    def __init__(self, dim: int = 10, **kwargs) -> None:
        super().__init__(func_id=9, dim=dim, **kwargs)
        self.a = 0.5
        self.b = 3.0
        self.k_max = 20
        k = np.arange(self.k_max + 1)
        self.ak = self.a**k
        self.bk = self.b**k

    def evaluate(self, x: np.ndarray) -> float:
        z = transform_x(x, self.shift, self.M) * 0.005  # scale to [-0.5, 0.5]
        # Sum over dimensions and frequencies
        val = 0.0
        for zi in z:
            val += np.sum(self.ak * np.cos(2.0 * np.pi * self.bk * (zi + 0.5)))
        c = len(z) * np.sum(self.ak * np.cos(2.0 * np.pi * self.bk * 0.5))
        return float(val - c + self.bias)


# ---------------------------------------------------------------------------
# F10: Shifted and Rotated Katsuura Function
# ---------------------------------------------------------------------------
@register_problem("cec2017_f10_pure", aliases=["cec2017_f10"])
class CEC2017F10(PureCEC2017Problem):
    """F10: Shifted and Rotated Katsuura Function (Multimodal)."""

    def __init__(self, dim: int = 10, **kwargs) -> None:
        super().__init__(func_id=10, dim=dim, **kwargs)

    def evaluate(self, x: np.ndarray) -> float:
        z = transform_x(x, self.shift, self.M) * 0.05  # scale to [-5, 5]
        d = len(z)
        prod = 1.0
        j_arr = np.arange(1, 33)
        for i, zi in enumerate(z):
            inner = np.sum(np.abs(2.0**j_arr * zi - np.round(2.0**j_arr * zi)) / (2.0**j_arr))
            term = 1.0 + (i + 1) * inner
            prod *= term ** (10.0 / (d**1.2))
        val = (10.0 / (d**2)) * prod - (10.0 / (d**2))
        return float(val + self.bias)
