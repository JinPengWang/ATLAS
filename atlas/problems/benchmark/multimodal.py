"""Multimodal benchmark functions.

These functions have many local minima, making them challenging for
optimisation algorithms.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_problem


# =====================================================================
# Rastrigin
# =====================================================================
@register_problem("rastrigin")
class Rastrigin(BaseProblem):
    """Rastrigin function.

    f(x) = 10d + Σ [x_i² - 10 cos(2π x_i)]

    * Type: Multimodal, separable
    * Default dim: 30
    * Bounds: [-5.12, 5.12]^d
    * Global optimum: f(x*) = 0 at x* = (0, …, 0)
    """

    def __init__(self, dim: int = 30, **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        self._lb = np.full(dim, -5.12)
        self._ub = np.full(dim, 5.12)

    def evaluate(self, x: np.ndarray) -> float:
        return float(10.0 * self.dim + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lb.copy(), self._ub.copy()

    def get_name(self) -> str:
        return "rastrigin"

    def get_optimum(self) -> Optional[float]:
        return 0.0

    def get_optimum_location(self) -> Optional[np.ndarray]:
        return np.zeros(self.dim)


# =====================================================================
# Ackley
# =====================================================================
@register_problem("ackley")
class Ackley(BaseProblem):
    """Ackley function.

    f(x) = -20 exp(-0.2 sqrt(1/d Σ x_i²)) - exp(1/d Σ cos(2π x_i)) + 20 + e

    * Type: Multimodal, non-separable
    * Default dim: 30
    * Bounds: [-32.768, 32.768]^d
    * Global optimum: f(x*) = 0 at x* = (0, …, 0)
    """

    def __init__(self, dim: int = 30, **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        self._lb = np.full(dim, -32.768)
        self._ub = np.full(dim, 32.768)

    def evaluate(self, x: np.ndarray) -> float:
        d = self.dim
        sum_sq = np.sum(x ** 2)
        sum_cos = np.sum(np.cos(2.0 * np.pi * x))
        return float(
            -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / d))
            - np.exp(sum_cos / d)
            + 20.0
            + np.e
        )

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lb.copy(), self._ub.copy()

    def get_name(self) -> str:
        return "ackley"

    def get_optimum(self) -> Optional[float]:
        return 0.0

    def get_optimum_location(self) -> Optional[np.ndarray]:
        return np.zeros(self.dim)


# =====================================================================
# Griewank
# =====================================================================
@register_problem("griewank")
class Griewank(BaseProblem):
    """Griewank function.

    f(x) = 1 + Σ x_i² / 4000 - Π cos(x_i / sqrt(i))

    * Type: Multimodal, non-separable
    * Default dim: 30
    * Bounds: [-600, 600]^d
    * Global optimum: f(x*) = 0 at x* = (0, …, 0)
    """

    def __init__(self, dim: int = 30, **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        self._lb = np.full(dim, -600.0)
        self._ub = np.full(dim, 600.0)

    def evaluate(self, x: np.ndarray) -> float:
        idx = np.arange(1, self.dim + 1, dtype=float)
        sum_sq = np.sum(x ** 2 / 4000.0)
        prod_cos = np.prod(np.cos(x / np.sqrt(idx)))
        return float(1.0 + sum_sq - prod_cos)

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lb.copy(), self._ub.copy()

    def get_name(self) -> str:
        return "griewank"

    def get_optimum(self) -> Optional[float]:
        return 0.0

    def get_optimum_location(self) -> Optional[np.ndarray]:
        return np.zeros(self.dim)


# =====================================================================
# Levy
# =====================================================================
@register_problem("levy")
class Levy(BaseProblem):
    """Levy function.

    f(x) = sin²(π w_1) + Σ_{i=1}^{d-1} (w_i - 1)² [1 + 10 sin²(π w_i + 1)]
           + (w_d - 1)² [1 + sin²(2π w_d)]
    where w_i = 1 + (x_i - 1) / 4

    * Type: Multimodal, non-separable
    * Default dim: 30
    * Bounds: [-10, 10]^d
    * Global optimum: f(x*) = 0 at x* = (1, 1, …, 1)
    """

    def __init__(self, dim: int = 30, **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        self._lb = np.full(dim, -10.0)
        self._ub = np.full(dim, 10.0)

    def evaluate(self, x: np.ndarray) -> float:
        w = 1.0 + (x - 1.0) / 4.0
        term1 = np.sin(np.pi * w[0]) ** 2
        term_mid = np.sum(
            (w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2)
        )
        term_last = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
        return float(term1 + term_mid + term_last)

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lb.copy(), self._ub.copy()

    def get_name(self) -> str:
        return "levy"

    def get_optimum(self) -> Optional[float]:
        return 0.0

    def get_optimum_location(self) -> Optional[np.ndarray]:
        return np.ones(self.dim)


# =====================================================================
# Schwefel
# =====================================================================
@register_problem("schwefel")
class Schwefel(BaseProblem):
    """Schwefel function.

    f(x) = 418.9829d - Σ x_i sin(sqrt(|x_i|))

    * Type: Multimodal, separable
    * Default dim: 30
    * Bounds: [-500, 500]^d
    * Global optimum: f(x*) = 0 at x* = (420.9687, …, 420.9687)
    """

    def __init__(self, dim: int = 30, **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        self._lb = np.full(dim, -500.0)
        self._ub = np.full(dim, 500.0)

    def evaluate(self, x: np.ndarray) -> float:
        return float(418.9829 * self.dim - np.sum(x * np.sin(np.sqrt(np.abs(x)))))

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lb.copy(), self._ub.copy()

    def get_name(self) -> str:
        return "schwefel"

    def get_optimum(self) -> Optional[float]:
        return 0.0

    def get_optimum_location(self) -> Optional[np.ndarray]:
        return np.full(self.dim, 420.9687)


# =====================================================================
# Michalewicz
# =====================================================================
@register_problem("michalewicz")
class Michalewicz(BaseProblem):
    """Michalewicz function.

    f(x) = -Σ sin(x_i) [sin(i x_i² / π)]^{2m}    (m = 10)

    * Type: Multimodal, separable
    * Default dim: 30
    * Bounds: [0, π]^d
    * Global optimum: varies with d (for d=2, f* ≈ -1.8013)
    """

    def __init__(self, dim: int = 30, m: float = 10.0, **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        self.m = m
        self._lb = np.zeros(dim)
        self._ub = np.full(dim, np.pi)

    def evaluate(self, x: np.ndarray) -> float:
        idx = np.arange(1, self.dim + 1, dtype=float)
        return float(-np.sum(np.sin(x) * (np.sin(idx * x ** 2 / np.pi)) ** (2 * self.m)))

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lb.copy(), self._ub.copy()

    def get_name(self) -> str:
        return "michalewicz"

    def get_optimum(self) -> Optional[float]:
        if self.dim == 2:
            return -1.8013
        return None
