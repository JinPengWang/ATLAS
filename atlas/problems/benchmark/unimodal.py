"""Unimodal benchmark functions.

These functions have a single global optimum and no local minima, making
them ideal for testing convergence speed.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_problem


# =====================================================================
# Sphere
# =====================================================================
@register_problem("sphere")
class Sphere(BaseProblem):
    """Sphere function.

    f(x) = Σ x_i²

    * Type: Unimodal, separable
    * Default dim: 30
    * Bounds: [-100, 100]^d
    * Global optimum: f(x*) = 0 at x* = (0, …, 0)
    """

    def __init__(self, dim: int = 30, **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        self._lb = np.full(dim, -100.0)
        self._ub = np.full(dim, 100.0)

    def evaluate(self, x: np.ndarray) -> float:
        return float(np.sum(x ** 2))

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lb.copy(), self._ub.copy()

    def get_name(self) -> str:
        return "sphere"

    def get_optimum(self) -> Optional[float]:
        return 0.0

    def get_optimum_location(self) -> Optional[np.ndarray]:
        return np.zeros(self.dim)


# =====================================================================
# Rosenbrock
# =====================================================================
@register_problem("rosenbrock")
class Rosenbrock(BaseProblem):
    """Rosenbrock (banana) function.

    f(x) = Σ_{i=1}^{d-1} [100(x_{i+1} - x_i²)² + (1 - x_i)²]

    * Type: Unimodal (for low d), non-separable
    * Default dim: 30
    * Bounds: [-30, 30]^d  (sometimes [-5, 10] or [-2.048, 2.048])
    * Global optimum: f(x*) = 0 at x* = (1, 1, …, 1)
    """

    def __init__(self, dim: int = 30, **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        self._lb = np.full(dim, -30.0)
        self._ub = np.full(dim, 30.0)

    def evaluate(self, x: np.ndarray) -> float:
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lb.copy(), self._ub.copy()

    def get_name(self) -> str:
        return "rosenbrock"

    def get_optimum(self) -> Optional[float]:
        return 0.0

    def get_optimum_location(self) -> Optional[np.ndarray]:
        return np.ones(self.dim)


# =====================================================================
# Schwefel 2.22
# =====================================================================
@register_problem("schwefel222")
class Schwefel222(BaseProblem):
    """Schwefel 2.22 function.

    f(x) = Σ |x_i| + Π |x_i|

    * Type: Unimodal, separable
    * Default dim: 30
    * Bounds: [-10, 10]^d
    * Global optimum: f(x*) = 0 at x* = (0, …, 0)
    """

    def __init__(self, dim: int = 30, **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        self._lb = np.full(dim, -10.0)
        self._ub = np.full(dim, 10.0)

    def evaluate(self, x: np.ndarray) -> float:
        abs_x = np.abs(x)
        return float(np.sum(abs_x) + np.prod(abs_x))

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lb.copy(), self._ub.copy()

    def get_name(self) -> str:
        return "schwefel222"

    def get_optimum(self) -> Optional[float]:
        return 0.0

    def get_optimum_location(self) -> Optional[np.ndarray]:
        return np.zeros(self.dim)


# =====================================================================
# Quartic (with noise)
# =====================================================================
@register_problem("quartic")
class Quartic(BaseProblem):
    """Quartic function with Gaussian noise.

    f(x) = Σ i · x_i⁴ + random_noise

    * Type: Unimodal, separable, noisy
    * Default dim: 30
    * Bounds: [-1.28, 1.28]^d
    * Global optimum: f(x*) ≈ 0 at x* ≈ (0, …, 0)
    """

    def __init__(self, dim: int = 30, **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        self._lb = np.full(dim, -1.28)
        self._ub = np.full(dim, 1.28)

    def evaluate(self, x: np.ndarray) -> float:
        idx = np.arange(1, self.dim + 1, dtype=float)
        return float(np.sum(idx * x ** 4))

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lb.copy(), self._ub.copy()

    def get_name(self) -> str:
        return "quartic"

    def get_optimum(self) -> Optional[float]:
        return 0.0

    def get_optimum_location(self) -> Optional[np.ndarray]:
        return np.zeros(self.dim)
