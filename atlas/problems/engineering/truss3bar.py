"""Three-Bar Truss Design Problem (Nowacki, 1974; Ray & Saini, 2001)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_problem


@register_problem("truss_3bar")
class Truss3Bar(BaseProblem):
    """Three-bar truss design problem.

    Minimises the structural volume / weight of a 3-bar planar truss structure
    subject to stress constraints on each of the members.

    Decision variables:
        x[0]: Cross-sectional area of outer bars 1 and 3 (A1 = A3)
        x[1]: Cross-sectional area of central bar 2 (A2)
    """

    def __init__(self, penalty_weight: float = 1e6, **kwargs) -> None:
        super().__init__(dim=2, boundary_strategy="clip", penalty_weight=penalty_weight, **kwargs)
        self._lb = np.array([0.001, 0.001], dtype=float)
        self._ub = np.array([1.0, 1.0], dtype=float)

    def evaluate(self, x: np.ndarray) -> float:
        x0, x1 = x[0], x[1]
        L = 100.0  # length in cm
        return float((2.0 * np.sqrt(2.0) * x0 + x1) * L)

    def get_constraints(self, x: np.ndarray) -> np.ndarray:
        x_c = self.clamp(x)
        x0, x1 = x_c[0], x_c[1]

        P = 2.0  # load in KN
        sigma = 2.0  # max allowable stress in KN/cm^2

        denom = np.sqrt(2.0) * (x0**2) + 2.0 * x0 * x1 + 1e-12

        g = np.zeros(3, dtype=float)
        g[0] = (np.sqrt(2.0) * x0 + x1) / denom * P - sigma
        g[1] = x1 / denom * P - sigma
        g[2] = 1.0 / (x0 + np.sqrt(2.0) * x1 + 1e-12) * P - sigma
        return g

    def evaluate_with_penalty(self, x: np.ndarray) -> float:
        x_c = self.clamp(x)
        cost = self.evaluate(x_c)
        g = self.get_constraints(x_c)
        violations = np.maximum(g, 0.0)
        return float(cost + self.penalty_weight * np.sum(violations**2))

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lb.copy(), self._ub.copy()

    def get_name(self) -> str:
        return "truss_3bar"

    def get_optimum(self) -> Optional[float]:
        return 263.8958434

    def get_optimum_location(self) -> Optional[np.ndarray]:
        return np.array([0.78867513, 0.40824829])
