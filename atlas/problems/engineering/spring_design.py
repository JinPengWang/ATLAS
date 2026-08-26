"""Tension/Compression Spring Design Problem (Belegundu, 1982; Arora, 1989)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_problem


@register_problem("spring_design")
class SpringDesign(BaseProblem):
    """Tension/Compression spring design problem.

    Minimises the weight of a tension/compression spring subject to constraints
    on minimum deflection, shear stress, surge frequency, and limits on outer diameter.

    Decision variables:
        x[0]: Wire diameter (d)
        x[1]: Mean coil diameter (D)
        x[2]: Number of active coils (N)
    """

    def __init__(self, penalty_weight: float = 1e6, **kwargs) -> None:
        super().__init__(dim=3, boundary_strategy="clip", penalty_weight=penalty_weight, **kwargs)
        self._lb = np.array([0.05, 0.25, 2.0], dtype=float)
        self._ub = np.array([2.0, 1.30, 15.0], dtype=float)

    def evaluate(self, x: np.ndarray) -> float:
        x0, x1, x2 = x[0], x[1], x[2]
        return float((x2 + 2.0) * x1 * (x0**2))

    def get_constraints(self, x: np.ndarray) -> np.ndarray:
        x_c = self.clamp(x)
        d, D, N = x_c[0], x_c[1], x_c[2]

        g = np.zeros(4, dtype=float)
        g[0] = 1.0 - ((D**3) * N) / (71785.0 * (d**4) + 1e-12)
        g[1] = (
            (4.0 * (D**2) - d * D) / (12566.0 * (D * (d**3) - d**4) + 1e-12)
            + 1.0 / (5108.0 * (d**2) + 1e-12)
            - 1.0
        )
        g[2] = 1.0 - (140.45 * d) / ((D**2) * N + 1e-12)
        g[3] = (d + D) / 1.5 - 1.0
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
        return "spring_design"

    def get_optimum(self) -> Optional[float]:
        return 0.012665

    def get_optimum_location(self) -> Optional[np.ndarray]:
        return np.array([0.051690, 0.356718, 11.28885])
