"""Pressure Vessel Design Problem (Kannan & Kramer, 1994)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_problem


@register_problem("pressure_vessel_eng")
class PressureVesselEng(BaseProblem):
    """Pressure vessel design engineering benchmark problem.

    Minimises the total manufacturing cost of a cylindrical pressure vessel
    with hemispherical heads.

    Decision variables:
        x[0]: Shell thickness (Ts)
        x[1]: Head thickness (Th)
        x[2]: Inner radius (R)
        x[3]: Length of cylindrical section (L)
    """

    def __init__(self, penalty_weight: float = 1e6, **kwargs) -> None:
        super().__init__(dim=4, boundary_strategy="clip", penalty_weight=penalty_weight, **kwargs)
        self._lb = np.array([0.0625, 0.0625, 10.0, 10.0], dtype=float)
        self._ub = np.array([6.1875, 6.1875, 200.0, 200.0], dtype=float)

    def evaluate(self, x: np.ndarray) -> float:
        x0, x1, x2, x3 = x[0], x[1], x[2], x[3]
        return float(
            0.6224 * x0 * x2 * x3
            + 1.7781 * x1 * (x2**2)
            + 3.1661 * (x0**2) * x3
            + 19.84 * (x0**2) * x2
        )

    def get_constraints(self, x: np.ndarray) -> np.ndarray:
        x_c = self.clamp(x)
        x0, x1, x2, x3 = x_c[0], x_c[1], x_c[2], x_c[3]

        g = np.zeros(4, dtype=float)
        g[0] = -x0 + 0.0193 * x2
        g[1] = -x1 + 0.00954 * x2
        g[2] = -np.pi * (x2**2) * x3 - (4.0 / 3.0) * np.pi * (x2**3) + 1296000.0
        g[3] = x3 - 240.0
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
        return "pressure_vessel_eng"

    def get_optimum(self) -> Optional[float]:
        return 6059.714335

    def get_optimum_location(self) -> Optional[np.ndarray]:
        return np.array([0.8125, 0.4375, 42.098446, 176.636596])
