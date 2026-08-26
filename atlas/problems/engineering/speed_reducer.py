"""Speed Reducer Design Problem (Golinski, 1970; Ray & Liew, 2003)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_problem


@register_problem("speed_reducer")
class SpeedReducer(BaseProblem):
    """Speed reducer design benchmark problem.

    Minimises the total weight of a gearbox / speed reducer subject to
    constraints on bending stress of the gear teeth, surface stress, transverse
    deflections of the shafts, and stresses in the shafts.

    Decision variables:
        x[0]: Face width (b)
        x[1]: Module of teeth (m)
        x[2]: Number of teeth in the pinion (z)
        x[3]: Length of first shaft between bearings (l1)
        x[4]: Length of second shaft between bearings (l2)
        x[5]: Diameter of first shaft (d1)
        x[6]: Diameter of second shaft (d2)
    """

    def __init__(self, penalty_weight: float = 1e6, **kwargs) -> None:
        super().__init__(dim=7, boundary_strategy="clip", penalty_weight=penalty_weight, **kwargs)
        self._lb = np.array([2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0], dtype=float)
        self._ub = np.array([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5], dtype=float)

    def evaluate(self, x: np.ndarray) -> float:
        x0, x1, x2, x3, x4, x5, x6 = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
        return float(
            0.7854 * x0 * (x1**2) * (3.3333 * (x2**2) + 14.9334 * x2 - 43.0934)
            - 1.508 * x0 * (x5**2 + x6**2)
            + 7.4777 * (x5**3 + x6**3)
            + 0.7854 * (x3 * (x5**2) + x4 * (x6**2))
        )

    def get_constraints(self, x: np.ndarray) -> np.ndarray:
        x_c = self.clamp(x)
        x0, x1, x2, x3, x4, x5, x6 = (
            x_c[0],
            x_c[1],
            x_c[2],
            x_c[3],
            x_c[4],
            x_c[5],
            x_c[6],
        )

        g = np.zeros(11, dtype=float)
        g[0] = 27.0 / (x0 * (x1**2) * x2 + 1e-12) - 1.0
        g[1] = 397.5 / (x0 * (x1**2) * (x2**2) + 1e-12) - 1.0
        g[2] = 1.93 * (x3**3) / (x1 * x2 * (x5**4) + 1e-12) - 1.0
        g[3] = 1.93 * (x4**3) / (x1 * x2 * (x6**4) + 1e-12) - 1.0
        g[4] = (
            np.sqrt((745.0 * x3 / (x1 * x2 + 1e-12)) ** 2 + 16.9e6)
            / (110.0 * (x5**3) + 1e-12)
            - 1.0
        )
        g[5] = (
            np.sqrt((745.0 * x4 / (x1 * x2 + 1e-12)) ** 2 + 157.5e6)
            / (85.0 * (x6**3) + 1e-12)
            - 1.0
        )
        g[6] = x1 * x2 / 40.0 - 1.0
        g[7] = 5.0 * x1 / (x0 + 1e-12) - 1.0
        g[8] = x0 / (12.0 * x1 + 1e-12) - 1.0
        g[9] = (1.5 * x5 + 1.9) / (x3 + 1e-12) - 1.0
        g[10] = (1.1 * x6 + 1.9) / (x4 + 1e-12) - 1.0
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
        return "speed_reducer"

    def get_optimum(self) -> Optional[float]:
        return 2994.471066

    def get_optimum_location(self) -> Optional[np.ndarray]:
        return np.array([3.500000, 0.700000, 17.0, 7.300000, 7.800000, 3.350214, 5.286683])
