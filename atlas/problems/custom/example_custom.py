"""Example custom optimisation problem: Pressure Vessel Design.

This module demonstrates how to define an engineering design problem
and register it with ATLAS.  You can use this as a template for your
own custom problems.

Problem description (minimisation):
    Minimise the total cost of a pressure vessel including material,
    forming, and welding costs.

Decision variables:
    x[0] = thickness of shell          (T_s)
    x[1] = thickness of head           (T_h)
    x[2] = inner radius                (R)
    x[3] = length of cylindrical section (L)

All variables are continuous.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_problem


@register_problem("pressure_vessel")
class PressureVessel(BaseProblem):
    """Pressure vessel design problem (Kannan & Kramer, 1994).

    Objective::

        minimise  f(x) = 0.6224 x0 x2 x3
                          + 1.7781 x1 x2²
                          + 3.1661 x0² x3
                          + 19.84 x0² x2

    Subject to::

        g1(x) = -x0 + 0.0193 x2 <= 0
        g2(x) = -x1 + 0.00954 x2 <= 0
        g3(x) = -π x2² x3 - 4/3 π x3³ + 1296000 <= 0
        g4(x) = x3 - 240 <= 0

    Bounds::

        0.0625 <= x0, x1 <= 6.1875   (multiples of 0.0625 in original)
        10     <= x2 <= 200
        10     <= x3 <= 200

    * Global optimum: f* ≈ 6059.714335
    * x* ≈ [0.8125, 0.4375, 42.0984, 176.6366]

    Reference:
        Kannan, B. and Kramer, S. N. (1994). An augmented Lagrange multiplier
        based method for mixed integer discrete continuous optimization and its
        applications to mechanical design. *Journal of Mechanical Design*,
        116(2): 405–411.

    Args:
        penalty_weight: Penalty coefficient for constraint violations.
    """

    def __init__(self, penalty_weight: float = 1e6, **kwargs) -> None:
        super().__init__(dim=4, boundary_strategy="clip", penalty_weight=penalty_weight, **kwargs)
        self._lb = np.array([0.0625, 0.0625, 10.0, 10.0])
        self._ub = np.array([6.1875, 6.1875, 200.0, 200.0])

    def evaluate(self, x: np.ndarray) -> float:
        x0, x1, x2, x3 = x[0], x[1], x[2], x[3]
        cost = (
            0.6224 * x0 * x2 * x3
            + 1.7781 * x1 * x2 ** 2
            + 3.1661 * x0 ** 2 * x3
            + 19.84 * x0 ** 2 * x2
        )
        return cost

    def evaluate_with_penalty(self, x: np.ndarray) -> float:
        """Evaluate with inequality constraints handled via penalty."""
        x_c = self.clamp(x)
        base = self.evaluate(x_c)

        # Constraints (g <= 0)
        g1 = -x_c[0] + 0.0193 * x_c[2]
        g2 = -x_c[1] + 0.00954 * x_c[2]
        g3 = -np.pi * x_c[2] ** 2 * x_c[3] - (4.0 / 3.0) * np.pi * x_c[3] ** 3 + 1_296_000.0
        g4 = x_c[3] - 240.0

        violation = max(0, g1) + max(0, g2) + max(0, g3) + max(0, g4)
        return base + self.penalty_weight * violation

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lb.copy(), self._ub.copy()

    def get_name(self) -> str:
        return "pressure_vessel"

    def get_optimum(self) -> Optional[float]:
        return 6059.714335

    def get_optimum_location(self) -> Optional[np.ndarray]:
        return np.array([0.8125, 0.4375, 42.0984, 176.6366])
