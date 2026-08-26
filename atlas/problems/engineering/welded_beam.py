"""Welded Beam Design Problem (Deb, 2000; Coello, 2000)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_problem


@register_problem("welded_beam")
class WeldedBeam(BaseProblem):
    """Welded beam design benchmark problem.

    Minimises the manufacturing cost of a welded beam subjected to constraints
    on shear stress, bending stress, buckling load, end deflection, and geometry.

    Decision variables:
        x[0]: Weld thickness (h)
        x[1]: Length of welded joint (l)
        x[2]: Width of beam (t)
        x[3]: Thickness of beam (b)
    """

    def __init__(self, penalty_weight: float = 1e6, **kwargs) -> None:
        super().__init__(dim=4, boundary_strategy="clip", penalty_weight=penalty_weight, **kwargs)
        self._lb = np.array([0.1, 0.1, 0.1, 0.1], dtype=float)
        self._ub = np.array([2.0, 10.0, 10.0, 2.0], dtype=float)

    def evaluate(self, x: np.ndarray) -> float:
        x0, x1, x2, x3 = x[0], x[1], x[2], x[3]
        return float(1.10471 * (x0**2) * x1 + 0.04811 * x2 * x3 * (14.0 + x1))

    def get_constraints(self, x: np.ndarray) -> np.ndarray:
        x_c = self.clamp(x)
        x0, x1, x2, x3 = x_c[0], x_c[1], x_c[2], x_c[3]

        P = 6000.0
        L = 14.0
        E = 30e6
        G = 12e6
        tau_max = 13600.0
        sigma_max = 30000.0
        delta_max = 0.25

        M = P * (L + x1 / 2.0)
        R = np.sqrt(x1**2 / 4.0 + ((x0 + x2) / 2.0) ** 2)
        J = 2.0 * (np.sqrt(2.0) * x0 * x1 * (x1**2 / 12.0 + ((x0 + x2) / 2.0) ** 2))
        tau_prime = P / (np.sqrt(2.0) * x0 * x1 + 1e-12)
        tau_double_prime = (M * R) / (J + 1e-12)
        tau = np.sqrt(
            tau_prime**2
            + (2.0 * tau_prime * tau_double_prime * x1) / (2.0 * R + 1e-12)
            + tau_double_prime**2
        )
        sigma = (6.0 * P * L) / (x3 * (x2**2) + 1e-12)
        delta = (4.0 * P * (L**3)) / (E * x3 * (x2**3) + 1e-12)
        P_c = (
            64746.022
            * (1.0 - 0.0282346 * x2)
            * x2
            * (x3**3)
        )

        g = np.zeros(7, dtype=float)
        g[0] = tau - tau_max
        g[1] = sigma - sigma_max
        g[2] = x0 - x3
        g[3] = 0.10471 * (x0**2) + 0.04811 * x2 * x3 * (14.0 + x1) - 5.0
        g[4] = 0.125 - x0
        g[5] = delta - delta_max
        g[6] = P - P_c
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
        return "welded_beam"

    def get_optimum(self) -> Optional[float]:
        return 1.724852

    def get_optimum_location(self) -> Optional[np.ndarray]:
        return np.array([0.205730, 3.470489, 9.036624, 0.205729])
