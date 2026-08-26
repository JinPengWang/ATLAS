"""Deb-Thiele-Laumanns-Zitzler (DTLZ) Scalable Many-Objective Benchmark Suite.

Reference:
    Deb, K., Thiele, L., Laumanns, M., & Zitzler, E. (2002).
    Scalable multi-objective optimization test problems.
    In Congress on Evolutionary Computation (CEC 2002), pp. 825-830.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from atlas.multiobjective.base_mo_problem import BaseMOProblem


class DTLZ1(BaseMOProblem):
    """DTLZ1 problem with linear hyperplane Pareto front.

    Pareto Front equation: sum(f_m) = 0.5.
    """

    def __init__(self, dim: int = 7, n_objectives: int = 3, **kwargs) -> None:
        super().__init__(dim=dim, n_objectives=n_objectives, **kwargs)
        self.k = dim - n_objectives + 1
        self._lb = np.zeros(dim, dtype=float)
        self._ub = np.ones(dim, dtype=float)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        M = self.n_objectives
        xm = x[M - 1 :]
        g = 100.0 * (self.k + np.sum((xm - 0.5) ** 2 - np.cos(20.0 * np.pi * (xm - 0.5))))

        f = np.zeros(M, dtype=float)
        for i in range(M):
            val = 0.5 * (1.0 + g)
            for j in range(M - 1 - i):
                val *= x[j]
            if i > 0:
                val *= 1.0 - x[M - 1 - i]
            f[i] = val
        return f

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lb.copy(), self._ub.copy()

    def get_name(self) -> str:
        return f"dtlz1_m{self.n_objectives}"


class DTLZ2(BaseMOProblem):
    """DTLZ2 problem with spherical concave Pareto front.

    Pareto Front equation: sum(f_m^2) = 1.0.
    """

    def __init__(self, dim: int = 12, n_objectives: int = 3, **kwargs) -> None:
        super().__init__(dim=dim, n_objectives=n_objectives, **kwargs)
        self._lb = np.zeros(dim, dtype=float)
        self._ub = np.ones(dim, dtype=float)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        M = self.n_objectives
        xm = x[M - 1 :]
        g = np.sum((xm - 0.5) ** 2)

        f = np.zeros(M, dtype=float)
        for i in range(M):
            val = 1.0 + g
            for j in range(M - 1 - i):
                val *= np.cos(x[j] * np.pi / 2.0)
            if i > 0:
                val *= np.sin(x[M - 1 - i] * np.pi / 2.0)
            f[i] = val
        return f

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lb.copy(), self._ub.copy()

    def get_name(self) -> str:
        return f"dtlz2_m{self.n_objectives}"
