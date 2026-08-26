"""Zitzler-Deb-Thiele (ZDT) Multi-Objective Benchmark Problem Suite.

Reference:
    Zitzler, E., Deb, K., & Thiele, L. (2000). Comparison of multiobjective
    evolutionary algorithms: Empirical results. Evolutionary Computation, 8(2), 173-195.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from atlas.multiobjective.base_mo_problem import BaseMOProblem


class ZDT1(BaseMOProblem):
    """ZDT1 problem with convex Pareto front.

    True Pareto Front: f2 = 1 - sqrt(f1), f1 in [0, 1].
    """

    def __init__(self, dim: int = 30, **kwargs) -> None:
        super().__init__(dim=dim, n_objectives=2, **kwargs)
        self._lb = np.zeros(dim, dtype=float)
        self._ub = np.ones(dim, dtype=float)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.dim - 1.0)
        h = 1.0 - np.sqrt(f1 / g)
        f2 = g * h
        return np.array([float(f1), float(f2)])

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lb.copy(), self._ub.copy()

    def get_name(self) -> str:
        return "zdt1"

    def get_pareto_front(self, n_points: int = 100) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        f2 = 1.0 - np.sqrt(f1)
        return np.column_stack([f1, f2])


class ZDT2(BaseMOProblem):
    """ZDT2 problem with non-convex (concave) Pareto front.

    True Pareto Front: f2 = 1 - f1^2, f1 in [0, 1].
    """

    def __init__(self, dim: int = 30, **kwargs) -> None:
        super().__init__(dim=dim, n_objectives=2, **kwargs)
        self._lb = np.zeros(dim, dtype=float)
        self._ub = np.ones(dim, dtype=float)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.dim - 1.0)
        h = 1.0 - (f1 / g) ** 2
        f2 = g * h
        return np.array([float(f1), float(f2)])

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lb.copy(), self._ub.copy()

    def get_name(self) -> str:
        return "zdt2"

    def get_pareto_front(self, n_points: int = 100) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        f2 = 1.0 - f1**2
        return np.column_stack([f1, f2])


class ZDT3(BaseMOProblem):
    """ZDT3 problem with disconnected Pareto front regions.

    True Pareto Front consists of 5 disconnected non-contiguous curves.
    """

    def __init__(self, dim: int = 30, **kwargs) -> None:
        super().__init__(dim=dim, n_objectives=2, **kwargs)
        self._lb = np.zeros(dim, dtype=float)
        self._ub = np.ones(dim, dtype=float)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.dim - 1.0)
        h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1)
        f2 = g * h
        return np.array([float(f1), float(f2)])

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lb.copy(), self._ub.copy()

    def get_name(self) -> str:
        return "zdt3"

    def get_pareto_front(self, n_points: int = 500) -> np.ndarray:
        f1_grid = np.linspace(0.0, 1.0, n_points)
        f2_grid = 1.0 - np.sqrt(f1_grid) - f1_grid * np.sin(10.0 * np.pi * f1_grid)
        pts = np.column_stack([f1_grid, f2_grid])
        # Filter non-dominated points
        non_dom = []
        for i in range(len(pts)):
            dom = False
            for j in range(len(pts)):
                if (
                    i != j
                    and np.all(pts[j] <= pts[i])
                    and np.any(pts[j] < pts[i])
                ):
                    dom = True
                    break
            if not dom:
                non_dom.append(pts[i])
        return np.array(non_dom)


class ZDT4(BaseMOProblem):
    """ZDT4 problem with 21^9 local Pareto fronts (rugged multi-modality)."""

    def __init__(self, dim: int = 10, **kwargs) -> None:
        super().__init__(dim=dim, n_objectives=2, **kwargs)
        self._lb = np.full(dim, -5.0, dtype=float)
        self._lb[0] = 0.0
        self._ub = np.full(dim, 5.0, dtype=float)
        self._ub[0] = 1.0

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = 1.0 + 10.0 * (self.dim - 1.0) + np.sum(x[1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * x[1:]))
        h = 1.0 - np.sqrt(f1 / g)
        f2 = g * h
        return np.array([float(f1), float(f2)])

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lb.copy(), self._ub.copy()

    def get_name(self) -> str:
        return "zdt4"

    def get_pareto_front(self, n_points: int = 100) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        f2 = 1.0 - np.sqrt(f1)
        return np.column_stack([f1, f2])


class ZDT6(BaseMOProblem):
    """ZDT6 problem with non-uniform mapping along the Pareto front."""

    def __init__(self, dim: int = 10, **kwargs) -> None:
        super().__init__(dim=dim, n_objectives=2, **kwargs)
        self._lb = np.zeros(dim, dtype=float)
        self._ub = np.ones(dim, dtype=float)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        f1 = 1.0 - np.exp(-4.0 * x[0]) * (np.sin(6.0 * np.pi * x[0]) ** 6)
        g = 1.0 + 9.0 * ((np.sum(x[1:]) / (self.dim - 1.0)) ** 0.25)
        h = 1.0 - (f1 / g) ** 2
        f2 = g * h
        return np.array([float(f1), float(f2)])

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lb.copy(), self._ub.copy()

    def get_name(self) -> str:
        return "zdt6"

    def get_pareto_front(self, n_points: int = 100) -> np.ndarray:
        # f1 ranges from ~0.28 to 1.0
        f1 = np.linspace(0.280775, 1.0, n_points)
        f2 = 1.0 - f1**2
        return np.column_stack([f1, f2])
