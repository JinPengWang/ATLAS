"""Simulated Annealing (SA) – NFE-based variant.

Reference:
    Kirkpatrick, S., Gelatt, C. D., and Vecchi, M. P. (1983). Optimization by
    simulated annealing. *Science*, 220(4598): 671–680.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from atlas.algorithms.nfe.base_nfe_algorithm import BaseNFEAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_algorithm


@register_algorithm("sa_nfe")
class SA_NFE(BaseNFEAlgorithm):
    """Simulated Annealing (NFE-based stopping).

    SA operates on a single candidate solution.  ``pop_size`` is ignored
    (always 1).

    Args:
        problem: Optimisation problem instance.
        max_nfe: Maximum function evaluations.
        seed: Random seed.
        verbose: Print progress.
        T_init: Initial temperature.
        T_min: Minimum (final) temperature.
        alpha: Cooling rate.
        step_size: Gaussian perturbation step size relative to search range.
    """

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 0,
        max_nfe: int = 10000,
        pop_size: int = 1,
        seed: int = 42,
        verbose: bool = False,
        T_init: float = 1000.0,
        T_min: float = 1e-3,
        alpha: float = 0.95,
        step_size: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            problem, max_iter, max_nfe, 1, seed, verbose,
            T_init=T_init, T_min=T_min, alpha=alpha, step_size=step_size,
            **kwargs,
        )
        self.T_init = T_init
        self.T_min = T_min
        self.alpha = alpha
        self.step_size = step_size
        self.pop_size = 1

    def get_name(self) -> str:
        return "SA_NFE"

    def initialize(self) -> None:
        self.current_x = self.rng.uniform(self.lb, self.ub)
        self.current_f = self._evaluate(self.current_x)
        self.best_x = self.current_x.copy()
        self.best_f = self.current_f

        self.g_best_x = self.best_x.copy()
        self.g_best_f = self.best_f

        self.population = self.current_x.reshape(1, -1)
        self.fitness = np.array([self.current_f])

        self._sigma = self.step_size * (self.ub - self.lb)

    def iterate(self, iter_idx: int) -> float:
        T = max(self.T_init * (self.alpha ** iter_idx), self.T_min)

        candidate = self.current_x + self.rng.normal(0, self._sigma)
        candidate = self._clip(candidate)
        candidate_f = self._evaluate(candidate)

        delta = candidate_f - self.current_f
        if delta < 0:
            accept = True
        else:
            prob = np.exp(-delta / T) if T > 1e-300 else 0.0
            accept = self.rng.random() < prob

        if accept:
            self.current_x = candidate
            self.current_f = candidate_f
            if candidate_f < self.best_f:
                self.best_x = candidate.copy()
                self.best_f = candidate_f

        self.g_best_x = self.best_x.copy()
        self.g_best_f = self.best_f
        self.population = self.best_x.reshape(1, -1)
        self.fitness = np.array([self.best_f])

        return self.g_best_f
