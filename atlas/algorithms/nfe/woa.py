"""Whale Optimisation Algorithm (WOA) – NFE-based variant.

Reference:
    Mirjalili, S. and Lewis, A. (2016). The whale optimization algorithm.
    *Advances in Engineering Software*, 95: 51–67.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from atlas.algorithms.nfe.base_nfe_algorithm import BaseNFEAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_algorithm


@register_algorithm("woa_nfe")
class WOA_NFE(BaseNFEAlgorithm):
    """Whale Optimisation Algorithm (NFE-based stopping).

    Args:
        problem: Optimisation problem instance.
        max_nfe: Maximum function evaluations.
        pop_size: Population size.
        seed: Random seed.
        verbose: Print progress.
        b: Spiral shape constant.
    """

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 0,
        max_nfe: int = 10000,
        pop_size: int = 30,
        seed: int = 42,
        verbose: bool = False,
        b: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(problem, max_iter, max_nfe, pop_size, seed, verbose, b=b, **kwargs)
        self.b = b

    def get_name(self) -> str:
        return "WOA_NFE"

    def initialize(self) -> None:
        self.population = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([
            self._evaluate(self.population[i])
            for i in range(self.pop_size)
        ])
        self._update_global_best()

    def iterate(self, iter_idx: int) -> float:
        # Linearly decreasing a: 2 -> 0
        # Use max_nfe / pop_size as approximate iteration count
        approx_max_iter = max(self.max_nfe // self.pop_size, 1)
        a = 2.0 - 2.0 * iter_idx / max(approx_max_iter - 1, 1)

        for i in range(self.pop_size):
            r1 = self.rng.random()
            r2 = self.rng.random()
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            p = self.rng.random()

            if p < 0.5:
                if abs(A) < 1:
                    D = abs(C * self.g_best_x - self.population[i])
                    self.population[i] = self.g_best_x - A * D
                else:
                    rand_idx = self.rng.integers(0, self.pop_size)
                    rand_agent = self.population[rand_idx]
                    D = abs(C * rand_agent - self.population[i])
                    self.population[i] = rand_agent - A * D
            else:
                D_prime = abs(self.g_best_x - self.population[i])
                l = self.rng.uniform(-1, 1)
                self.population[i] = (
                    D_prime * np.exp(self.b * l) * np.cos(2.0 * np.pi * l)
                    + self.g_best_x
                )

            self.population[i] = self._clip(self.population[i])
            self.fitness[i] = self._evaluate(self.population[i])

        self._update_global_best()
        return self.g_best_f
