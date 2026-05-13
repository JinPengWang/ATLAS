"""Whale Optimisation Algorithm (WOA).

Reference:
    Mirjalili, S. and Lewis, A. (2016). The whale optimization algorithm.
    *Advances in Engineering Software*, 95: 51–67.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from atlas.core.base_algorithm import BaseAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_algorithm


@register_algorithm("woa")
class WOA(BaseAlgorithm):
    """Whale Optimisation Algorithm.

    Args:
        problem: Optimisation problem instance.
        max_iter: Maximum iterations.
        pop_size: Population size.
        seed: Random seed.
        verbose: Print progress.
        b: Spiral shape constant.
    """

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 500,
        pop_size: int = 30,
        seed: int = 42,
        verbose: bool = False,
        b: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(problem, max_iter, pop_size, seed, verbose, b=b, **kwargs)
        self.b = b

    def get_name(self) -> str:
        return "WOA"

    def initialize(self) -> None:
        self.population = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([
            self.problem.evaluate_with_penalty(self.population[i])
            for i in range(self.pop_size)
        ])
        self._update_global_best()

    def iterate(self, iter_idx: int) -> float:
        # Linearly decreasing a: 2 → 0
        a = 2.0 - 2.0 * iter_idx / max(self.max_iter - 1, 1)

        for i in range(self.pop_size):
            r1 = self.rng.random()
            r2 = self.rng.random()
            A = 2.0 * a * r1 - a   # coefficient vector
            C = 2.0 * r2            # coefficient vector
            p = self.rng.random()    # probability for spiral vs encircling

            if p < 0.5:
                # Encircling prey or search for prey
                if abs(A) < 1:
                    # Encircling prey (exploitation)
                    D = abs(C * self.g_best_x - self.population[i])
                    self.population[i] = self.g_best_x - A * D
                else:
                    # Search for prey (exploration) – random agent
                    rand_idx = self.rng.integers(0, self.pop_size)
                    rand_agent = self.population[rand_idx]
                    D = abs(C * rand_agent - self.population[i])
                    self.population[i] = rand_agent - A * D
            else:
                # Spiral update (bubble-net attack)
                D_prime = abs(self.g_best_x - self.population[i])
                l = self.rng.uniform(-1, 1)
                self.population[i] = (
                    D_prime * np.exp(self.b * l) * np.cos(2.0 * np.pi * l)
                    + self.g_best_x
                )

            self.population[i] = self._clip(self.population[i])
            self.fitness[i] = self.problem.evaluate_with_penalty(self.population[i])

        self._update_global_best()
        return self.g_best_f
