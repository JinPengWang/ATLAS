"""Logistic-Gauss Circle Optimizer (LGC)."""

from __future__ import annotations

from typing import Any

import numpy as np

from atlas.core.base_algorithm import BaseAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_algorithm


@register_algorithm("lgc")
class LGC(BaseAlgorithm):
    """Logistic-Gauss Circle Optimizer."""

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 500,
        pop_size: int = 30,
        seed: int = 42,
        verbose: bool = False,
        u_max: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(problem, max_iter, pop_size, seed, verbose, u_max=u_max, **kwargs)
        self.u_max = u_max

    def get_name(self) -> str:
        return "LGC"

    def initialize(self) -> None:
        self.population = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([
            self.problem.evaluate_with_penalty(self.population[i])
            for i in range(self.pop_size)
        ])
        self._update_global_best()

    def _wrap_to_bounds(self, x: np.ndarray) -> np.ndarray:
        span = self.ub - self.lb
        return self.lb + np.mod(x - self.lb, span + np.finfo(float).eps)

    def iterate(self, iter_idx: int) -> float:
        t = iter_idx + 1
        total = max(self.max_iter, 2)
        u = (1.0 - np.log(t) / np.log(total)) * self.u_max
        dist = np.sqrt(np.sum((self.g_best_x - self.population) ** 2, axis=1))
        g = u * np.log1p(dist)

        next_population = self.population.copy()
        next_fitness = self.fitness.copy()
        for i in range(self.pop_size):
            candidate = self.population[i].copy()
            for j in range(self.dim):
                if self.rng.random() < u:
                    candidate[j] = (
                        (2.0 * self.rng.random() - 1.0)
                        * self.population[i, j]
                        * (self.ub[j] - self.population[i, j])
                    )
                    if candidate[j] > self.ub[j] or candidate[j] < self.lb[j]:
                        candidate[j] = self._wrap_to_bounds(candidate)[j]
                else:
                    mod_base = self.lb[j] + np.mod(
                        self.rng.random() / np.pi * self.population[i, j] - self.lb[j],
                        self.ub[j] - self.lb[j] + np.finfo(float).eps,
                    )
                    candidate[j] = self.g_best_x[j] + g[i] * (2.0 * self.rng.random() - 1.0) * (
                        self.g_best_x[j] - mod_base
                    )
            candidate = self._clip(candidate)
            candidate_f = self.problem.evaluate_with_penalty(candidate)
            if candidate_f < self.fitness[i]:
                next_population[i] = candidate
                next_fitness[i] = candidate_f

        self.population = next_population
        self.fitness = next_fitness
        self._update_global_best()
        return self.g_best_f

