"""Logistic-Gauss Circle Optimizer (LGC) -- NFE-based variant."""

from __future__ import annotations

from typing import Any

import numpy as np

from atlas.algorithms.nfe.base_nfe_algorithm import BaseNFEAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_algorithm


@register_algorithm("lgc_nfe")
class LGC_NFE(BaseNFEAlgorithm):
    """Logistic-Gauss Circle Optimizer with NFE-based stopping."""

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 0,
        max_nfe: int = 10000,
        pop_size: int = 30,
        seed: int = 42,
        verbose: bool = False,
        u_max: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(problem, max_iter, max_nfe, pop_size, seed, verbose, u_max=u_max, **kwargs)
        self.u_max = u_max

    def get_name(self) -> str:
        return "LGC_NFE"

    def initialize(self) -> None:
        self.population = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([self._evaluate(self.population[i]) for i in range(self.pop_size)])
        self._update_global_best()

    def _wrap_to_bounds(self, x: np.ndarray) -> np.ndarray:
        span = self.ub - self.lb
        return self.lb + np.mod(x - self.lb, span + np.finfo(float).eps)

    def iterate(self, iter_idx: int) -> float:
        progress = min(self._nfe / max(self.max_nfe, 2), 1.0)
        u = (1.0 - np.log(max(self._nfe, 1)) / np.log(max(self.max_nfe, 2))) * self.u_max
        u = max(float(u), 0.0)
        dist = np.sqrt(np.sum((self.g_best_x - self.population) ** 2, axis=1))
        g = u * np.log1p(dist)
        next_population = self.population.copy()
        next_fitness = self.fitness.copy()

        for i in range(self.pop_size):
            if self._nfe >= self.max_nfe:
                break
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
            candidate_f = self._evaluate(candidate)
            if candidate_f < self.fitness[i]:
                next_population[i] = candidate
                next_fitness[i] = candidate_f

        self.population = next_population
        self.fitness = next_fitness
        self._update_global_best()
        return self.g_best_f

