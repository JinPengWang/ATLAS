"""Traffic Jam Optimizer (TJO) -- NFE-based variant."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from atlas.algorithms.nfe.base_nfe_algorithm import BaseNFEAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_algorithm


@register_algorithm("tjo_nfe")
class TJO_NFE(BaseNFEAlgorithm):
    """Traffic Jam Optimizer with function-evaluation-based stopping."""

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 0,
        max_nfe: int = 10000,
        pop_size: int = 30,
        seed: int = 42,
        verbose: bool = False,
        a: Tuple[float, float] = (2.0, 0.0),
        c: Tuple[float, float] = (2.0, 0.0),
        **kwargs: Any,
    ) -> None:
        super().__init__(problem, max_iter, max_nfe, pop_size, seed, verbose, a=a, c=c, **kwargs)
        self.a_range = a
        self.c_range = c

    def get_name(self) -> str:
        return "TJO_NFE"

    def initialize(self) -> None:
        self.population = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([self._evaluate(self.population[i]) for i in range(self.pop_size)])
        self.memory_x = self.population.copy()
        self.memory_f = self.fitness.copy()
        self._update_global_best()

    def iterate(self, iter_idx: int) -> float:
        progress = min(self._nfe / max(self.max_nfe, 1), 1.0)
        a_t = self.a_range[0] + progress * (self.a_range[1] - self.a_range[0])
        c_t = self.c_range[0] + progress * (self.c_range[1] - self.c_range[0])
        best_x = (1.0 - progress) * self.memory_x + progress * self.g_best_x
        jam = (
            (1.0 - progress)
            * np.exp(-progress)
            * np.sin(2.0 * np.pi * self.rng.random((self.pop_size, 1)))
            * np.cos(2.0 * np.pi * self.rng.random((self.pop_size, 1)))
            * c_t
        )
        x = best_x + jam * ((self.ub - self.lb) * self.rng.random((self.pop_size, self.dim)) + self.lb)
        for i in range(self.pop_size):
            rand_idx = self.rng.integers(0, self.pop_size)
            guide = x[rand_idx] if self.rng.random() > 0.5 else best_x[rand_idx]
            x[i] = x[i] + c_t * np.sin(np.pi * self.rng.random()) * (guide - x[i])
        x = best_x + a_t * np.sin(2.0 * np.pi * self.rng.random((self.pop_size, 1))) * (best_x - x)
        x = self._clip(x)

        for i in range(self.pop_size):
            if self._nfe >= self.max_nfe:
                break
            self.fitness[i] = self._evaluate(x[i])
        improved = self.fitness < self.memory_f
        self.memory_f[improved] = self.fitness[improved]
        self.memory_x[improved] = x[improved]
        self.population = x
        self._update_global_best()
        return self.g_best_f

