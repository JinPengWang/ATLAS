"""Traffic Jam Optimizer (TJO)."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from atlas.core.base_algorithm import BaseAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_algorithm


@register_algorithm("tjo", aliases=["tjo_nfe"])
class TJO(BaseAlgorithm):
    """Traffic Jam Optimizer.

    Args:
        problem: Optimisation problem instance.
        max_iter: Maximum iterations.
        pop_size: Number of cars/drivers.
        seed: Random seed.
        verbose: Print progress.
        a: Start/end control range for traffic-police guidance.
        c: Start/end control range for driver self-adjustment.
    """

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 500,
        pop_size: int = 30,
        seed: int = 42,
        verbose: bool = False,
        a: Tuple[float, float] = (2.0, 0.0),
        c: Tuple[float, float] = (2.0, 0.0),
        **kwargs: Any,
    ) -> None:
        super().__init__(problem, max_iter=max_iter, pop_size=pop_size, seed=seed, verbose=verbose, a=a, c=c, **kwargs)
        self.a_range = a
        self.c_range = c

    def initialize(self) -> None:
        self.population = self.init_population()
        self.fitness = self.evaluate_population(self.population)
        self.memory_x = self.population.copy()
        self.memory_f = self.fitness.copy()
        self._update_global_best()

    def iterate(self, iter_idx: int) -> float:
        total_it = max(self.max_iter, 1) if self.max_iter > 0 else 500
        progress = min((iter_idx + 1) / total_it, 1.0)
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
            if self.rng.random() > 0.5:
                x[i] = x[i] + c_t * np.sin(np.pi * self.rng.random()) * (x[rand_idx] - x[i])
            else:
                x[i] = x[i] + c_t * np.sin(np.pi * self.rng.random()) * (best_x[rand_idx] - x[i])

        x = best_x + a_t * np.sin(2.0 * np.pi * self.rng.random((self.pop_size, 1))) * (best_x - x)
        x = self._clip(x)

        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(x[i])

        improved = self.fitness < self.memory_f
        self.memory_f[improved] = self.fitness[improved]
        self.memory_x[improved] = x[improved]
        self.population = x
        return self.g_best_f


