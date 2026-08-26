"""Artificial Bee Colony (ABC).

Reference:
    Karaboga, D. (2005). An idea based on honey bee swarm for numerical optimization.
    Technical Report TR06, Erciyes University, Turkey.
    Karaboga, D. and Basturk, B. (2007). A powerful and efficient algorithm for
    numerical function optimization: artificial bee colony (ABC) algorithm.
    *Journal of Global Optimization*, 39(3): 459–471.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from atlas.core.base_algorithm import BaseAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_algorithm


@register_algorithm("abc", aliases=["abc_nfe"])
class ABC(BaseAlgorithm):
    """Artificial Bee Colony.

    Args:
        problem: Optimisation problem instance.
        max_iter: Maximum iterations.
        pop_size: Number of food sources (colony size / 2).
        seed: Random seed.
        verbose: Print progress.
        limit: Abandonment limit for scout bees (defaults to pop_size * dim).
    """

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 500,
        pop_size: int = 30,
        seed: int = 42,
        verbose: bool = False,
        limit: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        dim = problem.get_dim()
        eff_limit = limit if limit is not None else pop_size * dim
        super().__init__(
            problem,
            max_iter=max_iter,
            pop_size=pop_size,
            seed=seed,
            verbose=verbose,
            limit=eff_limit,
            **kwargs,
        )
        self.limit = eff_limit
        self.trial: np.ndarray = np.zeros(self.pop_size, dtype=int)

    def initialize(self) -> None:
        self.population = self.init_population()
        self.fitness = self.evaluate_population(self.population)
        self.trial = np.zeros(self.pop_size, dtype=int)
        self._update_global_best()

    def iterate(self, iter_idx: int) -> float:
        # 1. Employed Bees phase
        for i in range(self.pop_size):
            # Select random neighbor k != i
            k = self.rng.integers(0, self.pop_size - 1)
            if k >= i:
                k += 1

            # Select random dimension
            j = self.rng.integers(0, self.dim)
            phi = self.rng.uniform(-1.0, 1.0)

            v = self.population[i].copy()
            v[j] = v[j] + phi * (self.population[i, j] - self.population[k, j])
            v = self._clip(v)
            f_v = self.evaluate(v)

            if f_v <= self.fitness[i]:
                self.population[i] = v
                self.fitness[i] = f_v
                self.trial[i] = 0
            else:
                self.trial[i] += 1

        # 2. Onlooker Bees phase
        # Calculate fitness for roulette wheel selection
        fit = np.where(self.fitness >= 0, 1.0 / (1.0 + self.fitness), 1.0 + np.abs(self.fitness))
        total_fit = float(np.sum(fit))
        probs = fit / total_fit if total_fit > 0 else np.ones(self.pop_size) / self.pop_size

        for _ in range(self.pop_size):
            i = int(self.rng.choice(self.pop_size, p=probs))

            # Select random neighbor k != i
            k = self.rng.integers(0, self.pop_size - 1)
            if k >= i:
                k += 1

            j = self.rng.integers(0, self.dim)
            phi = self.rng.uniform(-1.0, 1.0)

            v = self.population[i].copy()
            v[j] = v[j] + phi * (self.population[i, j] - self.population[k, j])
            v = self._clip(v)
            f_v = self.evaluate(v)

            if f_v <= self.fitness[i]:
                self.population[i] = v
                self.fitness[i] = f_v
                self.trial[i] = 0
            else:
                self.trial[i] += 1

        # 3. Scout Bees phase
        max_trial_idx = int(np.argmax(self.trial))
        if self.trial[max_trial_idx] > self.limit:
            self.population[max_trial_idx] = self.rng.uniform(self.lb, self.ub)
            self.fitness[max_trial_idx] = self.evaluate(self.population[max_trial_idx])
            self.trial[max_trial_idx] = 0

        return self.g_best_f
