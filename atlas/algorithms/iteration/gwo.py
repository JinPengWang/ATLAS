"""Grey Wolf Optimizer (GWO).

Reference:
    Mirjalili, S., Mirjalili, S. M., and Lewis, A. (2014). Grey Wolf Optimizer.
    *Advances in Engineering Software*, 69: 46–61.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from atlas.core.base_algorithm import BaseAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_algorithm


@register_algorithm("gwo", aliases=["gwo_nfe"])
class GWO(BaseAlgorithm):
    """Grey Wolf Optimizer.

    Args:
        problem: Optimisation problem instance.
        max_iter: Maximum iterations.
        pop_size: Population size.
        seed: Random seed.
        verbose: Print progress.
    """

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 500,
        pop_size: int = 30,
        seed: int = 42,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            problem,
            max_iter=max_iter,
            pop_size=pop_size,
            seed=seed,
            verbose=verbose,
            **kwargs,
        )

    def initialize(self) -> None:
        self.population = self.init_population()
        self.fitness = self.evaluate_population(self.population)

        # Sort initial wolves to identify alpha, beta, and delta
        order = np.argsort(self.fitness)
        self.alpha_x = self.population[order[0]].copy()
        self.alpha_f = float(self.fitness[order[0]])
        self.beta_x = self.population[order[1]].copy() if self.pop_size > 1 else self.alpha_x.copy()
        self.beta_f = float(self.fitness[order[1]]) if self.pop_size > 1 else self.alpha_f
        self.delta_x = self.population[order[2]].copy() if self.pop_size > 2 else self.beta_x.copy()
        self.delta_f = float(self.fitness[order[2]]) if self.pop_size > 2 else self.beta_f

        self._update_global_best()

    def iterate(self, iter_idx: int) -> float:
        total_it = max(self.max_iter, 1) if self.max_iter > 0 else 500
        # Linearly decreasing a: 2 -> 0
        a = 2.0 - 2.0 * (iter_idx / total_it)

        for i in range(self.pop_size):
            r1 = self.rng.random(self.dim)
            r2 = self.rng.random(self.dim)
            A1 = 2.0 * a * r1 - a
            C1 = 2.0 * r2
            D_alpha = np.abs(C1 * self.alpha_x - self.population[i])
            X1 = self.alpha_x - A1 * D_alpha

            r1 = self.rng.random(self.dim)
            r2 = self.rng.random(self.dim)
            A2 = 2.0 * a * r1 - a
            C2 = 2.0 * r2
            D_beta = np.abs(C2 * self.beta_x - self.population[i])
            X2 = self.beta_x - A2 * D_beta

            r1 = self.rng.random(self.dim)
            r2 = self.rng.random(self.dim)
            A3 = 2.0 * a * r1 - a
            C3 = 2.0 * r2
            D_delta = np.abs(C3 * self.delta_x - self.population[i])
            X3 = self.delta_x - A3 * D_delta

            X_new = (X1 + X2 + X3) / 3.0
            self.population[i] = self._clip(X_new)
            f = self.evaluate(self.population[i])
            self.fitness[i] = f

            # Update alpha, beta, and delta positions and fitnesses
            if f < self.alpha_f:
                self.delta_f = self.beta_f
                self.delta_x = self.beta_x.copy()
                self.beta_f = self.alpha_f
                self.beta_x = self.alpha_x.copy()
                self.alpha_f = f
                self.alpha_x = self.population[i].copy()
            elif f < self.beta_f:
                self.delta_f = self.beta_f
                self.delta_x = self.beta_x.copy()
                self.beta_f = f
                self.beta_x = self.population[i].copy()
            elif f < self.delta_f:
                self.delta_f = f
                self.delta_x = self.population[i].copy()

        return self.g_best_f
