"""Slime Mould Algorithm (SMA).

Reference:
    Li, S., Chen, H., Wang, M., Heidari, A. A., and Mirjalili, S. (2020).
    Slime mould algorithm: A new method for stochastic optimization.
    *Future Generation Computer Systems*, 111: 300–323.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from atlas.core.base_algorithm import BaseAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_algorithm


@register_algorithm("sma", aliases=["sma_nfe"])
class SMA(BaseAlgorithm):
    """Slime Mould Algorithm.

    Args:
        problem: Optimisation problem instance.
        max_iter: Maximum iterations.
        pop_size: Population size.
        seed: Random seed.
        verbose: Print progress.
        z: Oscillation factor / exploration probability threshold.
    """

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 500,
        pop_size: int = 30,
        seed: int = 42,
        verbose: bool = False,
        z: float = 0.03,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            problem,
            max_iter=max_iter,
            pop_size=pop_size,
            seed=seed,
            verbose=verbose,
            z=z,
            **kwargs,
        )
        self.z = z

    def initialize(self) -> None:
        self.population = self.init_population()
        self.fitness = self.evaluate_population(self.population)
        self._update_global_best()

    def iterate(self, iter_idx: int) -> float:
        total_it = max(self.max_iter, 1) if self.max_iter > 0 else 500
        progress = min(iter_idx / total_it, 1.0)

        # Parameter a decreasing non-linearly
        b_val = 1.0 - progress
        a = np.arctanh(np.clip(b_val, -0.999999, 0.999999))

        # Sort population by fitness
        order = np.argsort(self.fitness)
        best_f = float(self.fitness[order[0]])
        worst_f = float(self.fitness[order[-1]])

        # Calculate smell weight matrix W
        w = np.zeros((self.pop_size, self.dim))
        eps = 1e-10
        half = self.pop_size // 2
        for k in range(self.pop_size):
            idx = order[k]
            r = self.rng.random(self.dim)
            ratio = (worst_f - self.fitness[idx]) / (worst_f - best_f + eps)
            if k < half:
                w[idx] = 1.0 + r * np.log10(ratio + 1.0)
            else:
                w[idx] = 1.0 - r * np.log10(ratio + 1.0)

        new_pop = np.empty_like(self.population)
        for i in range(self.pop_size):
            p = np.tanh(np.abs(self.fitness[i] - self.g_best_f))
            r_explore = self.rng.random()
            if r_explore < self.z:
                new_pop[i] = self.rng.uniform(self.lb, self.ub)
            else:
                r_p = self.rng.random()
                idx_a = self.rng.integers(0, self.pop_size)
                idx_b = self.rng.integers(0, self.pop_size)
                v_b = self.rng.uniform(-a, a, size=self.dim)
                v_c = self.rng.uniform(-b_val, b_val, size=self.dim)
                if r_p < p:
                    new_pop[i] = self.g_best_x + v_b * (w[i] * self.population[idx_a] - self.population[idx_b])
                else:
                    new_pop[i] = v_c * self.population[i]

            new_pop[i] = self._clip(new_pop[i])
            self.population[i] = new_pop[i]
            self.fitness[i] = self.evaluate(new_pop[i])

        return self.g_best_f
