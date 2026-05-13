"""Genetic Algorithm (GA) – NFE-based variant.

Reference:
    Holland, J. H. (1992). *Adaptation in Natural and Artificial Systems*.
    MIT Press.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from atlas.algorithms.nfe.base_nfe_algorithm import BaseNFEAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_algorithm


@register_algorithm("ga_nfe")
class GA_NFE(BaseNFEAlgorithm):
    """Real-coded Genetic Algorithm (NFE-based stopping).

    Uses tournament selection, SBX crossover, and polynomial mutation.

    Args:
        problem: Optimisation problem instance.
        max_nfe: Maximum function evaluations.
        pop_size: Population size.
        seed: Random seed.
        verbose: Print progress.
        crossover_prob: Probability of crossover.
        mutation_prob: Per-gene mutation probability.
        tournament_size: Number of individuals in tournament selection.
        eta_c: SBX crossover distribution index.
        eta_m: Polynomial mutation distribution index.
    """

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 0,
        max_nfe: int = 10000,
        pop_size: int = 50,
        seed: int = 42,
        verbose: bool = False,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.01,
        tournament_size: int = 3,
        eta_c: float = 20.0,
        eta_m: float = 20.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            problem, max_iter, max_nfe, pop_size, seed, verbose,
            crossover_prob=crossover_prob, mutation_prob=mutation_prob,
            tournament_size=tournament_size, eta_c=eta_c, eta_m=eta_m,
            **kwargs,
        )
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.eta_c = eta_c
        self.eta_m = eta_m

    def get_name(self) -> str:
        return "GA_NFE"

    def initialize(self) -> None:
        self.population = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([
            self._evaluate(self.population[i])
            for i in range(self.pop_size)
        ])
        self._update_global_best()

    def iterate(self, iter_idx: int) -> float:
        new_pop = np.empty_like(self.population)

        for i in range(self.pop_size):
            p1 = self._tournament_select()
            p2 = self._tournament_select()

            if self.rng.random() < self.crossover_prob:
                c1, c2 = self._sbx_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = self._polynomial_mutation(c1)
            c2 = self._polynomial_mutation(c2)

            c1 = self._clip(c1)
            c2 = self._clip(c2)

            f1 = self._evaluate(c1)
            f2 = self._evaluate(c2)
            if f1 <= f2:
                new_pop[i] = c1
                self.fitness[i] = f1
            else:
                new_pop[i] = c2
                self.fitness[i] = f2

        self.population = new_pop
        self._update_global_best()
        return self.g_best_f

    # ---- Selection ----------------------------------------------------
    def _tournament_select(self) -> np.ndarray:
        idxs = self.rng.integers(0, self.pop_size, size=self.tournament_size)
        best = idxs[np.argmin(self.fitness[idxs])]
        return self.population[best].copy()

    # ---- SBX crossover ------------------------------------------------
    def _sbx_crossover(self, p1: np.ndarray, p2: np.ndarray) -> tuple:
        c1 = p1.copy()
        c2 = p2.copy()
        for j in range(self.dim):
            if self.rng.random() > 0.5:
                continue
            if abs(p1[j] - p2[j]) < 1e-14:
                continue
            u = self.rng.random()
            if u <= 0.5:
                beta = (2.0 * u) ** (1.0 / (self.eta_c + 1.0))
            else:
                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.eta_c + 1.0))
            c1[j] = 0.5 * ((1 + beta) * p1[j] + (1 - beta) * p2[j])
            c2[j] = 0.5 * ((1 - beta) * p1[j] + (1 + beta) * p2[j])
        return c1, c2

    # ---- Polynomial mutation ------------------------------------------
    def _polynomial_mutation(self, x: np.ndarray) -> np.ndarray:
        x_new = x.copy()
        for j in range(self.dim):
            if self.rng.random() > self.mutation_prob:
                continue
            delta_max = self.ub[j] - self.lb[j]
            if delta_max < 1e-14:
                continue
            u = self.rng.random()
            if u < 0.5:
                delta = (2.0 * u) ** (1.0 / (self.eta_m + 1.0)) - 1.0
            else:
                delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (self.eta_m + 1.0))
            x_new[j] = x[j] + delta * delta_max
        return x_new
