"""Differential Evolution (DE) – NFE-based variant.

Reference:
    Storn, R. and Price, K. (1997). Differential evolution – a simple and
    efficient heuristic for global optimization over continuous spaces.
    *Journal of Global Optimization*, 11(4): 341–359.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from atlas.algorithms.nfe.base_nfe_algorithm import BaseNFEAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_algorithm


@register_algorithm("de_nfe")
class DE_NFE(BaseNFEAlgorithm):
    """Differential Evolution (NFE-based stopping).

    Args:
        problem: Optimisation problem instance.
        max_nfe: Maximum function evaluations.
        pop_size: Population size.
        seed: Random seed.
        verbose: Print progress.
        F: Mutation scaling factor.
        CR: Crossover probability.
        strategy: Mutation strategy.
    """

    STRATEGIES = ("rand/1/bin", "best/1/bin", "rand/2/bin", "best/2/bin")

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 0,
        max_nfe: int = 10000,
        pop_size: int = 30,
        seed: int = 42,
        verbose: bool = False,
        F: float = 0.8,
        CR: float = 0.9,
        strategy: str = "rand/1/bin",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            problem, max_iter, max_nfe, pop_size, seed, verbose,
            F=F, CR=CR, strategy=strategy, **kwargs,
        )
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")
        self.F = F
        self.CR = CR
        self.strategy = strategy

    def get_name(self) -> str:
        return "DE_NFE"

    def initialize(self) -> None:
        self.population = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([
            self._evaluate(self.population[i])
            for i in range(self.pop_size)
        ])
        self._update_global_best()

    def iterate(self, iter_idx: int) -> float:
        for i in range(self.pop_size):
            v = self._mutate(i)
            u = self._crossover(self.population[i], v)
            f_u = self._evaluate(u)
            if f_u <= self.fitness[i]:
                self.population[i] = u
                self.fitness[i] = f_u
                if f_u < self.g_best_f:
                    self.g_best_f = f_u
                    self.g_best_x = u.copy()

        return self.g_best_f

    # ---- Mutation strategies ------------------------------------------
    def _mutate(self, target_idx: int) -> np.ndarray:
        pop = self.population
        n = self.pop_size
        idxs = list(range(n))
        idxs.remove(target_idx)

        if self.strategy == "rand/1/bin":
            a, b, c = self.rng.choice(idxs, size=3, replace=False)
            v = pop[a] + self.F * (pop[b] - pop[c])
        elif self.strategy == "best/1/bin":
            best_idx = int(np.argmin(self.fitness))
            a, b = self.rng.choice(idxs, size=2, replace=False)
            v = pop[best_idx] + self.F * (pop[a] - pop[b])
        elif self.strategy == "rand/2/bin":
            a, b, c, d, e = self.rng.choice(idxs, size=5, replace=False)
            v = pop[a] + self.F * (pop[b] - pop[c]) + self.F * (pop[d] - pop[e])
        elif self.strategy == "best/2/bin":
            best_idx = int(np.argmin(self.fitness))
            a, b, c, d = self.rng.choice(idxs, size=4, replace=False)
            v = pop[best_idx] + self.F * (pop[a] - pop[b]) + self.F * (pop[c] - pop[d])
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return self._clip(v)

    # ---- Binomial crossover -------------------------------------------
    def _crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        u = target.copy()
        j_rand = self.rng.integers(0, self.dim)
        for j in range(self.dim):
            if self.rng.random() < self.CR or j == j_rand:
                u[j] = mutant[j]
        return u
