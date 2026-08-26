"""Differential Evolution (DE).

Reference:
    Storn, R. and Price, K. (1997). Differential evolution – a simple and
    efficient heuristic for global optimization over continuous spaces.
    *Journal of Global Optimization*, 11(4): 341–359.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from atlas.core.base_algorithm import BaseAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_algorithm


@register_algorithm("de", aliases=["de_nfe"])
class DE(BaseAlgorithm):
    """Differential Evolution.

    Args:
        problem: Optimisation problem instance.
        max_iter: Maximum generations.
        pop_size: Population size.
        seed: Random seed.
        verbose: Print progress.
        F: Mutation scaling factor.
        CR: Crossover probability.
        strategy: Mutation strategy.  Supported: ``'rand/1/bin'``,
            ``'best/1/bin'``, ``'rand/2/bin'``, ``'best/2/bin'``.
    """

    STRATEGIES = ("rand/1/bin", "best/1/bin", "rand/2/bin", "best/2/bin")

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 500,
        pop_size: int = 30,
        seed: int = 42,
        verbose: bool = False,
        F: float = 0.8,
        CR: float = 0.9,
        strategy: str = "rand/1/bin",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            problem, max_iter=max_iter, pop_size=pop_size, seed=seed, verbose=verbose,
            F=F, CR=CR, strategy=strategy, **kwargs,
        )
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")
        self.F = F
        self.CR = CR
        self.strategy = strategy

    def initialize(self) -> None:
        self.population = self.init_population()
        self.fitness = self.evaluate_population(self.population)
        self._update_global_best()

    def iterate(self, iter_idx: int) -> float:
        for i in range(self.pop_size):
            # Mutation
            v = self._mutate(i)
            # Crossover (binomial)
            u = self._crossover(self.population[i], v)
            # Selection
            f_u = self.evaluate(u)
            if f_u <= self.fitness[i]:
                self.population[i] = u
                self.fitness[i] = f_u

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
