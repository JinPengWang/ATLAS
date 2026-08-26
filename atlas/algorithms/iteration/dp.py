"""Delta Plus (DP) optimisation algorithm.

Reference:
    Gao, Y. and Wang, J. (2025). Freedom from inspiration! Achieving
    efficient metaheuristic optimization with Delta Plus. *Scientific Reports*.

The public paper describes DP as an inspiration-free metaheuristic based on
"Delta Operation", i.e. updates driven by changes between the current and
previous iteration rather than by a nature metaphor.  This implementation keeps
that design principle: the global best is tracked for reporting only and is not
used as an attraction point in the search equation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from atlas.core.base_algorithm import BaseAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_algorithm


@register_algorithm("dp", aliases=["dp_nfe"])
class DP(BaseAlgorithm):
    """Delta Plus optimiser.

    Args:
        problem: Optimisation problem instance.
        max_iter: Maximum iterations.
        pop_size: Population size.
        seed: Random seed.
        verbose: Print progress.
        delta_weight: Weight of the previous displacement term.
        peer_weight: Weight of the random peer-difference term.
        noise_weight: Initial random perturbation scale relative to bounds.
    """

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 500,
        pop_size: int = 30,
        seed: int = 42,
        verbose: bool = False,
        delta_weight: float = 0.7,
        peer_weight: float = 0.5,
        noise_weight: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            problem,
            max_iter=max_iter,
            pop_size=pop_size,
            seed=seed,
            verbose=verbose,
            delta_weight=delta_weight,
            peer_weight=peer_weight,
            noise_weight=noise_weight,
            **kwargs,
        )
        self.delta_weight = delta_weight
        self.peer_weight = peer_weight
        self.noise_weight = noise_weight

    def initialize(self) -> None:
        self.population = self.init_population()
        span = self.ub - self.lb
        self.previous_population = self._clip(
            self.population + self.rng.normal(0.0, 0.01 * span, size=(self.pop_size, self.dim))
        )
        self.fitness = self.evaluate_population(self.population)
        self._update_global_best()

    def iterate(self, iter_idx: int) -> float:
        total_it = max(self.max_iter - 1, 1) if self.max_iter > 0 else 500
        progress = min(iter_idx / total_it, 1.0)
        decay = 1.0 - progress
        span = self.ub - self.lb

        next_population = np.empty_like(self.population)
        next_fitness = self.fitness.copy()

        for i in range(self.pop_size):
            a, b = self.rng.choice(self.pop_size, size=2, replace=False)
            delta = self.population[i] - self.previous_population[i]
            peer_delta = self.population[a] - self.population[b]
            perturb = self.rng.normal(0.0, self.noise_weight * decay, size=self.dim) * span

            candidate = (
                self.population[i]
                + self.delta_weight * decay * delta
                + self.peer_weight * self.rng.random(self.dim) * peer_delta
                + perturb
            )
            candidate = self._clip(candidate)
            candidate_f = self.evaluate(candidate)

            if candidate_f <= self.fitness[i]:
                next_population[i] = candidate
                next_fitness[i] = candidate_f
            else:
                next_population[i] = self.population[i]

        self.previous_population = self.population.copy()
        self.population = next_population
        self.fitness = next_fitness
        return self.g_best_f


