"""Delta Plus (DP) optimisation algorithm -- NFE-based variant."""

from __future__ import annotations

from typing import Any

import numpy as np

from atlas.algorithms.nfe.base_nfe_algorithm import BaseNFEAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_algorithm


@register_algorithm("dp_nfe")
class DP_NFE(BaseNFEAlgorithm):
    """Delta Plus optimiser with function-evaluation-based stopping."""

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 0,
        max_nfe: int = 10000,
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
            max_iter,
            max_nfe,
            pop_size,
            seed,
            verbose,
            delta_weight=delta_weight,
            peer_weight=peer_weight,
            noise_weight=noise_weight,
            **kwargs,
        )
        self.delta_weight = delta_weight
        self.peer_weight = peer_weight
        self.noise_weight = noise_weight

    def get_name(self) -> str:
        return "DP_NFE"

    def initialize(self) -> None:
        self.population = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        span = self.ub - self.lb
        self.previous_population = self.population + self.rng.normal(
            0.0, 0.01 * span, size=(self.pop_size, self.dim)
        )
        self.previous_population = self._clip(self.previous_population)
        self.fitness = np.array([
            self._evaluate(self.population[i])
            for i in range(self.pop_size)
        ])
        self._update_global_best()

    def iterate(self, iter_idx: int) -> float:
        progress = min(self._nfe / max(self.max_nfe, 1), 1.0)
        decay = 1.0 - progress
        span = self.ub - self.lb

        next_population = np.empty_like(self.population)
        next_fitness = self.fitness.copy()

        for i in range(self.pop_size):
            if self._nfe >= self.max_nfe:
                next_population[i:] = self.population[i:]
                break

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
            candidate_f = self._evaluate(candidate)

            if candidate_f <= self.fitness[i]:
                next_population[i] = candidate
                next_fitness[i] = candidate_f
            else:
                next_population[i] = self.population[i]

        self.previous_population = self.population.copy()
        self.population = next_population
        self.fitness = next_fitness
        self._update_global_best()
        return self.g_best_f

