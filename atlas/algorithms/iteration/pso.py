"""Particle Swarm Optimisation (PSO).

Reference:
    Kennedy, J. and Eberhart, R. (1995). Particle swarm optimization.
    In *Proceedings of ICNN'95 - International Conference on Neural Networks*,
    vol. 4, pp. 1942–1948.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from atlas.core.base_algorithm import BaseAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_algorithm


@register_algorithm("pso")
class PSO(BaseAlgorithm):
    """Particle Swarm Optimisation.

    Args:
        problem: Optimisation problem instance.
        max_iter: Maximum iterations.
        pop_size: Swarm size.
        seed: Random seed.
        verbose: Print progress.
        w: Inertia weight.
        c1: Cognitive (personal-best) coefficient.
        c2: Social (global-best) coefficient.
    """

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 500,
        pop_size: int = 30,
        seed: int = 42,
        verbose: bool = False,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(problem, max_iter, pop_size, seed, verbose, w=w, c1=c1, c2=c2, **kwargs)
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def get_name(self) -> str:
        return "PSO"

    def initialize(self) -> None:
        # Positions uniformly sampled in [lb, ub]
        self.population = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        # Velocities initialised to zero
        self.velocity = np.zeros((self.pop_size, self.dim))
        # Evaluate initial fitness
        self.fitness = np.array([
            self.problem.evaluate_with_penalty(self.population[i])
            for i in range(self.pop_size)
        ])
        # Personal bests
        self.p_best_x = self.population.copy()
        self.p_best_f = self.fitness.copy()
        # Global best
        self._update_global_best()

    def iterate(self, iter_idx: int) -> float:
        r1 = self.rng.random((self.pop_size, self.dim))
        r2 = self.rng.random((self.pop_size, self.dim))

        # Velocity update
        self.velocity = (
            self.w * self.velocity
            + self.c1 * r1 * (self.p_best_x - self.population)
            + self.c2 * r2 * (self.g_best_x - self.population)
        )

        # Position update
        self.population = self.population + self.velocity
        self.population = self._clip(self.population)

        # Evaluate
        for i in range(self.pop_size):
            f = self.problem.evaluate_with_penalty(self.population[i])
            self.fitness[i] = f
            # Update personal best
            if f < self.p_best_f[i]:
                self.p_best_f[i] = f
                self.p_best_x[i] = self.population[i].copy()
                # Update global best
                if f < self.g_best_f:
                    self.g_best_f = f
                    self.g_best_x = self.population[i].copy()

        return self.g_best_f
