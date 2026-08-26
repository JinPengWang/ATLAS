"""Ant Colony Optimisation for continuous domains (ACO-R).

Reference:
    Socha, K. and Dorigo, M. (2008). Ant colony optimization for continuous
    domains. *European Journal of Operational Research*, 185(3): 1155–1173.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from atlas.core.base_algorithm import BaseAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_algorithm


@register_algorithm("aco", aliases=["aco_nfe"])
class ACO(BaseAlgorithm):
    """Ant Colony Optimisation for continuous domains (ACO-R variant).

    Uses a Gaussian kernel mixture to sample new solutions and a
    rank-based weight update.

    Args:
        problem: Optimisation problem instance.
        max_iter: Maximum iterations.
        pop_size: Number of ants (solutions per iteration).  The internal
            archive size is ``k * n_ants``.
        seed: Random seed.
        verbose: Print progress.
        n_ants: Number of ants per iteration.
        k: Number of Gaussian kernels (archive size = k * n_ants).
        q: Locality parameter for weight computation (smaller → more
            exploitation of the best solutions).
        xi: Pheromone evaporation / contraction factor.
    """

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 500,
        pop_size: int = 20,
        seed: int = 42,
        verbose: bool = False,
        n_ants: int = 20,
        q: float = 0.1,
        xi: float = 0.85,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            problem, max_iter=max_iter, pop_size=n_ants, seed=seed, verbose=verbose,
            n_ants=n_ants, q=q, xi=xi, **kwargs,
        )
        self.n_ants = n_ants
        self.q = q
        self.xi = xi
        self.pop_size = n_ants  # consistency with base class
        # Archive size (number of stored Gaussian kernels)
        self.k = max(10, n_ants)

    def initialize(self) -> None:
        # Build initial archive by random sampling
        archive_size = self.k
        self.archive_x = self.init_population(archive_size)
        self.archive_f = self.evaluate_population(self.archive_x)
        # Sort archive by fitness (ascending)
        order = np.argsort(self.archive_f)
        self.archive_x = self.archive_x[order]
        self.archive_f = self.archive_f[order]

        self.population = self.archive_x.copy()
        self.fitness = self.archive_f.copy()

    def iterate(self, iter_idx: int) -> float:
        archive_size = self.k

        # Compute weights: w_i ∝ exp(-q * k * (rank_i / k)^2)
        ranks = np.arange(archive_size)
        weights = np.exp(-self.q * archive_size * (ranks / archive_size) ** 2)
        weights /= weights.sum()

        # Compute per-dimension std for each kernel:
        # σ_j^i = xi * sum_{e=1}^k |archive_x[e, j] - archive_x[i, j]| / (k - 1)
        sigma = np.zeros((archive_size, self.dim))
        for i in range(archive_size):
            diffs = np.abs(self.archive_x - self.archive_x[i])
            sigma[i] = self.xi * np.sum(diffs, axis=0) / max(archive_size - 1, 1)
        sigma = np.clip(sigma, 1e-10, None)  # avoid zero std

        # Generate new solutions
        new_solutions = np.empty((self.n_ants, self.dim))
        new_fitness = np.empty(self.n_ants)

        for m in range(self.n_ants):
            # Select a kernel by roulette wheel
            kernel_idx = self.rng.choice(archive_size, p=weights)
            # Sample from the selected Gaussian kernel
            x_new = self.rng.normal(self.archive_x[kernel_idx], sigma[kernel_idx])
            x_new = self._clip(x_new)
            new_solutions[m] = x_new
            new_fitness[m] = self.evaluate(x_new)

        # Merge archive + new solutions, keep top-k
        merged_x = np.vstack([self.archive_x, new_solutions])
        merged_f = np.concatenate([self.archive_f, new_fitness])
        order = np.argsort(merged_f)[:archive_size]
        self.archive_x = merged_x[order].copy()
        self.archive_f = merged_f[order].copy()

        self.population = self.archive_x.copy()
        self.fitness = self.archive_f.copy()

        return self.g_best_f

