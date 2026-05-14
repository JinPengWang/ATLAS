"""Philoponella Prominens Optimizer (PPO) -- NFE-based variant."""

from __future__ import annotations

import numpy as np
from scipy.special import gamma

from atlas.algorithms.nfe.base_nfe_algorithm import BaseNFEAlgorithm
from atlas.utils.registry import register_algorithm


def _levy_flight(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    beta = 1.5
    sigma = (
        gamma(1 + beta)
        * np.sin(np.pi * beta / 2)
        / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = rng.normal(0.0, sigma, size=(n, d))
    v = rng.normal(0.0, 1.0, size=(n, d))
    return u / (np.abs(v) ** (1 / beta) + np.finfo(float).eps)


@register_algorithm("ppo_nfe")
class PPO_NFE(BaseNFEAlgorithm):
    """Philoponella Prominens Optimizer with NFE-based stopping."""

    def get_name(self) -> str:
        return "PPO_NFE"

    def initialize(self) -> None:
        self.population = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([self._evaluate(self.population[i]) for i in range(self.pop_size)])
        self.memory_x = self.population.copy()
        self.memory_f = self.fitness.copy()
        self.distance = np.zeros(self.pop_size)
        self._update_global_best()

    def iterate(self, iter_idx: int) -> float:
        progress = min(self._nfe / max(self.max_nfe, 1), 1.0)
        female_x = self.memory_x[self.rng.permutation(self.pop_size)]
        strong = np.max(self.fitness) + np.min(self.fitness) - self.fitness
        strong = strong / (np.max(strong) + np.finfo(float).eps)
        dd = np.mean(np.sqrt(np.sum((self.population - female_x) ** 2, axis=1)) / self.dim)
        for i in range(self.pop_size):
            direction = strong[i] * np.abs(self.population[i] - female_x[i])
            self.population[i] = female_x[i] + np.cos(self.rng.random(self.dim) * np.pi) * direction
            self.distance[i] = np.linalg.norm(self.population[i] - female_x[i])
        dmean = np.mean(self.distance) * ((1.0 - progress) + 0.5)
        for i in range(self.pop_size):
            if self.distance[i] < dmean:
                female_x[i] = female_x[i] + self.rng.random() * strong[i] * (self.population[i] - female_x[i])
                self.population[i] = female_x[i] + _levy_flight(self.rng, 1, self.dim)[0] * dd * np.exp(1.0 - progress)
            else:
                self.population[i] = self.g_best_x + np.cos(self.rng.random() * np.pi) * (
                    self.population[i] - self.g_best_x
                )
        self.population = self._clip(self.population)
        for i in range(self.pop_size):
            if self._nfe >= self.max_nfe:
                break
            self.fitness[i] = self._evaluate(self.population[i])
        improved = self.fitness < self.memory_f
        self.memory_f[improved] = self.fitness[improved]
        self.memory_x[improved] = self.population[improved]
        self._update_global_best()
        return self.g_best_f

