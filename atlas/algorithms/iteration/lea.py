"""Love Evolution Algorithm (LEA)."""

from __future__ import annotations

from typing import Any

import numpy as np

from atlas.core.base_algorithm import BaseAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_algorithm


@register_algorithm("lea")
class LEA(BaseAlgorithm):
    """Love Evolution Algorithm."""

    def get_name(self) -> str:
        return "LEA"

    def initialize(self) -> None:
        if self.pop_size % 2 != 0:
            self.pop_size += 1
        self.population = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([
            self.problem.evaluate_with_penalty(self.population[i])
            for i in range(self.pop_size)
        ])
        self._update_global_best()

    def _gap_probability(self, f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
        p = (0.5 + self.rng.random(len(f1))) * (f1 - f2) ** 2
        return p / (np.max(p) + np.min(p) + np.finfo(float).eps)

    def _promotion(self, p1: np.ndarray, p2: np.ndarray, g: float) -> tuple[np.ndarray, np.ndarray]:
        c1 = p1.copy()
        c2 = p2.copy()
        span = self.ub - self.lb + np.finfo(float).eps
        for j in range(self.dim):
            r3 = self.rng.integers(0, self.dim)
            r4 = self.rng.integers(0, self.dim)
            s1 = (3.0 * self.rng.random() - 1.5) * (p1[j] / (p2[j] + np.finfo(float).eps))
            s2 = (3.0 * self.rng.random() - 1.5) * (p2[j] / (p1[j] + np.finfo(float).eps))
            d = 0.5 * (p1[r3] / span[r3] + p2[r4] / span[r4])
            c1[j] = self.g_best_x[j] + s1 * g * d
            c2[j] = self.g_best_x[j] + s2 * g * d
        return c1, c2

    def iterate(self, iter_idx: int) -> float:
        order = self.rng.permutation(self.pop_size)
        pairs = np.vstack((order[: self.pop_size // 2], order[self.pop_size // 2 :]))
        p = self._gap_probability(self.fitness[pairs[0]], self.fitness[pairs[1]])
        h = 0.7 * (1.0 - iter_idx / max(self.max_iter, 1))
        g = np.sum(np.sqrt(np.sum((self.population - self.g_best_x) ** 2, axis=1)) / self.pop_size) / self.dim
        g += np.finfo(float).eps

        for pair_idx in range(self.pop_size // 2):
            idx1, idx2 = pairs[:, pair_idx]
            x1 = self.population[idx1].copy()
            x2 = self.population[idx2].copy()

            if p[pair_idx] < 0.5:
                c1 = np.empty(self.dim)
                c2 = np.empty(self.dim)
                for j in range(self.dim):
                    C1 = self.g_best_x[j] * x1[j]
                    C2 = self.g_best_x[j] ** 2 + x1[j] * x2[j]
                    C3 = self.g_best_x[j] * x2[j]
                    X1 = np.sqrt(abs(C2 - C1))
                    X2 = np.sqrt(abs(C2 - C3))
                    c1[j] = self.rng.random() * x1[j] + self.rng.normal() * X1
                    c2[j] = self.rng.random() * x2[j] + self.rng.normal() * X2
                c1 = self._clip(c1)
                c2 = self._clip(c2)
                f1 = self.problem.evaluate_with_penalty(c1)
                f2 = self.problem.evaluate_with_penalty(c2)
                self.population[idx1], self.population[idx2] = c1, c2
                self.fitness[idx1], self.fitness[idx2] = f1, f2
                self._update_global_best()

                p[pair_idx] = (
                    (self.rng.random() + 0.5)
                    * p[pair_idx]
                    * np.sum(np.abs(self.population[idx1] - self.population[idx2]))
                    / (self.dim * g)
                )
                if p[pair_idx] < 0.5:
                    s = self.population[idx1] * self.population[idx2]
                    s = (s - np.min(s)) / (np.max(s) - np.min(s) + np.finfo(float).eps) + h
                    c1 = self.g_best_x + self.rng.normal(size=self.dim) * g * s
                    c2 = self.g_best_x + self.rng.normal(size=self.dim) * g * s
                else:
                    c1, c2 = self._promotion(self.population[idx1], self.population[idx2], g)
            else:
                c1, c2 = self._promotion(x1, x2, g)

            c1 = self._clip(c1)
            c2 = self._clip(c2)
            self.population[idx1], self.population[idx2] = c1, c2
            self.fitness[idx1] = self.problem.evaluate_with_penalty(c1)
            self.fitness[idx2] = self.problem.evaluate_with_penalty(c2)
            self._update_global_best()

        return self.g_best_f

