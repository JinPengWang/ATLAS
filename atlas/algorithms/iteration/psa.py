"""PID-based Search Algorithm (PSA)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import gamma

from atlas.core.base_algorithm import BaseAlgorithm
from atlas.core.base_problem import BaseProblem
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


@register_algorithm("psa")
class PSA(BaseAlgorithm):
    """PID-based Search Algorithm."""

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 500,
        pop_size: int = 30,
        seed: int = 42,
        verbose: bool = False,
        kp: float = 1.0,
        ki: float = 0.5,
        kd: float = 1.2,
        **kwargs: Any,
    ) -> None:
        super().__init__(problem, max_iter, pop_size, seed, verbose, kp=kp, ki=ki, kd=kd, **kwargs)
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def get_name(self) -> str:
        return "PSA"

    def initialize(self) -> None:
        self.population = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([
            self.problem.evaluate_with_penalty(self.population[i])
            for i in range(self.pop_size)
        ])
        self._update_global_best()
        self.target_x = self.g_best_x.copy()
        self.target_f = self.g_best_f
        self.ek = self.target_x - self.population
        self.ek_1 = self.ek.copy()
        self.ek_2 = self.ek.copy()

    def iterate(self, iter_idx: int) -> float:
        self._update_global_best()
        new_target_x = self.g_best_x.copy()
        new_target_f = self.g_best_f

        if iter_idx > 0:
            self.ek_2 = self.ek_1.copy()
            self.ek_1 = self.ek + new_target_x - self.target_x
            self.ek = new_target_x - self.population
            self.target_x = new_target_x
            self.target_f = new_target_f

        t = iter_idx + 1
        log_t = np.log(max(self.max_iter, 2))
        a = (np.log(self.max_iter - t + 2) / log_t) ** 2
        out0 = (
            np.cos(1.0 - t / max(self.max_iter, 1))
            + a * self.rng.random((self.pop_size, self.dim)) * _levy_flight(self.rng, self.pop_size, self.dim)
        ) * self.ek
        pid = (
            self.rng.random((self.pop_size, 1)) * self.kp * (self.ek - self.ek_1)
            + self.rng.random((self.pop_size, 1)) * self.ki * self.ek
            + self.rng.random((self.pop_size, 1)) * self.kd * (self.ek - 2 * self.ek_1 + self.ek_2)
        )
        r = self.rng.random((self.pop_size, 1)) * np.cos(t / max(self.max_iter, 1))
        self.population = self._clip(self.population + r * pid + (1.0 - r) * out0)

        for i in range(self.pop_size):
            self.fitness[i] = self.problem.evaluate_with_penalty(self.population[i])
        self._update_global_best()
        return self.g_best_f

