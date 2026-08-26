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


@register_algorithm("psa", aliases=["psa_nfe"])
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
        super().__init__(problem, max_iter=max_iter, pop_size=pop_size, seed=seed, verbose=verbose, kp=kp, ki=ki, kd=kd, **kwargs)
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def initialize(self) -> None:
        self.population = self.init_population()
        self.fitness = self.evaluate_population(self.population)
        self._update_global_best()
        self.target_x = self.g_best_x.copy()
        self.target_f = self.g_best_f
        self.ek = self.target_x - self.population
        self.ek_1 = self.ek.copy()
        self.ek_2 = self.ek.copy()

    def iterate(self, iter_idx: int) -> float:
        new_target_x = self.g_best_x.copy()
        new_target_f = self.g_best_f

        if iter_idx > 0:
            self.ek_2 = self.ek_1.copy()
            self.ek_1 = self.ek + new_target_x - self.target_x
            self.ek = new_target_x - self.population
            self.target_x = new_target_x
            self.target_f = new_target_f

        t = iter_idx + 1
        total_it = max(self.max_iter, 2) if self.max_iter > 0 else 500
        log_t = np.log(total_it)
        a = (np.log(max(total_it - t + 2, 2)) / log_t) ** 2
        out0 = (
            np.cos(1.0 - min(t / total_it, 1.0))
            + a * self.rng.random((self.pop_size, self.dim)) * _levy_flight(self.rng, self.pop_size, self.dim)
        ) * self.ek
        pid = (
            self.rng.random((self.pop_size, 1)) * self.kp * (self.ek - self.ek_1)
            + self.rng.random((self.pop_size, 1)) * self.ki * self.ek
            + self.rng.random((self.pop_size, 1)) * self.kd * (self.ek - 2 * self.ek_1 + self.ek_2)
        )
        r = self.rng.random((self.pop_size, 1)) * np.cos(min(t / total_it, 1.0))
        self.population = self._clip(self.population + r * pid + (1.0 - r) * out0)

        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(self.population[i])
        return self.g_best_f


