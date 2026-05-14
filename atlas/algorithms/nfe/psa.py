"""PID-based Search Algorithm (PSA) -- NFE-based variant."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import gamma

from atlas.algorithms.nfe.base_nfe_algorithm import BaseNFEAlgorithm
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


@register_algorithm("psa_nfe")
class PSA_NFE(BaseNFEAlgorithm):
    """PID-based Search Algorithm with NFE-based stopping."""

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 0,
        max_nfe: int = 10000,
        pop_size: int = 30,
        seed: int = 42,
        verbose: bool = False,
        kp: float = 1.0,
        ki: float = 0.5,
        kd: float = 1.2,
        **kwargs: Any,
    ) -> None:
        super().__init__(problem, max_iter, max_nfe, pop_size, seed, verbose, kp=kp, ki=ki, kd=kd, **kwargs)
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def get_name(self) -> str:
        return "PSA_NFE"

    def initialize(self) -> None:
        self.population = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([self._evaluate(self.population[i]) for i in range(self.pop_size)])
        self._update_global_best()
        self.target_x = self.g_best_x.copy()
        self.ek = self.target_x - self.population
        self.ek_1 = self.ek.copy()
        self.ek_2 = self.ek.copy()

    def iterate(self, iter_idx: int) -> float:
        self._update_global_best()
        new_target_x = self.g_best_x.copy()
        self.ek_2 = self.ek_1.copy()
        self.ek_1 = self.ek + new_target_x - self.target_x
        self.ek = new_target_x - self.population
        self.target_x = new_target_x

        progress = min(self._nfe / max(self.max_nfe, 1), 1.0)
        a = (np.log((1.0 - progress) * max(self.max_nfe, 2) + 1.0) / np.log(max(self.max_nfe, 2))) ** 2
        out0 = (
            np.cos(1.0 - progress)
            + a * self.rng.random((self.pop_size, self.dim)) * _levy_flight(self.rng, self.pop_size, self.dim)
        ) * self.ek
        pid = (
            self.rng.random((self.pop_size, 1)) * self.kp * (self.ek - self.ek_1)
            + self.rng.random((self.pop_size, 1)) * self.ki * self.ek
            + self.rng.random((self.pop_size, 1)) * self.kd * (self.ek - 2 * self.ek_1 + self.ek_2)
        )
        r = self.rng.random((self.pop_size, 1)) * np.cos(progress)
        self.population = self._clip(self.population + r * pid + (1.0 - r) * out0)

        for i in range(self.pop_size):
            if self._nfe >= self.max_nfe:
                break
            self.fitness[i] = self._evaluate(self.population[i])
        self._update_global_best()
        return self.g_best_f

