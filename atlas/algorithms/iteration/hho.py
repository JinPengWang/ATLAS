"""Harris Hawks Optimization (HHO).

Reference:
    Heidari, A. A., Mirjalili, S., Faris, H., Aljarah, I., Mafarja, M., and Chen, H. (2019).
    Harris hawks optimization: Algorithm and applications.
    *Future Generation Computer Systems*, 97: 849–872.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import gamma

from atlas.core.base_algorithm import BaseAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_algorithm


def _levy_flight(rng: np.random.Generator, d: int) -> np.ndarray:
    beta = 1.5
    sigma = (
        gamma(1 + beta)
        * np.sin(np.pi * beta / 2)
        / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = rng.normal(0.0, sigma, size=d)
    v = rng.normal(0.0, 1.0, size=d)
    return u / (np.abs(v) ** (1 / beta) + np.finfo(float).eps)


@register_algorithm("hho", aliases=["hho_nfe"])
class HHO(BaseAlgorithm):
    """Harris Hawks Optimization.

    Args:
        problem: Optimisation problem instance.
        max_iter: Maximum iterations.
        pop_size: Population size.
        seed: Random seed.
        verbose: Print progress.
    """

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 500,
        pop_size: int = 30,
        seed: int = 42,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            problem,
            max_iter=max_iter,
            pop_size=pop_size,
            seed=seed,
            verbose=verbose,
            **kwargs,
        )

    def initialize(self) -> None:
        self.population = self.init_population()
        self.fitness = self.evaluate_population(self.population)
        self._update_global_best()

    def iterate(self, iter_idx: int) -> float:
        total_it = max(self.max_iter, 1) if self.max_iter > 0 else 500
        rabbit_x = self.g_best_x.copy()
        x_mean = np.mean(self.population, axis=0)

        for i in range(self.pop_size):
            e0 = 2.0 * self.rng.random() - 1.0  # [-1, 1]
            e = 2.0 * e0 * (1.0 - min(iter_idx / total_it, 1.0))
            j = 2.0 * (1.0 - self.rng.random())  # jump strength

            if abs(e) >= 1.0:
                # Exploration phase
                q = self.rng.random()
                if q >= 0.5:
                    rand_idx = self.rng.integers(0, self.pop_size)
                    x_rand = self.population[rand_idx]
                    r1 = self.rng.random()
                    r2 = self.rng.random()
                    x_new = x_rand - r1 * np.abs(x_rand - 2.0 * r2 * self.population[i])
                else:
                    r3 = self.rng.random()
                    r4 = self.rng.random()
                    x_new = (rabbit_x - x_mean) - r3 * (self.lb + r4 * (self.ub - self.lb))

                x_new = self._clip(x_new)
                f_new = self.evaluate(x_new)
                if f_new < self.fitness[i]:
                    self.population[i] = x_new
                    self.fitness[i] = f_new
            else:
                # Exploitation phase
                r = self.rng.random()
                if r >= 0.5 and abs(e) >= 0.5:
                    # Soft besiege
                    delta_x = rabbit_x - self.population[i]
                    x_new = rabbit_x - e * np.abs(j * rabbit_x - self.population[i])
                    x_new = self._clip(x_new)
                    f_new = self.evaluate(x_new)
                    if f_new < self.fitness[i]:
                        self.population[i] = x_new
                        self.fitness[i] = f_new
                elif r >= 0.5 and abs(e) < 0.5:
                    # Hard besiege
                    delta_x = rabbit_x - self.population[i]
                    x_new = rabbit_x - e * np.abs(delta_x)
                    x_new = self._clip(x_new)
                    f_new = self.evaluate(x_new)
                    if f_new < self.fitness[i]:
                        self.population[i] = x_new
                        self.fitness[i] = f_new
                elif r < 0.5 and abs(e) >= 0.5:
                    # Soft besiege with progressive rapid dives
                    y = rabbit_x - e * np.abs(j * rabbit_x - self.population[i])
                    y = self._clip(y)
                    f_y = self.evaluate(y)
                    if f_y < self.fitness[i]:
                        self.population[i] = y
                        self.fitness[i] = f_y
                    else:
                        lf = _levy_flight(self.rng, self.dim)
                        s = self.rng.random(self.dim)
                        z = y + s * lf
                        z = self._clip(z)
                        f_z = self.evaluate(z)
                        if f_z < self.fitness[i]:
                            self.population[i] = z
                            self.fitness[i] = f_z
                else:
                    # Hard besiege with progressive rapid dives
                    y = rabbit_x - e * np.abs(j * rabbit_x - x_mean)
                    y = self._clip(y)
                    f_y = self.evaluate(y)
                    if f_y < self.fitness[i]:
                        self.population[i] = y
                        self.fitness[i] = f_y
                    else:
                        lf = _levy_flight(self.rng, self.dim)
                        s = self.rng.random(self.dim)
                        z = y + s * lf
                        z = self._clip(z)
                        f_z = self.evaluate(z)
                        if f_z < self.fitness[i]:
                            self.population[i] = z
                            self.fitness[i] = f_z

        return self.g_best_f
