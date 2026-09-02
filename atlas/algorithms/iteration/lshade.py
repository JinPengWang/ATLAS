"""L-SHADE: Linear population size reduction SHADE.

Reference:
    Tanabe, R. and Fukunaga, A. (2014). Improving the search performance of
    SHADE using linear population size reduction. In *Proceedings of the 2014
    IEEE Congress on Evolutionary Computation (CEC)*, pp. 1658-1665.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from atlas.core.base_algorithm import BaseAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_algorithm


@register_algorithm("lshade", aliases=["lshade_nfe"])
class LSHADE(BaseAlgorithm):
    """L-SHADE: Success-History based Adaptive DE with Linear Population
    Size Reduction.

    Args:
        problem: Optimisation problem instance.
        max_iter: Maximum number of iterations.
        pop_size: Initial population size (N_init).
        seed: Random seed for reproducibility.
        H: Size of the historical memory for F and CR.
        p_best_rate: Fraction of top individuals to use as p-best.
        archive_rate: Archive size multiplier relative to current N.
        verbose: Print progress.
    """

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 500,
        pop_size: int = 100,
        seed: int = 42,
        H: int = 5,
        p_best_rate: float = 0.11,
        archive_rate: float = 2.6,
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
        self.H = H
        self.p_best_rate = p_best_rate
        self.archive_rate = archive_rate
        self._N_init = pop_size
        self._N_min = 4

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        self.population = self.init_population()  # (N, dim)
        self.fitness = self.evaluate_population(self.population)

        # Historical memory for F and CR
        self._M_F: np.ndarray = np.full(self.H, 0.5)
        self._M_CR: np.ndarray = np.full(self.H, 0.5)
        self._mem_idx: int = 0

        # External archive
        self._archive: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Main iteration
    # ------------------------------------------------------------------
    def iterate(self, iter_idx: int) -> float:
        N = len(self.population)  # current pop size (may have been reduced)
        dim = self.dim

        new_population = self.population.copy()
        new_fitness = self.fitness.copy()

        S_F: List[float] = []
        S_CR: List[float] = []
        S_delta: List[float] = []

        # Determine number of p-best individuals
        p_num = max(2, round(self.p_best_rate * N))

        for i in range(N):
            # ----- Parameter generation -----
            ri = self.rng.integers(0, self.H)
            # Sample F from Cauchy distribution; regenerate until F > 0
            F = 0.0
            while F <= 0.0:
                F = float(self._M_F[ri] + 0.1 * np.tan(np.pi * (self.rng.random() - 0.5)))
                F = min(F, 1.0)

            # Sample CR from truncated normal [0, 1]
            CR = float(np.clip(self.rng.normal(self._M_CR[ri], 0.1), 0.0, 1.0))

            # ----- Mutation: current-to-p-best/1 -----
            # x_pbest: one of the top p_num individuals
            sorted_idx = np.argsort(self.fitness[:N])
            p_best_idx = sorted_idx[self.rng.integers(0, p_num)]
            x_pbest = self.population[p_best_idx]

            # x_r1: random from current population, != i
            candidates_r1 = [j for j in range(N) if j != i]
            r1 = candidates_r1[int(self.rng.integers(0, len(candidates_r1)))]
            x_r1 = self.population[r1]

            # x_r2: random from population ∪ archive, != i, != r1
            union = list(self.population) + self._archive
            union_len = len(union)
            # build valid indices excluding i and r1's position
            # For archive elements, they are at indices N..N+|archive|-1
            # We need to exclude the row matching i in population
            candidates_r2 = [
                j for j in range(union_len)
                if not (j == i or (j < N and j == r1))
            ]
            if len(candidates_r2) == 0:
                candidates_r2 = [j for j in range(union_len) if j != i]
            r2_sel = candidates_r2[int(self.rng.integers(0, len(candidates_r2)))]
            x_r2 = union[r2_sel]

            v = self.population[i] + F * (x_pbest - self.population[i]) + F * (x_r1 - x_r2)
            v = self._clip(v)

            # ----- Crossover -----
            mask = self.rng.random(dim) < CR
            if not mask.any():
                mask[self.rng.integers(0, dim)] = True
            u = np.where(mask, v, self.population[i])

            # ----- Selection -----
            f_u = self.evaluate(u)
            f_i = self.fitness[i]

            if f_u <= f_i:
                new_population[i] = u
                new_fitness[i] = f_u
                # Add old individual to archive
                self._archive.append(self.population[i].copy())
                S_F.append(F)
                S_CR.append(CR)
                S_delta.append(abs(f_i - f_u))

        self.population = new_population
        self.fitness = new_fitness

        # ----- Historical memory update -----
        if len(S_F) > 0:
            S_F_arr = np.array(S_F)
            S_CR_arr = np.array(S_CR)
            S_delta_arr = np.array(S_delta)
            weights = S_delta_arr / (S_delta_arr.sum() + 1e-300)

            # Lehmer mean for F
            lehmer_F = (weights * S_F_arr ** 2).sum() / ((weights * S_F_arr).sum() + 1e-300)
            self._M_F[self._mem_idx] = lehmer_F

            # Weighted mean for CR
            wmean_CR = (weights * S_CR_arr).sum()
            self._M_CR[self._mem_idx] = wmean_CR

            self._mem_idx = (self._mem_idx + 1) % self.H

        # ----- Archive trimming -----
        archive_max = max(1, round(self.archive_rate * N))
        if len(self._archive) > archive_max:
            # Randomly drop excess entries
            excess = len(self._archive) - archive_max
            remove_idx = self.rng.choice(len(self._archive), size=excess, replace=False)
            self._archive = [
                self._archive[j] for j in range(len(self._archive)) if j not in set(remove_idx.tolist())
            ]

        # ----- Linear population size reduction -----
        # Use NFE-based progress when running in NFE mode (max_iter <= 0)
        if self.max_iter > 0:
            _progress = min((iter_idx + 1) / max(self.max_iter, 1), 1.0)
        else:
            _max_nfe = getattr(self, "max_nfe", 0) or getattr(self, "_max_nfe", 0)
            _cur_nfe = getattr(self, "_nfe", 0)
            _progress = min(_cur_nfe / max(_max_nfe, 1), 1.0)
        N_new = max(
            self._N_min,
            round((self._N_min - self._N_init) * _progress + self._N_init),
        )
        if N_new < N:
            # Remove the worst (N - N_new) individuals
            n_remove = N - N_new
            worst_idx = np.argsort(self.fitness)[::-1][:n_remove]
            keep_mask = np.ones(N, dtype=bool)
            keep_mask[worst_idx] = False
            self.population = self.population[keep_mask]
            self.fitness = self.fitness[keep_mask]
            self.pop_size = len(self.population)

        return self.g_best_f
