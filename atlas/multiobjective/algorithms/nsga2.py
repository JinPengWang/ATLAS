"""Non-dominated Sorting Genetic Algorithm II (NSGA-II).

Reference:
    Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
    A fast and elitist multiobjective genetic algorithm: NSGA-II.
    IEEE Transactions on Evolutionary Computation, 6(2), 182-197.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from atlas.multiobjective.base_mo_algorithm import BaseMOAlgorithm
from atlas.multiobjective.base_mo_problem import BaseMOProblem
from atlas.multiobjective.pareto import crowding_distance, non_dominated_sort


class NSGA2(BaseMOAlgorithm):
    """NSGA-II multi-objective evolutionary algorithm.

    Features:
    - Fast non-dominated sorting
    - Crowding distance diversity preservation
    - Simulated Binary Crossover (SBX)
    - Polynomial mutation
    - Elitist replacement
    """

    def __init__(
        self,
        problem: BaseMOProblem,
        max_iter: int = 150,
        pop_size: int = 100,
        crossover_prob: float = 0.9,
        crossover_eta: float = 20.0,
        mutation_prob: Optional[float] = None,
        mutation_eta: float = 20.0,
        seed: int = 42,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            problem=problem,
            max_iter=max_iter,
            pop_size=pop_size,
            seed=seed,
            verbose=verbose,
            **kwargs,
        )
        self.crossover_prob = crossover_prob
        self.crossover_eta = crossover_eta
        self.mutation_prob = mutation_prob if mutation_prob is not None else (1.0 / self.dim)
        self.mutation_eta = mutation_eta

    def initialize(self) -> None:
        self.population = self.init_population()
        self.objectives = self.evaluate_population(self.population)

    def _tournament_selection(
        self,
        ranks: np.ndarray,
        crowd_dists: np.ndarray,
    ) -> int:
        """Binary tournament selection based on rank and crowding distance."""
        i, j = self.rng.choice(self.pop_size, size=2, replace=False)
        if ranks[i] < ranks[j]:
            return i
        elif ranks[j] < ranks[i]:
            return j
        else:
            return i if crowd_dists[i] >= crowd_dists[j] else j

    def _sbx_crossover(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX)."""
        if self.rng.random() > self.crossover_prob:
            return p1.copy(), p2.copy()

        c1 = np.empty(self.dim, dtype=float)
        c2 = np.empty(self.dim, dtype=float)
        eta = self.crossover_eta

        for i in range(self.dim):
            if self.rng.random() <= 0.5:
                y1 = min(p1[i], p2[i])
                y2 = max(p1[i], p2[i])
                lb = self.lb[i]
                ub = self.ub[i]

                if abs(y1 - y2) > 1e-14:
                    # Beta calculation
                    rand = self.rng.random()
                    beta = 1.0 + (2.0 * (y1 - lb) / (y2 - y1))
                    alpha = 2.0 - (beta ** (-(eta + 1.0)))
                    beta_q = (
                        (rand * alpha) ** (1.0 / (eta + 1.0))
                        if rand <= (1.0 / alpha)
                        else (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                    )
                    c1[i] = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))

                    beta = 1.0 + (2.0 * (ub - y2) / (y2 - y1))
                    alpha = 2.0 - (beta ** (-(eta + 1.0)))
                    beta_q = (
                        (rand * alpha) ** (1.0 / (eta + 1.0))
                        if rand <= (1.0 / alpha)
                        else (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                    )
                    c2[i] = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
                else:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
            else:
                c1[i] = p1[i]
                c2[i] = p2[i]

        return self._clip(c1), self._clip(c2)

    def _polynomial_mutation(self, x: np.ndarray) -> np.ndarray:
        """Polynomial Mutation."""
        y = x.copy()
        eta = self.mutation_eta

        for i in range(self.dim):
            if self.rng.random() <= self.mutation_prob:
                lb = self.lb[i]
                ub = self.ub[i]
                val = y[i]
                delta1 = (val - lb) / (ub - lb + 1e-14)
                delta2 = (ub - val) / (ub - lb + 1e-14)
                rand = self.rng.random()
                mut_pow = 1.0 / (eta + 1.0)

                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val_mut = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                    delta_q = (val_mut**mut_pow) - 1.0
                else:
                    xy = 1.0 - delta2
                    val_mut = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                    delta_q = 1.0 - (val_mut**mut_pow)

                y[i] = val + delta_q * (ub - lb)

        return self._clip(y)

    def iterate(self, iter_idx: int) -> None:
        # 1. Rank & Crowding distance for current population
        fronts = non_dominated_sort(self.objectives)
        ranks = np.zeros(self.pop_size, dtype=int)
        crowd_dists = np.zeros(self.pop_size, dtype=float)

        for rank_idx, front in enumerate(fronts):
            cd = crowding_distance(self.objectives, front)
            for idx, d in zip(front, cd):
                ranks[idx] = rank_idx
                crowd_dists[idx] = d

        # 2. Selection + Crossover + Mutation -> Offspring Q (size N)
        offspring_pop = []
        for _ in range(self.pop_size // 2):
            p1_idx = self._tournament_selection(ranks, crowd_dists)
            p2_idx = self._tournament_selection(ranks, crowd_dists)
            c1, c2 = self._sbx_crossover(self.population[p1_idx], self.population[p2_idx])
            c1 = self._polynomial_mutation(c1)
            c2 = self._polynomial_mutation(c2)
            offspring_pop.extend([c1, c2])

        offspring_pop = np.array(offspring_pop[: self.pop_size])
        offspring_obj = self.evaluate_population(offspring_pop)

        # 3. Combine Parents (P) and Offspring (Q) -> R (size 2N)
        combined_pop = np.vstack([self.population, offspring_pop])
        combined_obj = np.vstack([self.objectives, offspring_obj])

        # 4. Environmental selection (Elitist sort)
        combined_fronts = non_dominated_sort(combined_obj)
        new_pop_indices = []

        for front in combined_fronts:
            if len(new_pop_indices) + len(front) <= self.pop_size:
                new_pop_indices.extend(front)
            else:
                # Need to select remaining from this front based on crowding distance
                remaining = self.pop_size - len(new_pop_indices)
                cd = crowding_distance(combined_obj, front)
                sorted_by_cd = np.argsort(-cd)  # descending crowding distance
                selected = [front[idx] for idx in sorted_by_cd[:remaining]]
                new_pop_indices.extend(selected)
                break

        self.population = combined_pop[new_pop_indices]
        self.objectives = combined_obj[new_pop_indices]
