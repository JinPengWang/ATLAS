"""Constraint handling strategies for metaheuristics and engineering optimization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np


class ConstraintHandler(ABC):
    """Abstract base class for constraint handling strategies."""

    @abstractmethod
    def compare(
        self,
        x1: Optional[np.ndarray],
        f1: float,
        v1: float,
        x2: Optional[np.ndarray],
        f2: float,
        v2: float,
    ) -> int:
        """Compare two candidate solutions.

        Args:
            x1: Decision vector for solution 1.
            f1: Objective fitness value for solution 1 (minimization).
            v1: Total constraint violation for solution 1 (>= 0).
            x2: Decision vector for solution 2.
            f2: Objective fitness value for solution 2 (minimization).
            v2: Total constraint violation for solution 2 (>= 0).

        Returns:
            -1 if solution 1 is preferred,
            +1 if solution 2 is preferred,
             0 if both solutions are considered equal.
        """

    def sort_population(
        self,
        solutions: Sequence[Any],
        fitnesses: Sequence[float],
        violations: Sequence[float],
    ) -> List[int]:
        """Sort population indices from best to worst according to the constraint strategy.

        Args:
            solutions: Sequence of decision vectors or objects.
            fitnesses: Sequence of objective values.
            violations: Sequence of constraint violations.

        Returns:
            List of sorted indices (best solution index first).
        """
        n = len(solutions)
        indices = list(range(n))
        # Default bubble sort using pairwise compare
        for i in range(n):
            swapped = False
            for j in range(n - 1):
                idx1 = indices[j]
                idx2 = indices[j + 1]
                cmp = self.compare(
                    solutions[idx1],
                    float(fitnesses[idx1]),
                    float(violations[idx1]),
                    solutions[idx2],
                    float(fitnesses[idx2]),
                    float(violations[idx2]),
                )
                if cmp > 0:  # idx2 is better than idx1
                    indices[j], indices[j + 1] = indices[j + 1], indices[j]
                    swapped = True
            if not swapped:
                break
        return indices


class DebFeasibilityHandler(ConstraintHandler):
    """Deb's Feasibility Rules (Deb, 2000).

    Selection rules:
    1. Any feasible solution is preferred to any infeasible solution.
    2. Among two feasible solutions, the one having a smaller objective function value is preferred.
    3. Among two infeasible solutions, the one having a smaller constraint violation is preferred.

    References:
        Deb, K. (2000). An efficient constraint handling method for genetic algorithms.
        Computer Methods in Applied Mechanics and Engineering, 186(2-4), 311-338.
    """

    FEASIBILITY_TOL: float = 1e-8

    def __init__(self, tol: float = 1e-8) -> None:
        self.tol = tol

    def compare(
        self,
        x1: Optional[np.ndarray],
        f1: float,
        v1: float,
        x2: Optional[np.ndarray],
        f2: float,
        v2: float,
    ) -> int:
        feas1 = v1 <= self.tol
        feas2 = v2 <= self.tol

        if feas1 and not feas2:
            return -1
        if not feas1 and feas2:
            return 1
        if feas1 and feas2:
            if f1 < f2:
                return -1
            if f2 < f1:
                return 1
            return 0
        # Both infeasible
        if v1 < v2:
            return -1
        if v2 < v1:
            return 1
        if f1 < f2:
            return -1
        if f2 < f1:
            return 1
        return 0


class EpsilonConstraintHandler(ConstraintHandler):
    """Adaptive Epsilon-Constraint Relaxation Method.

    Treats solutions with constraint violation <= epsilon(t) as feasible.
    Epsilon decreases from epsilon0 to 0 over iterations according to:
        epsilon(t) = epsilon0 * (1 - t / T_max) ** cp

    References:
        Takahama, T., & Sakai, S. (2006). Constrained optimization by the epsilon
        constrained differential evolution with an archive and gradient-based mutation.
        In IEEE Congress on Evolutionary Computation (pp. 936-943).
    """

    def __init__(
        self,
        epsilon0: float = 1.0,
        cp: float = 2.0,
        T_max: int = 500,
        tol: float = 1e-8,
    ) -> None:
        self.epsilon0 = float(epsilon0)
        self.cp = float(cp)
        self.T_max = max(1, int(T_max))
        self.tol = float(tol)
        self.epsilon: float = self.epsilon0

    def update_epsilon(self, t: int) -> float:
        """Update current relaxation threshold epsilon based on iteration t."""
        if t >= self.T_max:
            self.epsilon = 0.0
        else:
            ratio = max(0.0, 1.0 - t / self.T_max)
            self.epsilon = float(self.epsilon0 * (ratio**self.cp))
        return self.epsilon

    def compare(
        self,
        x1: Optional[np.ndarray],
        f1: float,
        v1: float,
        x2: Optional[np.ndarray],
        f2: float,
        v2: float,
    ) -> int:
        threshold = max(self.epsilon, self.tol)
        feas1 = v1 <= threshold
        feas2 = v2 <= threshold

        if feas1 and not feas2:
            return -1
        if not feas1 and feas2:
            return 1
        if feas1 and feas2:
            if f1 < f2:
                return -1
            if f2 < f1:
                return 1
            return 0
        # Both infeasible
        if v1 < v2:
            return -1
        if v2 < v1:
            return 1
        if f1 < f2:
            return -1
        if f2 < f1:
            return 1
        return 0


class StochasticRankingHandler(ConstraintHandler):
    """Stochastic Ranking (Runarsson & Yao, 2000).

    Balances objective fitness and constraint violation by comparing based on
    objective function with probability Pf, and based on constraint violation
    with probability (1 - Pf) when at least one solution is infeasible.

    References:
        Runarsson, T. P., & Yao, X. (2000). Stochastic ranking for constrained
        evolutionary optimization. IEEE Transactions on Evolutionary Computation, 4(3), 284-294.
    """

    def __init__(self, Pf: float = 0.45, seed: int = 42, tol: float = 1e-8) -> None:
        self.Pf = float(Pf)
        self.tol = float(tol)
        self.rng = np.random.default_rng(seed)

    def compare(
        self,
        x1: Optional[np.ndarray],
        f1: float,
        v1: float,
        x2: Optional[np.ndarray],
        f2: float,
        v2: float,
    ) -> int:
        feas1 = v1 <= self.tol
        feas2 = v2 <= self.tol

        if feas1 and feas2:
            if f1 < f2:
                return -1
            if f2 < f1:
                return 1
            return 0

        # At least one infeasible
        if self.rng.random() < self.Pf:
            if f1 < f2:
                return -1
            if f2 < f1:
                return 1
            return 0
        else:
            if v1 < v2:
                return -1
            if v2 < v1:
                return 1
            if f1 < f2:
                return -1
            if f2 < f1:
                return 1
            return 0

    def sort_population(
        self,
        solutions: Sequence[Any],
        fitnesses: Sequence[float],
        violations: Sequence[float],
    ) -> List[int]:
        """Stochastic ranking bubble sort implementation."""
        n = len(solutions)
        indices = list(range(n))
        fits = np.asarray(fitnesses, dtype=float)
        viols = np.asarray(violations, dtype=float)

        for _ in range(n):
            swapped = False
            for j in range(n - 1):
                idx1 = indices[j]
                idx2 = indices[j + 1]
                v1, v2 = viols[idx1], viols[idx2]
                f1, f2 = fits[idx1], fits[idx2]
                u = self.rng.random()

                # Rule: if both feasible or u < Pf -> compare by fitness
                if (v1 <= self.tol and v2 <= self.tol) or (u < self.Pf):
                    if f1 > f2:
                        indices[j], indices[j + 1] = idx2, idx1
                        swapped = True
                else:
                    if v1 > v2:
                        indices[j], indices[j + 1] = idx2, idx1
                        swapped = True
            if not swapped:
                break
        return indices
