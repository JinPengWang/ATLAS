"""Unit tests for constraint handling strategies."""

import numpy as np
import pytest

from atlas.constraints.handlers import (
    DebFeasibilityHandler,
    EpsilonConstraintHandler,
    StochasticRankingHandler,
)


class TestDebFeasibility:
    def setup_method(self):
        self.h = DebFeasibilityHandler()

    def test_feasible_beats_infeasible(self):
        # x1 feasible (v=0), x2 infeasible (v=1.0)
        assert self.h.compare(None, 10.0, 0.0, None, 5.0, 1.0) == -1
        assert self.h.compare(None, 5.0, 1.0, None, 10.0, 0.0) == 1

    def test_better_fitness_wins_among_feasible(self):
        assert self.h.compare(None, 3.0, 0.0, None, 5.0, 0.0) == -1
        assert self.h.compare(None, 5.0, 0.0, None, 3.0, 0.0) == 1

    def test_less_violation_wins_among_infeasible(self):
        assert self.h.compare(None, 10.0, 0.5, None, 5.0, 2.0) == -1
        assert self.h.compare(None, 5.0, 2.0, None, 10.0, 0.5) == 1

    def test_equal_feasible_returns_zero(self):
        assert self.h.compare(None, 5.0, 0.0, None, 5.0, 0.0) == 0

    def test_sort_population(self):
        solutions = [np.array([i]) for i in range(4)]
        fitnesses = [10.0, 20.0, 5.0, 15.0]
        violations = [0.0, 0.0, 1.0, 0.5]
        # Feasible: sol 0 (f=10, v=0), sol 1 (f=20, v=0) -> 0, 1
        # Infeasible: sol 3 (v=0.5), sol 2 (v=1.0) -> 3, 2
        # Expected order: [0, 1, 3, 2]
        sorted_idx = self.h.sort_population(solutions, fitnesses, violations)
        assert sorted_idx == [0, 1, 3, 2]


class TestEpsilonConstraint:
    def test_epsilon_decreases_over_time(self):
        h = EpsilonConstraintHandler(epsilon0=1.0, cp=2.0, T_max=100)
        eps0 = h.epsilon
        h.update_epsilon(50)
        assert h.epsilon < eps0

    def test_at_T_max_epsilon_is_zero(self):
        h = EpsilonConstraintHandler(epsilon0=1.0, cp=2.0, T_max=100)
        h.update_epsilon(100)
        assert h.epsilon == pytest.approx(0.0, abs=1e-10)

    def test_within_epsilon_treated_as_feasible(self):
        h = EpsilonConstraintHandler(epsilon0=2.0, cp=1.0, T_max=100)
        h.update_epsilon(0)  # epsilon = 2.0
        # Both within epsilon (0.5 <= 2.0): compare by fitness
        assert h.compare(None, 3.0, 0.5, None, 5.0, 0.5) == -1


class TestStochasticRanking:
    def test_sort_population_returns_all_indices(self):
        h = StochasticRankingHandler(Pf=0.45, seed=42)
        n = 10
        solutions = list(range(n))
        fitnesses = np.random.rand(n)
        violations = np.random.rand(n)
        sorted_idx = h.sort_population(solutions, fitnesses, violations)
        assert sorted(sorted_idx) == list(range(n))

    def test_pf_zero_sorts_by_violation_only(self):
        """With Pf=0, always compare by violation when any infeasible."""
        h = StochasticRankingHandler(Pf=0.0, seed=42)
        solutions = [0, 1, 2]
        fitnesses = np.array([1.0, 0.5, 3.0])
        violations = np.array([2.0, 0.0, 1.0])  # order by viol: 1 (v=0), 2 (v=1), 0 (v=2)
        sorted_idx = h.sort_population(solutions, fitnesses, violations)
        assert sorted_idx[0] == 1
        assert sorted_idx[1] == 2
        assert sorted_idx[2] == 0
