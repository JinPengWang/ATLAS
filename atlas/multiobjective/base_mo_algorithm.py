"""Abstract base class for multi-objective optimization algorithms."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from atlas.multiobjective.base_mo_problem import BaseMOProblem
from atlas.multiobjective.mo_result import MOResult
from atlas.multiobjective.pareto import non_dominated_sort


class BaseMOAlgorithm(ABC):
    """Abstract base class that every multi-objective algorithm subclasses.

    Subclasses must implement :meth:`initialize` and :meth:`iterate`.

    Args:
        problem: The multi-objective problem to solve.
        max_iter: Maximum number of generations / iterations.
        pop_size: Population size.
        seed: Random seed.
        verbose: Whether to log progress.
    """

    def __init__(
        self,
        problem: BaseMOProblem,
        max_iter: int = 250,
        pop_size: int = 100,
        seed: int = 42,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        self.problem = problem
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.seed = seed
        self.verbose = verbose
        self.extra_params = kwargs

        self.rng = np.random.default_rng(seed)
        self.dim = problem.get_dim()
        self.n_objectives = problem.get_num_objectives()
        self.lb, self.ub = problem.get_bounds()

        self.population: np.ndarray = np.empty((0, self.dim))
        self.objectives: np.ndarray = np.empty((0, self.n_objectives))
        self._nfe: int = 0

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate a single candidate solution.

        Args:
            x: Decision vector of shape (dim,).

        Returns:
            Objective vector of shape (n_objectives,).
        """
        self._nfe += 1
        return self.problem.evaluate_with_penalty(x)

    def evaluate_population(self, pop: np.ndarray) -> np.ndarray:
        """Evaluate an array of candidate solutions.

        Args:
            pop: 2-D array of shape (N, dim).

        Returns:
            2-D array of shape (N, n_objectives).
        """
        N = len(pop)
        objs = np.empty((N, self.n_objectives), dtype=float)
        for i in range(N):
            objs[i] = self.evaluate(pop[i])
        return objs

    def init_population(self, size: Optional[int] = None) -> np.ndarray:
        """Sample a uniform initial population."""
        n = size if size is not None else self.pop_size
        return self.rng.uniform(self.lb, self.ub, size=(n, self.dim))

    def _clip(self, pop: np.ndarray) -> np.ndarray:
        """Clip solutions to problem bounds."""
        return np.clip(pop, self.lb, self.ub)

    def get_name(self) -> str:
        """Return human-readable algorithm name."""
        return getattr(self, "_algo_name", self.__class__.__name__)

    @abstractmethod
    def initialize(self) -> None:
        """Initialize population and evaluate initial objectives."""

    @abstractmethod
    def iterate(self, iter_idx: int) -> None:
        """Execute one generation of the multi-objective algorithm."""

    def run(self, run_id: int = 0) -> MOResult:
        """Execute the multi-objective search algorithm."""
        self._nfe = 0
        start_time = time.perf_counter()

        self.initialize()

        for it in range(self.max_iter):
            self.iterate(it)
            if self.verbose and (it % max(1, self.max_iter // 10) == 0 or it == self.max_iter - 1):
                print(f"[{self.get_name()}] Gen {it:>4d}/{self.max_iter} (NFE={self._nfe})")

        elapsed = time.perf_counter() - start_time

        # Extract first non-dominated front
        fronts = non_dominated_sort(self.objectives)
        first_front_idx = fronts[0] if fronts else []
        pareto_sols = self.population[first_front_idx] if len(first_front_idx) > 0 else np.empty((0, self.dim))
        pareto_front = self.objectives[first_front_idx] if len(first_front_idx) > 0 else np.empty((0, self.n_objectives))

        return MOResult(
            algorithm_name=self.get_name(),
            problem_name=self.problem.get_name(),
            run_id=run_id,
            seed=self.seed,
            pareto_solutions=pareto_sols,
            pareto_front=pareto_front,
            all_solutions=self.population.copy(),
            all_objectives=self.objectives.copy(),
            iterations=self.max_iter,
            nfe=self._nfe,
            wall_time_sec=elapsed,
        )
