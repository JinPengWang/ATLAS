"""Abstract base class for all metaheuristic algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from atlas.core.base_problem import BaseProblem
from atlas.core.result import Result


class BaseAlgorithm(ABC):
    """Abstract base class that every algorithm must subclass.

    Subclasses **must** implement:

    * :meth:`initialize` – set up the population / initial state.
    * :meth:`iterate` – perform one iteration, return current best fitness.
    * :meth:`get_name` – return a human-readable algorithm name.

    Args:
        problem: The optimisation problem to solve.
        max_iter: Maximum number of iterations.
        pop_size: Population size (or equivalent).
        seed: Random seed for reproducibility.
        verbose: Whether to print per-iteration progress.
        **kwargs: Algorithm-specific hyper-parameters.
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
        self.problem = problem
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.seed = seed
        self.verbose = verbose
        self.extra_params = kwargs

        # RNG – use the new-style Generator for reproducibility
        self.rng: np.random.Generator = np.random.default_rng(seed)

        # Problem info
        self.dim: int = problem.get_dim()
        self.lb: np.ndarray
        self.ub: np.ndarray
        self.lb, self.ub = problem.get_bounds()

        # State initialised in initialize()
        self.population: np.ndarray = np.empty(0)
        self.fitness: np.ndarray = np.empty(0)
        self.g_best_x: np.ndarray = np.zeros(self.dim)
        self.g_best_f: float = np.inf

        # Convergence tracking
        self._convergence: List[float] = []

    # ------------------------------------------------------------------
    # Template method
    # ------------------------------------------------------------------
    def run(self, run_id: int = 0) -> Result:
        """Execute the algorithm: initialise then iterate.

        Args:
            run_id: Identifier for this particular run (used in the result).

        Returns:
            A :class:`Result` object containing the outcome.
        """
        self._convergence = []
        self.initialize()

        for it in range(self.max_iter):
            best_f = self.iterate(it)
            self._convergence.append(float(best_f))

            if self.verbose and (it % max(1, self.max_iter // 20) == 0 or it == self.max_iter - 1):
                print(f"  [{self.get_name()}] iter {it:>5d}/{self.max_iter}  best={best_f:.6e}")

        return Result(
            algorithm_name=self.get_name(),
            problem_name=self.problem.get_name(),
            run_id=run_id,
            seed=self.seed,
            best_fitness=float(self.g_best_f),
            best_solution=self.g_best_x.copy(),
            convergence_curve=list(self._convergence),
            iterations=self.max_iter,
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def initialize(self) -> None:
        """Set up the initial population and evaluate fitness.

        Must populate ``self.population`` (shape ``(pop_size, dim)``),
        ``self.fitness`` (shape ``(pop_size,)``), and set
        ``self.g_best_x`` / ``self.g_best_f``.
        """

    @abstractmethod
    def iterate(self, iter_idx: int) -> float:
        """Perform one iteration of the algorithm.

        Args:
            iter_idx: Current iteration index (0-based).

        Returns:
            The best fitness value found so far.
        """

    @abstractmethod
    def get_name(self) -> str:
        """Return a short human-readable algorithm name (e.g. ``'PSO'``)."""

    # ------------------------------------------------------------------
    # Helpers available to subclasses
    # ------------------------------------------------------------------
    def get_params(self) -> Dict[str, Any]:
        """Return a dictionary of current hyper-parameters.

        Returns:
            Parameter dictionary including common and algorithm-specific
            settings.
        """
        params = {
            "max_iter": self.max_iter,
            "pop_size": self.pop_size,
            "seed": self.seed,
        }
        params.update(self.extra_params)
        return params

    def _update_global_best(self) -> None:
        """Scan ``self.fitness`` and update the global best."""
        idx = int(np.argmin(self.fitness))
        if self.fitness[idx] < self.g_best_f:
            self.g_best_f = float(self.fitness[idx])
            self.g_best_x = self.population[idx].copy()

    def _clip(self, x: np.ndarray) -> np.ndarray:
        """Clip a solution to the problem bounds."""
        return np.clip(x, self.lb, self.ub)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(problem={self.problem.get_name()!r})"
