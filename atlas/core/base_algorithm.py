"""Abstract base class for all metaheuristic algorithms."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from atlas.core.base_problem import BaseProblem
from atlas.core.result import Result
from atlas.visualization.diversity import DiversityTracker


class BaseAlgorithm(ABC):
    """Abstract base class that every algorithm subclasses.

    To create a new algorithm, you only need to subclass :class:`BaseAlgorithm`
    and implement:

    * :meth:`initialize` – set up population and initial fitness.
    * :meth:`iterate` – execute one step of search, return current best fitness.
    * :meth:`get_name` – (optional) human-readable name, defaults to class name.

    The base class automatically handles:
    * Function evaluation counting (NFE)
    * Dual stopping criteria (iteration-based vs NFE-based)
    * Solution boundary clamping / reflection / penalty
    * Global best solution and fitness tracking
    * Target fitness and early-stopping checks
    * Optional population diversity tracking

    Args:
        problem: The optimisation problem to solve.
        max_iter: Maximum number of iterations.
        max_nfe: Maximum number of function evaluations (for NFE stopping).
        stopping_criterion: ``'iterations'`` (default) or ``'nfe'``.
        pop_size: Population size (or equivalent).
        seed: Random seed for reproducibility.
        verbose: Whether to print per-iteration progress.
        target_fitness: Optional target fitness to trigger early stopping.
        track_diversity: Whether to track population diversity across iterations.
        **kwargs: Algorithm-specific hyper-parameters.
    """

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 500,
        max_nfe: Optional[int] = None,
        stopping_criterion: Optional[str] = None,
        pop_size: int = 30,
        seed: int = 42,
        verbose: bool = False,
        target_fitness: Optional[float] = None,
        track_diversity: bool = False,
        **kwargs: Any,
    ) -> None:
        self.problem = problem
        self.max_iter = max_iter
        self.max_nfe = max_nfe if max_nfe is not None else 10000
        self.pop_size = pop_size
        self.seed = seed
        self.verbose = verbose
        self.target_fitness = target_fitness
        self.track_diversity = track_diversity
        self.extra_params = kwargs

        if stopping_criterion is None:
            if max_nfe is not None and max_nfe > 0:
                self.stopping_criterion = "nfe"
            elif max_iter <= 0:
                self.stopping_criterion = "nfe"
            else:
                self.stopping_criterion = "iterations"
        else:
            self.stopping_criterion = stopping_criterion


        # RNG – use Generator for thread-safe reproducibility
        self.rng: np.random.Generator = np.random.default_rng(seed)

        # Problem info
        self.dim: int = problem.get_dim()
        self.lb, self.ub = problem.get_bounds()

        # State initialised in initialize()
        self.population: np.ndarray = np.empty((0, self.dim))
        self.fitness: np.ndarray = np.empty(0)
        self.g_best_x: np.ndarray = np.zeros(self.dim)
        self.g_best_f: float = np.inf

        # Evaluation & convergence tracking
        self._nfe: int = 0
        self._convergence: List[float] = []
        self._stopped_early: bool = False
        self._diversity_tracker: Optional[DiversityTracker] = None

    # ------------------------------------------------------------------
    # Convenient Evaluation API for Subclasses
    # ------------------------------------------------------------------
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate a single candidate solution with boundary handling.

        Automatically increments the NFE counter, tracks global best,
        and checks for early stopping.

        Args:
            x: Decision vector of shape ``(dim,)``.

        Returns:
            Scalar fitness value.
        """
        self._nfe += 1
        f = self.problem.evaluate_with_penalty(x)
        if f < self.g_best_f:
            self.g_best_f = float(f)
            self.g_best_x = x.copy()
            if self.target_fitness is not None and self.g_best_f <= self.target_fitness:
                self._stopped_early = True
        return float(f)

    def _evaluate(self, x: np.ndarray) -> float:
        """Backward-compatible alias for :meth:`evaluate`."""
        return self.evaluate(x)

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate an array of candidate solutions.

        Args:
            population: 2-D array of shape ``(pop_size, dim)``.

        Returns:
            1-D array of fitness values of shape ``(pop_size,)``.
        """
        n = len(population)
        fitness = np.empty(n)
        for i in range(n):
            fitness[i] = self.evaluate(population[i])
        return fitness

    def init_population(self, pop_size: Optional[int] = None) -> np.ndarray:
        """Uniformly sample a new population within problem bounds.

        Args:
            pop_size: Size of population (defaults to ``self.pop_size``).

        Returns:
            Array of shape ``(pop_size, dim)``.
        """
        size = pop_size if pop_size is not None else self.pop_size
        return self.rng.uniform(self.lb, self.ub, size=(size, self.dim))

    # ------------------------------------------------------------------
    # Template Execution Method
    # ------------------------------------------------------------------
    def run(self, run_id: int = 0) -> Result:
        """Execute the algorithm under the configured stopping criterion.

        Args:
            run_id: Identifier for this particular run.

        Returns:
            A :class:`Result` object containing the outcome.
        """
        self._convergence = []
        self._nfe = 0
        self._stopped_early = False
        self.g_best_f = np.inf
        self._diversity_tracker = DiversityTracker() if self.track_diversity else None
        start_time = time.perf_counter()

        self.initialize()

        it = 0
        # Determine effective stopping criterion
        use_nfe = (
            self.stopping_criterion == "nfe"
            or (self.max_iter <= 0 and self.max_nfe > 0)
            or self.get_name().upper().endswith("_NFE")
        )

        if use_nfe:
            while self._nfe < self.max_nfe and not self._stopped_early:
                best_f = self.iterate(it)
                self._convergence.append(float(self.g_best_f))
                if self.track_diversity and self._diversity_tracker is not None and len(self.population) > 0:
                    self._diversity_tracker.update(self.population)

                if self.verbose and (
                    it % max(1, self.max_nfe // (max(self.pop_size, 1) * 20)) == 0
                    or self._nfe >= self.max_nfe
                ):
                    print(
                        f"  [{self.get_name()}] iter {it:>5d}  "
                        f"NFE={self._nfe}/{self.max_nfe}  best={self.g_best_f:.6e}"
                    )
                it += 1
        else:
            for it_idx in range(self.max_iter):
                if self._stopped_early:
                    break
                best_f = self.iterate(it_idx)
                self._convergence.append(float(self.g_best_f))
                if self.track_diversity and self._diversity_tracker is not None and len(self.population) > 0:
                    self._diversity_tracker.update(self.population)

                if self.verbose and (
                    it_idx % max(1, self.max_iter // 20) == 0
                    or it_idx == self.max_iter - 1
                ):
                    print(
                        f"  [{self.get_name()}] iter {it_idx:>5d}/{self.max_iter}  "
                        f"best={self.g_best_f:.6e}"
                    )
                it = it_idx + 1

        elapsed = time.perf_counter() - start_time

        extra_info: Dict[str, Any] = {"target_reached": self._stopped_early}
        if self.track_diversity and self._diversity_tracker is not None:
            extra_info["diversity_curve"] = self._diversity_tracker.get_diversity_curve()

        return Result(
            algorithm_name=self.get_name(),
            problem_name=self.problem.get_name(),
            run_id=run_id,
            seed=self.seed,
            best_fitness=float(self.g_best_f),
            best_solution=self.g_best_x.copy(),
            convergence_curve=list(self._convergence),
            iterations=it,
            nfe=self._nfe,
            wall_time_sec=elapsed,
            extra=extra_info,
        )

    # ------------------------------------------------------------------
    # Abstract interface to implement
    # ------------------------------------------------------------------
    @abstractmethod
    def initialize(self) -> None:
        """Set up the initial population and evaluate initial fitness.

        Subclasses should populate ``self.population`` and ``self.fitness``.
        """

    @abstractmethod
    def iterate(self, iter_idx: int) -> float:
        """Perform one iteration of the search algorithm.

        Args:
            iter_idx: Current iteration index (0-based).

        Returns:
            The best fitness value found so far.
        """

    def get_name(self) -> str:
        """Return human-readable algorithm name (defaults to class name)."""
        base = getattr(self, "_algo_name", self.__class__.__name__)
        if (self.stopping_criterion == "nfe" or (self.max_iter <= 0 and self.max_nfe > 0)) and not base.endswith("_NFE"):
            return f"{base}_NFE"
        return base


    # ------------------------------------------------------------------
    # Subclass Helpers
    # ------------------------------------------------------------------
    def get_params(self) -> Dict[str, Any]:
        """Return dictionary of current hyper-parameters."""
        params = {
            "max_iter": self.max_iter,
            "max_nfe": self.max_nfe,
            "stopping_criterion": self.stopping_criterion,
            "pop_size": self.pop_size,
            "seed": self.seed,
            "track_diversity": self.track_diversity,
        }
        params.update(self.extra_params)
        return params

    def _update_global_best(self) -> None:
        """Scan ``self.fitness`` and update the global best."""
        if len(self.fitness) > 0:
            idx = int(np.argmin(self.fitness))
            if self.fitness[idx] < self.g_best_f:
                self.g_best_f = float(self.fitness[idx])
                self.g_best_x = self.population[idx].copy()

    def _clip(self, x: np.ndarray) -> np.ndarray:
        """Clip a solution to the problem bounds."""
        return np.clip(x, self.lb, self.ub)

    def __repr__(self) -> str:
        return f"{self.get_name()}(problem={self.problem.get_name()!r})"
