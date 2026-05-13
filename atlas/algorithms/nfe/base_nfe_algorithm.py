"""Base class for NFE-based (function-evaluation-counted) algorithm variants."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from atlas.core.base_algorithm import BaseAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.core.result import Result


class BaseNFEAlgorithm(BaseAlgorithm, ABC):
    """Base for algorithm variants that stop after a maximum number of
    function evaluations (NFE) rather than a fixed iteration count.

    Subclasses implement ``initialize()`` and ``iterate()`` exactly as for
    the iteration-based counterpart, but must call ``self._evaluate(x)``
    instead of ``self.problem.evaluate_with_penalty(x)`` so that the NFE
    counter is incremented.

    Args:
        problem: Optimisation problem instance.
        max_nfe: Maximum number of function evaluations.
        pop_size: Population size.
        seed: Random seed.
        verbose: Print progress.
        **kwargs: Algorithm-specific hyper-parameters.
    """

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 0,
        max_nfe: int = 10000,
        pop_size: int = 30,
        seed: int = 42,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        # max_iter is ignored for NFE-based algorithms; stopping is
        # controlled by max_nfe.  We accept max_iter for API compatibility
        # with Experiment._make_algorithm which always passes it.
        super().__init__(
            problem, max_iter=0, pop_size=pop_size,
            seed=seed, verbose=verbose, **kwargs,
        )
        self.max_nfe = max_nfe
        self._nfe: int = 0

    # ------------------------------------------------------------------
    # Evaluation counter
    # ------------------------------------------------------------------
    def _evaluate(self, x: np.ndarray) -> float:
        """Evaluate *x* and increment the NFE counter."""
        self._nfe += 1
        return self.problem.evaluate_with_penalty(x)

    # ------------------------------------------------------------------
    # Template method (NFE-based stopping)
    # ------------------------------------------------------------------
    def run(self, run_id: int = 0) -> Result:
        """Execute the algorithm until the NFE budget is exhausted.

        Args:
            run_id: Identifier for this particular run.

        Returns:
            A :class:`Result` object containing the outcome.
        """
        self._convergence = []
        self._nfe = 0
        self.initialize()

        it = 0
        while self._nfe < self.max_nfe:
            best_f = self.iterate(it)
            self._convergence.append(float(best_f))

            if self.verbose and it % max(1, self.max_nfe // (self.pop_size * 20)) == 0:
                print(
                    f"  [{self.get_name()}] iter {it:>5d}  "
                    f"NFE={self._nfe}/{self.max_nfe}  best={best_f:.6e}"
                )
            it += 1

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
        )

    def get_params(self) -> dict:
        params = super().get_params()
        params["max_nfe"] = self.max_nfe
        return params
