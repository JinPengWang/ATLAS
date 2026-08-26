"""Data structures for storing experiment results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Result:
    """Container for the outcome of a single algorithm run.

    Attributes:
        algorithm_name: Identifier of the algorithm.
        problem_name: Identifier of the problem.
        run_id: Run index (0-based).
        seed: Random seed used for this run.
        best_fitness: Best fitness value found.
        best_solution: Decision vector corresponding to *best_fitness*.
        convergence_curve: List of best-so-far fitness at each iteration.
        iterations: Total iterations executed.
        extra: Arbitrary additional metadata.
    """

    algorithm_name: str
    problem_name: str
    run_id: int
    seed: int
    best_fitness: float
    best_solution: np.ndarray
    convergence_curve: List[float] = field(default_factory=list)
    iterations: int = 0
    nfe: int = 0
    wall_time_sec: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def best_solution_dict(self) -> Dict[str, float]:
        """Return best solution as ``{x0: v0, x1: v1, ...}`` dict."""
        return {f"x{i}": float(v) for i, v in enumerate(self.best_solution)}

    def __repr__(self) -> str:
        return (
            f"Result(algo={self.algorithm_name!r}, "
            f"problem={self.problem_name!r}, "
            f"run={self.run_id}, fitness={self.best_fitness:.6e})"
        )
