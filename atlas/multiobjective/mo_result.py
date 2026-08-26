"""Data structure for multi-objective optimization results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class MOResult:
    """Container for the outcome of a multi-objective algorithm run."""

    algorithm_name: str
    problem_name: str
    run_id: int
    seed: int
    pareto_solutions: np.ndarray  # Decision vectors on first Pareto front, shape (N, dim)
    pareto_front: np.ndarray  # Objective vectors on first Pareto front, shape (N, n_objectives)
    all_solutions: Optional[np.ndarray] = None  # Full final population
    all_objectives: Optional[np.ndarray] = None  # Full final objectives
    iterations: int = 0
    nfe: int = 0
    wall_time_sec: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        n_pf = len(self.pareto_front) if self.pareto_front is not None else 0
        return (
            f"MOResult(algo={self.algorithm_name!r}, "
            f"problem={self.problem_name!r}, "
            f"n_pareto={n_pf}, nfe={self.nfe})"
        )
