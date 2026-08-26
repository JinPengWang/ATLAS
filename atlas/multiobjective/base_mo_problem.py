"""Abstract base class for multi-objective optimization problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class BaseMOProblem(ABC):
    """Abstract base class that all multi-objective problems subclass.

    Subclasses must implement :meth:`evaluate`, :meth:`get_bounds`, and
    :meth:`get_name`.

    Args:
        dim: Decision space dimensionality.
        n_objectives: Objective space dimensionality (>= 2).
        boundary_strategy: Strategy to handle boundary violations ('clip', 'reflect', 'penalty').
        penalty_weight: Multiplier used for penalty boundary strategy.
    """

    VALID_STRATEGIES = ("clip", "reflect", "penalty")

    def __init__(
        self,
        dim: int = 30,
        n_objectives: int = 2,
        boundary_strategy: str = "clip",
        penalty_weight: float = 1e6,
    ) -> None:
        if boundary_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"boundary_strategy must be one of {self.VALID_STRATEGIES}, "
                f"got '{boundary_strategy}'"
            )
        self.dim = dim
        self.n_objectives = n_objectives
        self.boundary_strategy = boundary_strategy
        self.penalty_weight = penalty_weight

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the objective functions at decision vector x.

        Args:
            x: 1-D numpy array of shape (dim,).

        Returns:
            1-D numpy array of shape (n_objectives,) containing objective values
            (minimisation for all objectives).
        """

    @abstractmethod
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return lower and upper bounds of the decision space."""

    @abstractmethod
    def get_name(self) -> str:
        """Return human-readable problem name."""

    def get_dim(self) -> int:
        """Return decision space dimensionality."""
        return self.dim

    def get_num_objectives(self) -> int:
        """Return number of objectives."""
        return self.n_objectives

    def get_pareto_front(self, n_points: int = 100) -> Optional[np.ndarray]:
        """Return points from true theoretical Pareto front if known."""
        return None

    def clamp(self, x: np.ndarray) -> np.ndarray:
        """Apply boundary strategy to decision vector x."""
        lb, ub = self.get_bounds()
        if self.boundary_strategy == "clip":
            return np.clip(x, lb, ub)
        elif self.boundary_strategy == "reflect":
            x_new = x.copy()
            range_ = ub - lb
            for _ in range(50):
                below = x_new < lb
                above = x_new > ub
                if not (below.any() or above.any()):
                    break
                x_new = np.where(below, lb + (lb - x_new) % range_, x_new)
                x_new = np.where(above, ub - (x_new - ub) % range_, x_new)
            return np.clip(x_new, lb, ub)
        return x

    def evaluate_with_penalty(self, x: np.ndarray) -> np.ndarray:
        """Evaluate decision vector x with automatic boundary handling."""
        if self.boundary_strategy == "penalty":
            lb, ub = self.get_bounds()
            violation_low = np.maximum(lb - x, 0.0)
            violation_high = np.maximum(x - ub, 0.0)
            penalty = np.sum(violation_low + violation_high)
            obj = self.evaluate(x)
            return obj + self.penalty_weight * penalty
        else:
            x_clamped = self.clamp(x)
            return self.evaluate(x_clamped)

    def __repr__(self) -> str:
        return f"{self.get_name()}(dim={self.dim}, n_obj={self.n_objectives})"
