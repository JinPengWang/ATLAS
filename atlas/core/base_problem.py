"""Abstract base class for all optimisation problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np


class BaseProblem(ABC):
    """Abstract base class that every optimisation problem must subclass.

    Subclasses **must** implement :meth:`evaluate`, :meth:`get_bounds`, and
    :meth:`get_name`.  The remaining methods have sensible defaults.

    Args:
        dim: Problem dimensionality.
        boundary_strategy: How to handle solutions outside the search space.
            One of ``'clip'``, ``'reflect'``, or ``'penalty'``.
        penalty_weight: Multiplier used when *boundary_strategy* is
            ``'penalty'``.
    """

    VALID_STRATEGIES = ("clip", "reflect", "penalty")

    def __init__(
        self,
        dim: int = 30,
        boundary_strategy: str = "clip",
        penalty_weight: float = 1e6,
    ) -> None:
        if boundary_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"boundary_strategy must be one of {self.VALID_STRATEGIES}, "
                f"got '{boundary_strategy}'"
            )
        self.dim = dim
        self.boundary_strategy = boundary_strategy
        self.penalty_weight = penalty_weight

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the objective function at decision vector *x*.

        Args:
            x: A 1-D numpy array of shape ``(dim,)``.

        Returns:
            Scalar fitness value (lower is better for minimisation).
        """

    @abstractmethod
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return lower and upper bounds.

        Returns:
            A tuple ``(lb, ub)`` where each element is a 1-D numpy array of
            shape ``(dim,)``.
        """

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable problem name."""

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------
    def get_optimum(self) -> Optional[float]:
        """Return the known global optimum value, or ``None`` if unknown."""
        return None

    def get_optimum_location(self) -> Optional[np.ndarray]:
        """Return the known global optimum location, or ``None`` if unknown."""
        return None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def get_dim(self) -> int:
        """Return problem dimensionality."""
        return self.dim

    def clamp(self, x: np.ndarray) -> np.ndarray:
        """Apply the configured boundary strategy to *x*.

        Args:
            x: A 1-D numpy array.

        Returns:
            The adjusted decision vector.
        """
        lb, ub = self.get_bounds()
        strategy = self.boundary_strategy

        if strategy == "clip":
            return np.clip(x, lb, ub)

        if strategy == "reflect":
            # Reflect back into bounds until within range
            x_new = x.copy()
            range_ = ub - lb
            for _ in range(100):  # safety limit
                below = x_new < lb
                above = x_new > ub
                if not (below.any() or above.any()):
                    break
                x_new = np.where(below, lb + (lb - x_new) % range_, x_new)
                x_new = np.where(above, ub - (x_new - ub) % range_, x_new)
            return np.clip(x_new, lb, ub)

        # penalty: return x unchanged; penalty applied in evaluate_with_penalty
        return x

    def evaluate_with_penalty(self, x: np.ndarray) -> float:
        """Evaluate with automatic boundary handling.

        If *boundary_strategy* is ``'penalty'``, a large penalty is added when
        *x* violates bounds.  Otherwise the solution is clamped first.

        Args:
            x: A 1-D numpy array.

        Returns:
            Scalar fitness value.
        """
        lb, ub = self.get_bounds()
        if self.boundary_strategy == "penalty":
            penalty = 0.0
            violation_low = np.maximum(lb - x, 0.0)
            violation_high = np.maximum(x - ub, 0.0)
            penalty = np.sum(violation_low + violation_high)
            return self.evaluate(x) + self.penalty_weight * penalty
        else:
            x_clamped = self.clamp(x)
            return self.evaluate(x_clamped)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate a batch of candidate solutions.

        Args:
            X: A 2-D numpy array of shape ``(N, dim)``.

        Returns:
            A 1-D numpy array of shape ``(N,)`` containing fitness values.
        """
        N = len(X)
        fitness = np.empty(N)
        for i in range(N):
            fitness[i] = self.evaluate_with_penalty(X[i])
        return fitness

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(dim={self.dim}, "
            f"strategy={self.boundary_strategy})"
        )

