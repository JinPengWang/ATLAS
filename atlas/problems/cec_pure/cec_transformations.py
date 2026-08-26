"""Vectorized transformation helpers for pure-Python CEC benchmarks."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def get_deterministic_shift_and_rotation(
    func_id: int,
    dim: int,
    bound: float = 80.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate deterministic orthogonal rotation matrix and shift vector for CEC benchmarks.

    Args:
        func_id: Function number (1, 2, ...).
        dim: Problem dimensionality.
        bound: Maximum absolute value for the shift vector.

    Returns:
        Tuple of (shift_vector of shape (dim,), rotation_matrix of shape (dim, dim)).
    """
    seed = int(func_id * 100003 + dim * 7919)
    rng = np.random.default_rng(seed)

    # Shift vector within [-bound, bound]
    shift = rng.uniform(-bound, bound, size=dim)

    # Orthogonal rotation matrix via QR decomposition
    A = rng.normal(0.0, 1.0, size=(dim, dim))
    Q, R = np.linalg.qr(A)
    # Ensure determinant is +1 (proper rotation)
    d = np.diagonal(R)
    ph = d / np.abs(d)
    M = Q * ph
    if np.linalg.det(M) < 0:
        M[:, 0] = -M[:, 0]

    return shift, M


def transform_x(x: np.ndarray, shift: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply shift and rotation to decision vector x: z = M @ (x - shift)."""
    return M @ (x - shift)
