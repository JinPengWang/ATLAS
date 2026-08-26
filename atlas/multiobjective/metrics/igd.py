"""Inverted Generational Distance (IGD), Generational Distance (GD), and Spacing metrics."""

from __future__ import annotations

import numpy as np


def calculate_igd(front: np.ndarray, true_pf: np.ndarray) -> float:
    """Calculate Inverted Generational Distance (IGD).

    Measures both convergence and diversity: average Euclidean distance from
    each point on the true Pareto front to the nearest point in the obtained front.

    Args:
        front: 2-D array of shape (N, M) representing obtained Pareto front.
        true_pf: 2-D array of shape (K, M) representing reference true Pareto front.

    Returns:
        Scalar IGD value (lower is better, 0.0 means perfect approximation).
    """
    if len(front) == 0 or len(true_pf) == 0:
        return float("inf")

    # For each point in true_pf, find minimum distance to any point in front
    distances = []
    for ref_p in true_pf:
        d = np.min(np.linalg.norm(front - ref_p, axis=1))
        distances.append(d)

    return float(np.mean(distances))


def calculate_gd(front: np.ndarray, true_pf: np.ndarray) -> float:
    """Calculate Generational Distance (GD).

    Measures convergence: average Euclidean distance from each point in the
    obtained front to the nearest point on the true Pareto front.

    Args:
        front: 2-D array of shape (N, M).
        true_pf: 2-D array of shape (K, M).

    Returns:
        Scalar GD value (lower is better).
    """
    if len(front) == 0 or len(true_pf) == 0:
        return float("inf")

    distances = []
    for p in front:
        d = np.min(np.linalg.norm(true_pf - p, axis=1))
        distances.append(d)

    return float(np.mean(distances))


def calculate_spacing(front: np.ndarray) -> float:
    """Calculate Schott's Spacing metric (S).

    Measures the uniformity of the distribution of points along the Pareto front.

    Args:
        front: 2-D array of shape (N, M).

    Returns:
        Scalar Spacing value (lower is better, 0.0 means equidistantly spaced).
    """
    N = len(front)
    if N <= 1:
        return 0.0

    # For each point i, find min L1 distance to any other point j != i
    min_d = np.zeros(N)
    for i in range(N):
        dists = []
        for j in range(N):
            if i != j:
                dists.append(np.sum(np.abs(front[i] - front[j])))
        min_d[i] = min(dists) if dists else 0.0

    d_bar = np.mean(min_d)
    spacing = np.sqrt(np.sum((min_d - d_bar) ** 2) / (N - 1))
    return float(spacing)
