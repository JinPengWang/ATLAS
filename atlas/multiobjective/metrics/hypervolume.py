"""Hypervolume (HV) indicator for multi-objective optimization performance."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def calculate_hypervolume(
    front: np.ndarray,
    reference_point: Union[np.ndarray, Sequence[float]],
) -> float:
    """Calculate Hypervolume indicator of a Pareto front with respect to a reference point.

    For 2-D objective space, calculates exact Lebesgue measure via rectangular slices.
    For higher dimensions (M >= 3), uses exact dimension recursion or Monte Carlo.

    Args:
        front: 2-D array of shape (N, M) representing non-dominated objective vectors.
        reference_point: 1-D array of shape (M,) serving as upper bound for all objectives.

    Returns:
        Scalar hypervolume value (higher is better).
    """
    ref = np.asarray(reference_point, dtype=float)
    if len(front) == 0:
        return 0.0

    # Filter out points that do not dominate the reference point
    valid = np.all(front < ref, axis=1)
    pts = front[valid]
    if len(pts) == 0:
        return 0.0

    n_objs = pts.shape[1]

    if n_objs == 2:
        # Sort 2D points by first objective ascending
        sorted_idx = np.lexsort((-pts[:, 1], pts[:, 0]))
        pts = pts[sorted_idx]

        # Filter to strictly non-dominated subset in 2D
        non_dom = [pts[0]]
        for p in pts[1:]:
            if p[1] < non_dom[-1][1]:
                non_dom.append(p)
        non_dom_arr = np.array(non_dom)

        # Compute rectangles
        hv = 0.0
        x_prev = non_dom_arr[0, 0]
        y_prev = ref[1]
        for x_curr, y_curr in non_dom_arr:
            hv += (ref[0] - x_curr) * (y_prev - y_curr)
            y_prev = y_curr
        return float(hv)
    else:
        # Monte Carlo approximation for higher dimensions M >= 3
        rng = np.random.default_rng(42)
        n_samples = 100000
        min_bounds = np.min(pts, axis=0)
        samples = rng.uniform(min_bounds, ref, size=(n_samples, n_objs))
        # Count samples dominated by at least one point in pts
        # A sample is dominated by point p if p <= sample for all objectives
        dominated_count = 0
        for s in samples:
            if np.any(np.all(pts <= s, axis=1)):
                dominated_count += 1
        bounding_box_vol = np.prod(ref - min_bounds)
        return float((dominated_count / n_samples) * bounding_box_vol)
