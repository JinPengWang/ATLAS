"""Pareto dominance, fast non-dominated sorting, and crowding distance assignment."""

from __future__ import annotations

from typing import List

import numpy as np


def dominates(obj1: np.ndarray, obj2: np.ndarray) -> bool:
    """Check if objective vector obj1 dominates obj2 (minimization).

    obj1 dominates obj2 if obj1 is no worse than obj2 in all objectives
    and strictly better in at least one objective.
    """
    return bool(np.all(obj1 <= obj2) and np.any(obj1 < obj2))


def non_dominated_sort(objectives: np.ndarray) -> List[List[int]]:
    """Fast non-dominated sorting algorithm (Deb et al., 2002).

    Sorts candidate solutions into hierarchical Pareto fronts (rank 0, rank 1, ...).

    Args:
        objectives: 2-D numpy array of shape (N, M) where N is the population size
            and M is the number of objectives (all objectives minimized).

    Returns:
        List of lists, where each inner list contains the indices of individuals
        belonging to that Pareto front (front 0 is the non-dominated front).
    """
    N = len(objectives)
    if N == 0:
        return []

    # S[p] = set of individual indices that individual p dominates
    S: List[List[int]] = [[] for _ in range(N)]
    # n[p] = domination count: number of individuals that dominate individual p
    n = np.zeros(N, dtype=int)
    fronts: List[List[int]] = [[]]

    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                S[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                n[p] += 1

        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        if len(next_front) > 0:
            fronts.append(next_front)
        else:
            break

    return fronts


def crowding_distance(objectives: np.ndarray, front_indices: List[int]) -> np.ndarray:
    """Compute crowding distance for individuals in a given Pareto front.

    Args:
        objectives: 2-D array of shape (N, M).
        front_indices: List of integer indices corresponding to the front.

    Returns:
        1-D array of shape (len(front_indices),) containing crowding distances.
        Boundary points along each objective are assigned infinity.
    """
    l = len(front_indices)
    if l <= 2:
        return np.full(l, np.inf)

    distances = np.zeros(l, dtype=float)
    front_objs = objectives[front_indices]
    n_objs = front_objs.shape[1]

    for m in range(n_objs):
        # Sort by m-th objective
        sorted_idx = np.argsort(front_objs[:, m])
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf

        f_min = front_objs[sorted_idx[0], m]
        f_max = front_objs[sorted_idx[-1], m]
        norm = f_max - f_min
        if norm <= 1e-12:
            norm = 1.0

        for i in range(1, l - 1):
            if not np.isinf(distances[sorted_idx[i]]):
                distances[sorted_idx[i]] += (
                    front_objs[sorted_idx[i + 1], m] - front_objs[sorted_idx[i - 1], m]
                ) / norm

    return distances
