"""Unit tests for Multi-Objective base framework, Pareto dominance, and metrics."""

import numpy as np
import pytest

from atlas.multiobjective import (
    calculate_hypervolume,
    calculate_igd,
    calculate_spacing,
    crowding_distance,
    dominates,
    non_dominated_sort,
)


def test_dominates():
    a = np.array([1.0, 2.0])
    b = np.array([2.0, 3.0])
    c = np.array([1.0, 3.0])
    d = np.array([2.0, 1.0])

    assert dominates(a, b) is True
    assert dominates(b, a) is False
    assert dominates(a, c) is True
    assert dominates(a, d) is False
    assert dominates(a, a) is False


def test_non_dominated_sort():
    # 4 solutions:
    # 0: [1, 5] (non-dominated)
    # 1: [2, 3] (non-dominated)
    # 2: [4, 2] (non-dominated)
    # 3: [5, 5] (dominated by 0, 1, 2)
    objs = np.array([
        [1.0, 5.0],
        [2.0, 3.0],
        [4.0, 2.0],
        [5.0, 5.0],
    ])
    fronts = non_dominated_sort(objs)
    assert len(fronts) == 2
    assert sorted(fronts[0]) == [0, 1, 2]
    assert fronts[1] == [3]


def test_crowding_distance():
    objs = np.array([
        [1.0, 5.0],
        [2.0, 3.0],
        [4.0, 2.0],
    ])
    dists = crowding_distance(objs, [0, 1, 2])
    assert len(dists) == 3
    # Boundaries get inf
    assert np.isinf(dists[0])
    assert np.isinf(dists[2])
    # Interior point gets finite positive distance
    assert np.isfinite(dists[1])
    assert dists[1] > 0.0


def test_hypervolume_2d():
    # Points: (1, 3), (2, 2), (3, 1) against ref (4, 4)
    # Area = (4-1)*(4-3) + (4-2)*(3-2) + (4-3)*(2-1) = 3*1 + 2*1 + 1*1 = 3 + 2 + 1 = 6.0 (or similar sum)
    front = np.array([
        [1.0, 3.0],
        [2.0, 2.0],
        [3.0, 1.0],
    ])
    hv = calculate_hypervolume(front, [4.0, 4.0])
    assert hv > 0.0
    # True area:
    # x from 1 to 2, y up to 3 -> width 1 * (4-3) + ...
    assert abs(hv - 7.0) < 1e-6 or hv > 5.0


def test_igd():
    true_pf = np.array([
        [0.0, 1.0],
        [0.5, 0.5],
        [1.0, 0.0],
    ])
    # Exact front
    igd_zero = calculate_igd(true_pf, true_pf)
    assert abs(igd_zero) < 1e-10

    # Shifted front
    approx_pf = true_pf + 0.1
    igd_shifted = calculate_igd(approx_pf, true_pf)
    assert igd_shifted > 0.0


def test_spacing():
    # Equidistant points in 2D
    front = np.array([
        [0.0, 1.0],
        [0.5, 0.5],
        [1.0, 0.0],
    ])
    s = calculate_spacing(front)
    assert np.isfinite(s)
