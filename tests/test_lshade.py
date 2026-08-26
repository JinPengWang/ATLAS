"""Tests for L-SHADE optimizer (Task 2.1)."""

import pytest
from atlas.problems.benchmark.unimodal import Sphere
from atlas.utils.registry import get_algorithm


def test_lshade_registered():
    cls = get_algorithm("lshade")
    assert cls is not None


def test_lshade_nfe_alias_registered():
    cls = get_algorithm("lshade_nfe")
    assert cls is not None


def test_lshade_runs_and_produces_result():
    prob = Sphere(dim=10)
    algo = get_algorithm("lshade")(problem=prob, max_iter=50, pop_size=40, seed=42)
    result = algo.run()
    assert result.best_fitness < 100.0  # 应该有一定收敛
    assert result.nfe > 0
    assert len(result.convergence_curve) > 0


def test_lshade_nfe_mode():
    prob = Sphere(dim=10)
    algo = get_algorithm("lshade_nfe")(problem=prob, max_nfe=2000, pop_size=40, seed=42)
    result = algo.run()
    assert result.nfe <= 2500  # 允许小幅超出（最后一代可能多小量评估）


def test_lshade_reproducible():
    prob = Sphere(dim=5)
    algo1 = get_algorithm("lshade")(problem=prob, max_iter=20, pop_size=20, seed=0)
    algo2 = get_algorithm("lshade")(problem=prob, max_iter=20, pop_size=20, seed=0)
    r1, r2 = algo1.run(), algo2.run()
    assert abs(r1.best_fitness - r2.best_fitness) < 1e-10
