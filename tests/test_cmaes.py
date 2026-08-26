"""Tests for CMA-ES optimizer (Task 2.2)."""

import math

import pytest
from atlas.problems.benchmark.unimodal import Sphere, Rosenbrock
from atlas.utils.registry import get_algorithm


def test_cmaes_registered():
    assert get_algorithm("cmaes") is not None
    assert get_algorithm("cma_es") is not None
    assert get_algorithm("cmaes_nfe") is not None


def test_cmaes_runs_sphere():
    prob = Sphere(dim=5)
    algo = get_algorithm("cmaes")(problem=prob, max_iter=100, seed=42)
    result = algo.run()
    assert result.best_fitness < 1.0
    assert result.nfe > 0


def test_cmaes_nfe_mode():
    prob = Sphere(dim=5)
    algo = get_algorithm("cmaes_nfe")(problem=prob, max_nfe=1000, seed=42)
    result = algo.run()
    assert result.nfe <= 1500


def test_cmaes_reproducible():
    prob = Sphere(dim=3)
    algo1 = get_algorithm("cmaes")(problem=prob, max_iter=30, seed=7)
    algo2 = get_algorithm("cmaes")(problem=prob, max_iter=30, seed=7)
    r1, r2 = algo1.run(), algo2.run()
    assert abs(r1.best_fitness - r2.best_fitness) < 1e-10


def test_cmaes_default_popsize():
    """Default pop_size should be 4 + floor(3*log(dim))."""
    prob = Sphere(dim=10)
    algo = get_algorithm("cmaes")(problem=prob, max_iter=5, seed=0)
    expected = 4 + int(3 * math.log(10))
    assert algo.pop_size == expected
