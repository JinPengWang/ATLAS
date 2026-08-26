"""Tests for modern algorithms: GWO, HHO, SMA, ABC (Task 2.3)."""

from __future__ import annotations

import pytest

from atlas.problems.benchmark.unimodal import Sphere
from atlas.utils.registry import get_algorithm


@pytest.mark.parametrize("name", ["gwo", "hho", "sma", "abc"])
def test_registered(name: str) -> None:
    assert get_algorithm(name) is not None


@pytest.mark.parametrize("name", ["gwo_nfe", "hho_nfe", "sma_nfe", "abc_nfe"])
def test_nfe_alias_registered(name: str) -> None:
    assert get_algorithm(name) is not None


@pytest.mark.parametrize("name", ["gwo", "hho", "sma", "abc"])
def test_runs_and_converges(name: str) -> None:
    prob = Sphere(dim=10)
    algo = get_algorithm(name)(problem=prob, max_iter=50, pop_size=20, seed=42)
    result = algo.run()
    assert result.best_fitness < 100.0
    assert result.nfe > 0
    assert len(result.convergence_curve) == 50


@pytest.mark.parametrize("name", ["gwo", "hho", "sma", "abc"])
def test_nfe_mode(name: str) -> None:
    prob = Sphere(dim=5)
    algo = get_algorithm(name + "_nfe")(problem=prob, max_nfe=1000, pop_size=10, seed=0)
    result = algo.run()
    assert result.nfe <= 1500


@pytest.mark.parametrize("name", ["gwo", "abc"])
def test_reproducible(name: str) -> None:
    prob = Sphere(dim=5)
    algo1 = get_algorithm(name)(problem=prob, max_iter=20, pop_size=10, seed=3)
    algo2 = get_algorithm(name)(problem=prob, max_iter=20, pop_size=10, seed=3)
    r1, r2 = algo1.run(), algo2.run()
    assert abs(r1.best_fitness - r2.best_fitness) < 1e-10
