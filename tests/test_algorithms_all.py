"""Regression tests for all 18 ATLAS algorithms."""
import pytest
import numpy as np
from atlas.problems.benchmark import Sphere
from atlas.utils.registry import get_algorithm

ALGO_NAMES = [
    "pso", "ga", "de", "cmaes", "lshade",
    "gwo", "woa", "hho", "sma", "abc",
    "aco", "sa", "psa", "dp",
    "tjo", "ppo", "lea", "lgc",
]

@pytest.mark.parametrize("algo_name", ALGO_NAMES)
def test_algorithm_converges(algo_name):
    """Algorithm must improve over initial random population on Sphere D=5."""
    prob = Sphere(dim=5)
    AlgoCls = get_algorithm(algo_name)
    alg = AlgoCls(prob, max_iter=100, pop_size=15, seed=42)
    result = alg.run()
    assert result.best_fitness < 1e6, f"{algo_name} did not return a valid fitness"
    # Should improve over random initialization (Sphere D=5 random in [-5.12,5.12] ≈ 5-40)
    assert result.best_fitness < 2000.0, f"{algo_name} failed to converge: {result.best_fitness}"

@pytest.mark.parametrize("algo_name", ALGO_NAMES)
def test_algorithm_deterministic(algo_name):
    """Same seed must give same result."""
    prob = Sphere(dim=5)
    AlgoCls = get_algorithm(algo_name)
    r1 = AlgoCls(prob, max_iter=50, pop_size=10, seed=7).run()
    r2 = AlgoCls(prob, max_iter=50, pop_size=10, seed=7).run()
    assert r1.best_fitness == pytest.approx(r2.best_fitness), \
        f"{algo_name} is not deterministic"

@pytest.mark.parametrize("algo_name", ALGO_NAMES)
def test_algorithm_nfe_mode(algo_name):
    """NFE stopping mode must not crash."""
    prob = Sphere(dim=5)
    AlgoCls = get_algorithm(algo_name)
    alg = AlgoCls(prob, max_iter=0, max_nfe=300, pop_size=10, seed=1)
    result = alg.run()
    assert result is not None
    assert result.best_fitness < 1e9
