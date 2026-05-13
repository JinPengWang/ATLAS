"""Unit tests for NFE-based algorithm variants."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import atlas  # noqa: E402 – triggers registration
from atlas.core.result import Result  # noqa: E402
from atlas.problems.benchmark.unimodal import Sphere  # noqa: E402
from atlas.utils.registry import get_algorithm, list_algorithms  # noqa: E402


# ---- Fixtures -------------------------------------------------------
@pytest.fixture
def sphere_problem():
    return Sphere(dim=10)


# ---- Registration tests ---------------------------------------------
class TestRegistration:
    def test_nfe_algorithms_registered(self):
        expected = {"pso_nfe", "ga_nfe", "de_nfe", "sa_nfe", "woa_nfe", "aco_nfe"}
        registered = set(list_algorithms())
        assert expected.issubset(registered)

    def test_total_algorithm_count(self):
        """Should have 12 algorithms (6 iteration + 6 NFE)."""
        algos = list_algorithms()
        assert len(algos) >= 12


# ---- Run tests ------------------------------------------------------
class TestNFEAlgorithmRun:
    """Test that every NFE algorithm can run on Sphere."""

    @pytest.mark.parametrize("algo_name", [
        "pso_nfe", "ga_nfe", "de_nfe", "sa_nfe", "woa_nfe", "aco_nfe",
    ])
    def test_algorithm_runs(self, algo_name, sphere_problem):
        cls = get_algorithm(algo_name)
        algo = cls(problem=sphere_problem, max_nfe=500, pop_size=10, seed=0)
        result = algo.run(run_id=0)

        assert isinstance(result, Result)
        assert result.algorithm_name.endswith("_NFE")
        assert result.problem_name == "sphere"
        assert np.isfinite(result.best_fitness)
        assert result.nfe > 0
        assert result.nfe <= 600  # allow some overshoot per iteration
        assert len(result.convergence_curve) > 0

    @pytest.mark.parametrize("algo_name", [
        "pso_nfe", "ga_nfe", "de_nfe", "sa_nfe", "woa_nfe", "aco_nfe",
    ])
    def test_algorithm_improves(self, algo_name, sphere_problem):
        """Fitness should not increase over iterations."""
        cls = get_algorithm(algo_name)
        algo = cls(problem=sphere_problem, max_nfe=500, pop_size=10, seed=0)
        result = algo.run(run_id=0)
        curve = result.convergence_curve
        for i in range(1, len(curve)):
            assert curve[i] <= curve[i - 1] + 1e-12, (
                f"{algo_name}: fitness increased at iter {i}: "
                f"{curve[i-1]:.6e} -> {curve[i]:.6e}"
            )

    @pytest.mark.parametrize("algo_name", ["pso_nfe", "de_nfe"])
    def test_reproducibility(self, algo_name, sphere_problem):
        """Same seed should produce identical results."""
        cls = get_algorithm(algo_name)
        r1 = cls(problem=sphere_problem, max_nfe=300, pop_size=10, seed=42).run()
        r2 = cls(problem=sphere_problem, max_nfe=300, pop_size=10, seed=42).run()
        assert r1.best_fitness == pytest.approx(r2.best_fitness, rel=1e-12)
        assert np.allclose(r1.best_solution, r2.best_solution)


# ---- Direct run -----------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
