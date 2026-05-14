"""Unit tests for ATLAS algorithms.

Each algorithm is tested on the Sphere function for 10 iterations to
verify it runs without errors and improves fitness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is importable
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


@pytest.fixture
def short_run_params():
    return {"max_iter": 10, "pop_size": 10, "seed": 0, "verbose": False}


# ---- Registration tests ---------------------------------------------
class TestRegistration:
    def test_list_algorithms_not_empty(self):
        algos = list_algorithms()
        assert len(algos) > 0

    def test_all_expected_registered(self):
        expected = {
            "pso", "ga", "de", "sa", "woa", "aco", "dp",
            "tjo", "lgc", "ppo", "lea", "psa",
        }
        registered = set(list_algorithms())
        assert expected.issubset(registered)

    def test_get_algorithm_unknown_raises(self):
        with pytest.raises(KeyError):
            get_algorithm("nonexistent_algo")


# ---- Run tests ------------------------------------------------------
class TestAlgorithmRun:
    """Test that every algorithm can run on Sphere for 10 iterations."""

    @pytest.mark.parametrize("algo_name", [
        "pso", "ga", "de", "sa", "woa", "aco", "dp",
        "tjo", "lgc", "ppo", "lea", "psa",
    ])
    def test_algorithm_runs(self, algo_name, sphere_problem, short_run_params):
        cls = get_algorithm(algo_name)
        algo = cls(problem=sphere_problem, **short_run_params)
        result = algo.run(run_id=0)

        assert isinstance(result, Result)
        assert result.algorithm_name == algo.get_name()
        assert result.problem_name == "sphere"
        assert result.iterations == 10
        assert len(result.convergence_curve) == 10
        assert np.isfinite(result.best_fitness)

    @pytest.mark.parametrize("algo_name", [
        "pso", "ga", "de", "sa", "woa", "aco", "dp",
        "tjo", "lgc", "ppo", "lea", "psa",
    ])
    def test_algorithm_improves(self, algo_name, sphere_problem, short_run_params):
        """Fitness should not increase over iterations."""
        cls = get_algorithm(algo_name)
        algo = cls(problem=sphere_problem, **short_run_params)
        result = algo.run(run_id=0)
        curve = result.convergence_curve
        for i in range(1, len(curve)):
            assert curve[i] <= curve[i - 1] + 1e-12, (
                f"{algo_name}: fitness increased at iter {i}: "
                f"{curve[i-1]:.6e} -> {curve[i]:.6e}"
            )

    @pytest.mark.parametrize("algo_name", ["pso", "de", "dp", "tjo", "lgc", "ppo", "lea", "psa"])
    def test_reproducibility(self, algo_name, sphere_problem, short_run_params):
        """Same seed should produce identical results."""
        cls = get_algorithm(algo_name)
        r1 = cls(problem=sphere_problem, **short_run_params).run()
        r2 = cls(problem=sphere_problem, **short_run_params).run()
        assert r1.best_fitness == pytest.approx(r2.best_fitness, rel=1e-12)
        assert np.allclose(r1.best_solution, r2.best_solution)


# ---- Direct run -----------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
