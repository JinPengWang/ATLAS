"""Unit tests for ATLAS problem definitions.

Verifies that each benchmark function evaluates correctly at known
optimal points and that bounds are consistent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import atlas  # noqa: E402
from atlas.utils.registry import get_problem, list_problems  # noqa: E402


# ---- Registration tests ---------------------------------------------
class TestRegistration:
    def test_list_problems_not_empty(self):
        probs = list_problems()
        assert len(probs) > 0

    def test_expected_problems_registered(self):
        expected = {
            "sphere", "rosenbrock", "schwefel222", "quartic",
            "rastrigin", "ackley", "griewank", "levy", "schwefel",
            "michalewicz", "pressure_vessel",
        }
        registered = set(list_problems())
        assert expected.issubset(registered)


# ---- Evaluation tests -----------------------------------------------
class TestBenchmarkEvaluation:
    """Test that each benchmark function returns the correct optimum."""

    @pytest.mark.parametrize(
        "prob_name, dim, opt_loc_offset",
        [
            ("sphere", 10, 0.0),
            ("rosenbrock", 10, 0.0),
            ("schwefel222", 10, 0.0),
            ("quartic", 10, 0.0),
            ("rastrigin", 10, 0.0),
            ("ackley", 10, 0.0),
            ("griewank", 10, 0.0),
            ("levy", 10, 0.0),
        ],
    )
    def test_optimum_value(self, prob_name, dim, opt_loc_offset):
        cls = get_problem(prob_name)
        prob = cls(dim=dim)
        opt_loc = prob.get_optimum_location()
        if opt_loc is None:
            pytest.skip("Optimum location unknown")
        val = prob.evaluate(opt_loc)
        expected = prob.get_optimum()
        if expected is None:
            pytest.skip("Optimum value unknown")
        assert val == pytest.approx(expected, abs=1e-6), (
            f"{prob_name}: expected {expected}, got {val}"
        )

    @pytest.mark.parametrize("prob_name", [
        "sphere", "rosenbrock", "rastrigin", "ackley",
    ])
    def test_bounds_consistency(self, prob_name):
        cls = get_problem(prob_name)
        prob = cls(dim=5)
        lb, ub = prob.get_bounds()
        assert lb.shape == (5,)
        assert ub.shape == (5,)
        assert np.all(lb < ub)

    @pytest.mark.parametrize("prob_name", [
        "sphere", "rastrigin", "ackley",
    ])
    def test_evaluate_finite(self, prob_name):
        cls = get_problem(prob_name)
        prob = cls(dim=5)
        lb, ub = prob.get_bounds()
        x = (lb + ub) / 2.0
        val = prob.evaluate(x)
        assert np.isfinite(val)


# ---- Boundary strategy tests ----------------------------------------
class TestBoundaryStrategies:
    def test_clip(self):
        from atlas.problems.benchmark.unimodal import Sphere
        prob = Sphere(dim=2, boundary_strategy="clip")
        x = np.array([200.0, -200.0])
        clamped = prob.clamp(x)
        assert np.all(clamped <= prob._ub)
        assert np.all(clamped >= prob._lb)

    def test_reflect(self):
        from atlas.problems.benchmark.unimodal import Sphere
        prob = Sphere(dim=2, boundary_strategy="reflect")
        x = np.array([150.0, -150.0])
        clamped = prob.clamp(x)
        assert np.all(clamped <= prob._ub)
        assert np.all(clamped >= prob._lb)

    def test_penalty(self):
        from atlas.problems.benchmark.unimodal import Sphere
        prob = Sphere(dim=2, boundary_strategy="penalty", penalty_weight=1e6)
        x = np.array([200.0, 0.0])
        val = prob.evaluate_with_penalty(x)
        assert val > 1e5  # penalty should dominate


# ---- Custom problem test --------------------------------------------
class TestPressureVessel:
    def test_known_optimum(self):
        from atlas.problems.custom.example_custom import PressureVessel
        prob = PressureVessel()
        opt_loc = prob.get_optimum_location()
        val = prob.evaluate_with_penalty(opt_loc)
        assert val < 6100.0  # close to known optimum


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
