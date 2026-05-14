"""Unit tests for CEC benchmark problem wrappers.

Requires ``opfunu`` to be installed.  All tests are skipped otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    import opfunu  # noqa: F401

    HAS_OPFUNU = True
except ImportError:
    HAS_OPFUNU = False

pytestmark = pytest.mark.skipif(not HAS_OPFUNU, reason="opfunu not installed")

import atlas  # noqa: E402 – triggers registration
from atlas.utils.registry import get_problem, list_problems  # noqa: E402


# ---- Registration tests ---------------------------------------------
class TestCECRegistration:
    def test_cec2017_registered(self):
        probs = list_problems()
        assert "cec2017_f1" in probs
        assert "cec2017_f29" in probs

    def test_cec2005_registered(self):
        probs = list_problems()
        assert "cec2005_f1" in probs
        assert "cec2005_f25" in probs

    def test_cec2013_registered(self):
        probs = list_problems()
        assert "cec2013_f1" in probs
        assert "cec2013_f28" in probs

    def test_cec2014_registered(self):
        probs = list_problems()
        assert "cec2014_f1" in probs
        assert "cec2014_f30" in probs

    def test_cec2019_registered(self):
        probs = list_problems()
        assert "cec2019_f1" in probs
        assert "cec2019_f10" in probs

    def test_cec2020_registered(self):
        probs = list_problems()
        assert "cec2020_f1" in probs
        assert "cec2020_f10" in probs

    def test_cec2022_registered(self):
        probs = list_problems()
        assert "cec2022_f1" in probs
        assert "cec2022_f12" in probs


# ---- Evaluation tests -----------------------------------------------
class TestCECEvaluation:
    @pytest.mark.parametrize("prob_name", [
        "cec2017_f1", "cec2017_f10", "cec2017_f29",
        "cec2005_f1", "cec2005_f13", "cec2005_f25",
        "cec2013_f1", "cec2013_f14", "cec2013_f28",
        "cec2014_f1", "cec2014_f15", "cec2014_f30",
        "cec2020_f1", "cec2020_f5", "cec2020_f10",
        "cec2022_f1", "cec2022_f6", "cec2022_f12",
    ])
    def test_evaluate_finite(self, prob_name):
        cls = get_problem(prob_name)
        prob = cls(dim=10)
        lb, ub = prob.get_bounds()
        x = (lb + ub) / 2.0
        val = prob.evaluate(x)
        assert np.isfinite(val), f"{prob_name} returned non-finite value"

    @pytest.mark.parametrize("prob_name", [
        "cec2017_f1", "cec2005_f1", "cec2013_f1", "cec2014_f1",
    ])
    def test_bounds_consistency(self, prob_name):
        cls = get_problem(prob_name)
        prob = cls(dim=10)
        lb, ub = prob.get_bounds()
        assert lb.shape == (10,)
        assert ub.shape == (10,)
        assert np.all(lb < ub)

    @pytest.mark.parametrize("prob_name", [
        "cec2017_f1", "cec2005_f1", "cec2013_f1",
    ])
    def test_get_name(self, prob_name):
        cls = get_problem(prob_name)
        prob = cls(dim=10)
        name = prob.get_name()
        assert isinstance(name, str)
        assert len(name) > 0

    @pytest.mark.parametrize("prob_name", [
        "cec2017_f1", "cec2005_f1",
    ])
    def test_get_optimum(self, prob_name):
        cls = get_problem(prob_name)
        prob = cls(dim=10)
        opt = prob.get_optimum()
        assert opt is not None
        assert np.isfinite(opt)


# ---- CEC 2019 fixed-dim tests --------------------------------------
class TestCEC2019FixedDim:
    def test_f1_dim9(self):
        cls = get_problem("cec2019_f1")
        prob = cls(dim=9)
        assert prob.get_dim() == 9
        val = prob.evaluate(np.zeros(9))
        assert np.isfinite(val)

    def test_f2_dim16(self):
        cls = get_problem("cec2019_f2")
        prob = cls(dim=16)
        assert prob.get_dim() == 16
        val = prob.evaluate(np.zeros(16))
        assert np.isfinite(val)


# ---- Direct run -----------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
