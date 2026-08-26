"""Unit tests for zero-dependency pure-Python CEC 2017 benchmark suite."""

import numpy as np
import pytest

from atlas.problems.cec_pure import (
    CEC2017F1,
    CEC2017F3,
    CEC2017F4,
    CEC2017F5,
    CEC2017F6,
    CEC2017F7,
    CEC2017F8,
    CEC2017F9,
    CEC2017F10,
)
from atlas.utils.registry import get_problem


@pytest.mark.parametrize(
    "func_id,name",
    [
        (1, "cec2017_f1_pure"),
        (3, "cec2017_f3_pure"),
        (4, "cec2017_f4_pure"),
        (5, "cec2017_f5_pure"),
        (6, "cec2017_f6_pure"),
        (7, "cec2017_f7_pure"),
        (8, "cec2017_f8_pure"),
        (9, "cec2017_f9_pure"),
        (10, "cec2017_f10_pure"),
    ],
)
def test_cec2017_pure_registered(func_id, name):
    cls = get_problem(name)
    assert cls is not None


@pytest.mark.parametrize(
    "cls,expected_bias",
    [
        (CEC2017F1, 100.0),
        (CEC2017F3, 300.0),
        (CEC2017F4, 400.0),
        (CEC2017F5, 500.0),
        (CEC2017F6, 600.0),
        (CEC2017F7, 700.0),
        (CEC2017F8, 800.0),
        (CEC2017F9, 900.0),
        (CEC2017F10, 1000.0),
    ],
)
def test_cec2017_optimum_at_shift(cls, expected_bias):
    prob = cls(dim=10)
    opt_loc = prob.get_optimum_location()
    assert opt_loc is not None
    val = prob.evaluate(opt_loc)
    assert abs(val - expected_bias) < 1e-4


@pytest.mark.parametrize("dim", [10, 30])
def test_cec2017_arbitrary_dimension(dim):
    prob = CEC2017F1(dim=dim)
    assert prob.get_dim() == dim
    lb, ub = prob.get_bounds()
    assert len(lb) == dim
    assert len(ub) == dim
    x = np.zeros(dim)
    val = prob.evaluate(x)
    assert np.isfinite(val)
    assert val > prob.get_optimum()


def test_cec2017_with_pso_optimizer():
    """Verify that an algorithm can optimize a pure-Python CEC problem."""
    from atlas.algorithms.iteration.pso import PSO

    prob = CEC2017F1(dim=10)
    algo = PSO(problem=prob, max_iter=30, pop_size=20, seed=42)
    res = algo.run()
    assert res.best_fitness < 1e12
    assert len(res.convergence_curve) == 30
