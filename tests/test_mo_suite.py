"""Unit tests for NSGA-II algorithm, ZDT/DTLZ problems, and MOO visualization."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from atlas.multiobjective import (
    DTLZ1,
    DTLZ2,
    NSGA2,
    ZDT1,
    ZDT2,
    ZDT3,
    ZDT4,
    ZDT6,
    calculate_igd,
    plot_mo_comparison,
    plot_pareto_front_2d,
    plot_pareto_front_3d,
)


@pytest.mark.parametrize("prob_cls", [ZDT1, ZDT2, ZDT3, ZDT4, ZDT6])
def test_zdt_problems_evaluate(prob_cls):
    prob = prob_cls(dim=10)
    lb, ub = prob.get_bounds()
    x = (lb + ub) / 2.0
    obj = prob.evaluate(x)
    assert len(obj) == 2
    assert np.all(np.isfinite(obj))
    pf = prob.get_pareto_front(n_points=50)
    assert pf is not None
    assert len(pf) > 0


@pytest.mark.parametrize("prob_cls", [DTLZ1, DTLZ2])
def test_dtlz_problems_evaluate(prob_cls):
    prob = prob_cls(dim=7, n_objectives=3)
    lb, ub = prob.get_bounds()
    x = (lb + ub) / 2.0
    obj = prob.evaluate(x)
    assert len(obj) == 3
    assert np.all(np.isfinite(obj))


def test_nsga2_runs_zdt1():
    prob = ZDT1(dim=10)
    algo = NSGA2(problem=prob, max_iter=25, pop_size=40, seed=42)
    res = algo.run()
    assert res.nfe > 0
    assert len(res.pareto_front) > 0
    assert res.pareto_front.shape[1] == 2
    assert res.pareto_solutions.shape[1] == 10

    # Calculate IGD to true front
    true_pf = prob.get_pareto_front(n_points=100)
    igd = calculate_igd(res.pareto_front, true_pf)
    assert np.isfinite(igd)
    assert igd < 2.0  # Basic convergence


def test_mo_viz_2d():
    prob = ZDT1(dim=5)
    true_pf = prob.get_pareto_front(50)
    front = true_pf + 0.05
    fig = plot_pareto_front_2d(front, true_pf=true_pf, title="ZDT1 Front")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_mo_viz_3d():
    front = np.random.rand(30, 3)
    fig = plot_pareto_front_3d(front, title="DTLZ2 3D Front")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_mo_viz_comparison():
    prob = ZDT1(dim=5)
    true_pf = prob.get_pareto_front(30)
    fronts = {
        "NSGA-II": true_pf + 0.02,
        "MOEA/D": true_pf + 0.05,
    }
    fig = plot_mo_comparison(fronts, true_pf=true_pf)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
