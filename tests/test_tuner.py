"""Unit tests for automated hyperparameter tuning module."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from atlas.problems.benchmark.unimodal import Sphere
from atlas.tuning import AlgorithmTuner, Hyperparameter, HyperparameterSpace


def test_hyperparameter_space_definition():
    space = HyperparameterSpace()
    space.add_float("w", 0.1, 0.9)
    space.add_float("c1", 1.0, 2.5)
    space.add_int("pop_size", 10, 50, step=5)
    space.add_categorical("strategy", ["rand", "best"])
    assert len(space) == 4


def test_tuner_pso_sphere():
    """Tune PSO hyperparameter w and c1 on Sphere."""
    space = HyperparameterSpace()
    space.add_float("w", 0.4, 0.9)
    space.add_float("c1", 1.0, 2.0)

    prob = Sphere(dim=5)
    tuner = AlgorithmTuner(
        algorithm="pso",
        space=space,
        problems=prob,
        n_trials=5,
        n_runs_per_trial=2,
        max_iter=15,
        pop_size=10,
        sampler="random",
        seed=42,
    )
    res = tuner.tune()
    assert res.n_trials == 5
    assert "w" in res.best_params
    assert "c1" in res.best_params
    assert res.best_value > 0.0 or res.best_value == pytest.approx(0.0, abs=1e-5)


def test_tuner_plot_history():
    space = HyperparameterSpace()
    space.add_float("w", 0.5, 0.8)

    prob = Sphere(dim=3)
    tuner = AlgorithmTuner(
        algorithm="pso",
        space=space,
        problems=prob,
        n_trials=3,
        n_runs_per_trial=1,
        max_iter=10,
        pop_size=10,
        seed=42,
    )
    res = tuner.tune()
    fig = tuner.plot_optimization_history(res)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
