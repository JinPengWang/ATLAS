"""Unit tests for population diversity tracking and exploration-exploitation analysis."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

# Ensure project root is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from atlas.algorithms.iteration.pso import PSO
from atlas.core.result import Result
from atlas.problems.benchmark.unimodal import Sphere
from atlas.visualization.diversity import (
    DiversityTracker,
    plot_diversity,
    plot_exploration_exploitation,
)
from atlas.visualization.plot_manager import PlotManager


def test_diversity_decreases():
    """Verify that diversity strictly decreases as population converges."""
    tracker = DiversityTracker()
    rng = np.random.default_rng(42)

    # Initial highly diverse population in [-100, 100]
    pop_diverse = rng.uniform(-100.0, 100.0, size=(50, 10))
    # Semi-converged population in [-10, 10]
    pop_medium = rng.uniform(-10.0, 10.0, size=(50, 10))
    # Highly converged population in [-0.01, 0.01]
    pop_converged = rng.uniform(-0.01, 0.01, size=(50, 10))

    tracker.update(pop_diverse)
    tracker.update(pop_medium)
    tracker.update(pop_converged)

    curve = tracker.get_diversity_curve()
    assert len(curve) == 3
    assert curve[0] > curve[1] > curve[2]


def test_exploitation_plus_exploration_equals_100():
    """Verify %Exploration + %Exploitation strictly equals 100.0 (tolerance 1e-6)."""
    tracker = DiversityTracker()
    rng = np.random.default_rng(123)

    for scale in [100.0, 80.0, 50.0, 20.0, 5.0, 0.5, 0.01]:
        pop = rng.uniform(-scale, scale, size=(30, 8))
        tracker.update(pop)

    exp_curve = tracker.get_exploration_curve()
    expl_curve = tracker.get_exploitation_curve()

    assert len(exp_curve) == 7
    assert len(expl_curve) == 7

    for exp, expl in zip(exp_curve, expl_curve):
        assert abs((exp + expl) - 100.0) < 1e-6


def test_plot_diversity_returns_figure():
    """Verify plot_diversity returns a matplotlib Figure and can save to disk."""
    tracker = DiversityTracker()
    rng = np.random.default_rng(42)

    for _ in range(5):
        tracker.update(rng.uniform(-10.0, 10.0, size=(20, 4)))

    fig = plot_diversity(tracker, title="Test Diversity Plot")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Test saving
    with tempfile.TemporaryDirectory() as tmpdir:
        save_file = str(Path(tmpdir) / "diversity_test.png")
        fig_saved = plot_diversity(tracker, save_path=save_file)
        assert Path(save_file).exists()
        assert Path(save_file).stat().st_size > 0
        plt.close(fig_saved)


def test_plot_exploration_exploitation_returns_figure():
    """Verify plot_exploration_exploitation returns a matplotlib Figure and can save to disk."""
    tracker = DiversityTracker()
    rng = np.random.default_rng(42)

    for _ in range(5):
        tracker.update(rng.uniform(-10.0, 10.0, size=(20, 4)))

    fig = plot_exploration_exploitation(tracker, title="Test Exploration vs Exploitation")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Test saving
    with tempfile.TemporaryDirectory() as tmpdir:
        save_file = str(Path(tmpdir) / "exp_expl_test.png")
        fig_saved = plot_exploration_exploitation(tracker, save_path=save_file)
        assert Path(save_file).exists()
        assert Path(save_file).stat().st_size > 0
        plt.close(fig_saved)


def test_track_diversity_in_run():
    """Verify running PSO with track_diversity=True records non-empty diversity_curve in Result.extra."""
    problem = Sphere(dim=5)
    max_iter = 15
    pop_size = 12

    algo = PSO(
        problem=problem,
        max_iter=max_iter,
        pop_size=pop_size,
        seed=42,
        track_diversity=True,
    )
    result = algo.run(run_id=0)

    assert isinstance(result, Result)
    assert "diversity_curve" in result.extra
    div_curve = result.extra["diversity_curve"]
    assert isinstance(div_curve, list)
    assert len(div_curve) == max_iter
    assert all(np.isfinite(div_curve))
    assert all(d >= 0.0 for d in div_curve)

    assert algo._diversity_tracker is not None
    assert len(algo._diversity_tracker.get_diversity_curve()) == max_iter

    # Verify track_diversity=False does not populate diversity_curve in extra
    algo_no_track = PSO(
        problem=problem,
        max_iter=max_iter,
        pop_size=pop_size,
        seed=42,
        track_diversity=False,
    )
    res_no_track = algo_no_track.run(run_id=0)
    assert "diversity_curve" not in res_no_track.extra
    assert algo_no_track._diversity_tracker is None


def test_single_point_population_diversity_zero():
    """Verify diversity is exactly 0.0 when all individuals in population are identical."""
    tracker = DiversityTracker()
    pop = np.full((30, 5), 42.0)
    tracker.update(pop)

    div_curve = tracker.get_diversity_curve()
    assert len(div_curve) == 1
    assert div_curve[0] == 0.0

    exp_curve = tracker.get_exploration_curve()
    expl_curve = tracker.get_exploitation_curve()
    assert exp_curve == [0.0]
    assert expl_curve == [100.0]


def test_empty_tracker_curves_and_plots():
    """Verify tracker handles empty state gracefully."""
    tracker = DiversityTracker()
    assert tracker.get_diversity_curve() == []
    assert tracker.get_exploration_curve() == []
    assert tracker.get_exploitation_curve() == []

    fig1 = plot_diversity(tracker)
    assert isinstance(fig1, plt.Figure)
    plt.close(fig1)

    fig2 = plot_exploration_exploitation(tracker)
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)


def test_plot_manager_diversity_methods():
    """Verify PlotManager wrapper methods for diversity visualization."""
    pm = PlotManager()
    tracker = DiversityTracker()
    rng = np.random.default_rng(42)
    for _ in range(5):
        tracker.update(rng.uniform(-5.0, 5.0, size=(10, 3)))

    fig1 = pm.plot_diversity(tracker, title="PM Diversity")
    assert isinstance(fig1, plt.Figure)
    plt.close(fig1)

    fig2 = pm.plot_exploration_exploitation(tracker, title="PM Exp vs Expl")
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)

