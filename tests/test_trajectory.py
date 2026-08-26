"""Tests for 2D trajectory and landscape visualization."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from atlas.visualization.trajectory import plot_fitness_landscape_2d, plot_trajectory_2d


def test_landscape_2d_sphere():
    """Should draw contour map for 2D Sphere problem."""
    from atlas.problems.benchmark.unimodal import Sphere

    prob = Sphere(dim=2)
    fig, ax = plt.subplots()
    returned_ax = plot_fitness_landscape_2d(prob, resolution=20, ax=ax)
    assert returned_ax is ax  # should return same ax
    assert len(ax.collections) > 0  # contours drawn
    plt.close(fig)


def test_landscape_2d_raises_for_wrong_dim():
    """Should raise ValueError when problem dimension is not 2."""
    from atlas.problems.benchmark.unimodal import Sphere

    prob = Sphere(dim=5)
    with pytest.raises(ValueError, match="dim=2"):
        plot_fitness_landscape_2d(prob, resolution=10)


def test_trajectory_2d_returns_figure():
    """Should return a matplotlib Figure with trajectory overlaid."""
    from atlas.problems.benchmark.unimodal import Sphere

    prob = Sphere(dim=2)
    rng = np.random.default_rng(0)
    history = [(i, rng.uniform(-5.12, 5.12, size=(10, 2))) for i in range(5)]
    fig = plot_trajectory_2d(history, prob, resolution=15)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_trajectory_2d_saves_file(tmp_path):
    """Should correctly save trajectory plot to the specified path."""
    from atlas.problems.benchmark.unimodal import Sphere

    prob = Sphere(dim=2)
    rng = np.random.default_rng(1)
    history = [(0, rng.uniform(-5.12, 5.12, size=(5, 2)))]
    out = tmp_path / "traj.png"
    fig = plot_trajectory_2d(history, prob, resolution=10, save_path=str(out))
    assert out.exists()
    plt.close(fig)


def test_trajectory_2d_raises_for_wrong_dim():
    """Should raise ValueError when problem dimension is not 2."""
    from atlas.problems.benchmark.unimodal import Sphere

    prob = Sphere(dim=5)
    history = [(0, np.ones((5, 5)))]
    with pytest.raises(ValueError, match="dim=2"):
        plot_trajectory_2d(history, prob)


def test_landscape_2d_creates_own_axes():
    """When ax=None, a new figure/axes should be created."""
    from atlas.problems.benchmark.unimodal import Sphere

    prob = Sphere(dim=2)
    ax = plot_fitness_landscape_2d(prob, resolution=10)
    assert ax is not None
    plt.close("all")
