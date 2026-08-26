"""2D search trajectory and fitness landscape visualisation module."""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from atlas.core.base_problem import BaseProblem


def plot_fitness_landscape_2d(
    problem: BaseProblem,
    resolution: int = 100,
    ax: Optional[plt.Axes] = None,
    cmap: str = "viridis",
) -> plt.Axes:
    """Plot 2-D contour map of the problem fitness landscape.

    Only valid for problems with dim=2. Draws filled contours + color bar.

    Args:
        problem: A BaseProblem instance with dim=2.
        resolution: Grid resolution (resolution x resolution grid points).
        ax: Optional existing Axes to draw onto; creates new figure if None.
        cmap: Matplotlib colormap name.

    Returns:
        The Axes with the contour plot drawn.

    Raises:
        ValueError: If problem.get_dim() != 2.
    """
    if problem.get_dim() != 2:
        raise ValueError(
            f"plot_fitness_landscape_2d requires a problem with dim=2, "
            f"got dim={problem.get_dim()}."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    lb, ub = problem.get_bounds()
    x_min, x_max = float(lb[0]), float(ub[0])
    y_min, y_max = float(lb[1]), float(ub[1])

    xx = np.linspace(x_min, x_max, resolution)
    yy = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(xx, yy)

    Z = np.empty_like(X)
    for i in range(resolution):
        for j in range(resolution):
            pt = np.array([X[i, j], Y[i, j]])
            Z[i, j] = problem.evaluate_with_penalty(pt)

    cs = ax.contourf(X, Y, Z, levels=50, cmap=cmap)
    ax.contour(X, Y, Z, levels=15, colors="white", linewidths=0.4, alpha=0.5)
    fig.colorbar(cs, ax=ax, label="Fitness")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"Fitness Landscape – {problem.get_name()}")

    return ax


def plot_trajectory_2d(
    history: List[Tuple[int, np.ndarray]],
    problem: BaseProblem,
    resolution: int = 60,
    figsize: Tuple[float, float] = (8, 7),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot population snapshots overlaid on 2D fitness landscape.

    Earlier populations are more transparent, later ones more opaque.
    Colors progress from autumn colormap (early=yellow, late=red).

    Args:
        history: List of (iter_idx, population_array) tuples.
                 population_array has shape (N, 2).
        problem: A BaseProblem with dim=2.
        resolution: Landscape grid resolution.
        figsize: Figure size in inches.
        save_path: If given, save figure to this path (PNG/PDF/SVG).

    Returns:
        matplotlib Figure.

    Raises:
        ValueError: If problem.get_dim() != 2.
    """
    if problem.get_dim() != 2:
        raise ValueError(
            f"plot_trajectory_2d requires a problem with dim=2, "
            f"got dim={problem.get_dim()}."
        )

    fig, ax = plt.subplots(figsize=figsize)
    plot_fitness_landscape_2d(problem=problem, resolution=resolution, ax=ax)

    n = len(history)
    for idx, (iter_idx, pop) in enumerate(history):
        ratio = idx / max(n - 1, 1)
        color = plt.cm.autumn(ratio)
        alpha = 0.3 + 0.7 * ratio
        pop_arr = np.asarray(pop)
        ax.scatter(
            pop_arr[:, 0],
            pop_arr[:, 1],
            color=color,
            alpha=alpha,
            s=20,
            label=f"Iter {iter_idx}" if n <= 5 else None,
        )

    ax.set_title(f"Search Trajectory – {problem.get_name()} ({n} snapshots)")

    if n <= 5 and n > 0:
        ax.legend(loc="upper right", framealpha=0.8)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig
