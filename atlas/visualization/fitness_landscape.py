"""Fitness landscape visualisation for 2-D problems."""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from atlas.core.base_problem import BaseProblem


def plot_fitness_landscape(
    problem: BaseProblem,
    resolution: int = 100,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    trajectory: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    figsize: tuple = (8, 6),
    dpi: int = 150,
    save_path: Optional[str] = None,
    plot_3d: bool = False,
) -> plt.Figure:
    """Plot the fitness landscape of a 2-D problem as a contour or surface.

    Only the first two dimensions are used; remaining dimensions are fixed
    at their midpoint.

    Args:
        problem: A 2-D optimisation problem.
        resolution: Grid resolution per axis.
        x_range: ``(min, max)`` for the x-axis.  Auto-detected from bounds.
        y_range: ``(min, max)`` for the y-axis.  Auto-detected from bounds.
        trajectory: Optional ``(N, 2)`` array of (x, y) positions to overlay
            as the algorithm search path.
        title: Plot title.
        figsize: Figure size.
        dpi: Resolution.
        save_path: If provided, save figure to this path.
        plot_3d: If ``True``, draw a 3-D surface instead of a contour plot.

    Returns:
        The matplotlib Figure object.

    Raises:
        ValueError: If the problem dimension is less than 2.
    """
    if problem.get_dim() < 2:
        raise ValueError("Fitness landscape visualisation requires dim >= 2.")

    lb, ub = problem.get_bounds()
    x_min = x_range[0] if x_range else lb[0]
    x_max = x_range[1] if x_range else ub[0]
    y_min = y_range[0] if y_range else lb[1]
    y_max = y_range[1] if y_range else ub[1]

    xx = np.linspace(x_min, x_max, resolution)
    yy = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(xx, yy)

    # Evaluate grid (fix extra dims at midpoint)
    mid = (lb + ub) / 2.0
    Z = np.empty_like(X)
    for i in range(resolution):
        for j in range(resolution):
            pt = mid.copy()
            pt[0] = X[i, j]
            pt[1] = Y[i, j]
            Z[i, j] = problem.evaluate_with_penalty(pt)

    if plot_3d:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8, edgecolor="none")
        if trajectory is not None:
            traj_f = []
            for pt in trajectory:
                full = mid.copy()
                full[0], full[1] = pt[0], pt[1]
                traj_f.append(problem.evaluate_with_penalty(full))
            ax.plot(
                trajectory[:, 0], trajectory[:, 1], traj_f,
                "r.-", markersize=4, linewidth=1, label="Search path",
            )
            ax.legend()
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_zlabel("Fitness")
        ax.set_title(title or f"Fitness Landscape – {problem.get_name()}")
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 30)
        cs = ax.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.8)
        ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.3, alpha=0.4)
        fig.colorbar(cs, ax=ax, shrink=0.8, label="Fitness")

        if trajectory is not None:
            ax.plot(
                trajectory[:, 0], trajectory[:, 1],
                "r.-", markersize=3, linewidth=1, label="Search path",
            )
            ax.legend()

        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_title(title or f"Fitness Landscape – {problem.get_name()}")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig
