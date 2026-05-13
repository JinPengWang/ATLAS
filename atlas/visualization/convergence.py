"""Convergence curve plotting utilities."""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from atlas.core.result import Result


def plot_convergence_single(
    result: Result,
    title: Optional[str] = None,
    figsize: tuple = (8, 5),
    dpi: int = 150,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot the convergence curve of a single algorithm run.

    Args:
        result: A single :class:`Result` object.
        title: Plot title.
        figsize: Figure size.
        dpi: Resolution.
        save_path: If provided, save figure to this path.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(result.convergence_curve, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness (log scale)")
    ax.set_yscale("log")
    ax.set_title(title or f"{result.algorithm_name} on {result.problem_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_convergence_comparison(
    results_dict: Dict[str, List[Result]],
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
    dpi: int = 150,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot mean ± std convergence curves for multiple algorithms.

    Args:
        results_dict: ``{algo_name: [Result, ...]}`` mapping.
        title: Plot title.
        figsize: Figure size.
        dpi: Resolution.
        save_path: If provided, save figure to this path.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for algo_name, results in results_dict.items():
        curves = np.array([r.convergence_curve for r in results])
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        iters = np.arange(len(mean_curve))

        ax.plot(iters, mean_curve, linewidth=1.8, label=algo_name)
        ax.fill_between(
            iters,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness (log scale)")
    ax.set_yscale("log")
    ax.set_title(title or "Convergence Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig
