"""Convergence curve plotting utilities."""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from atlas.core.result import Result


def _apply_robust_yscale(ax: plt.Axes, all_values: np.ndarray) -> None:
    """Apply log, symlog, or linear y-scale depending on data signs and ranges."""
    finite_vals = all_values[np.isfinite(all_values)]
    if len(finite_vals) == 0:
        return

    min_val = np.min(finite_vals)
    max_val = np.max(finite_vals)

    if min_val > 0 and (max_val / max(min_val, 1e-300)) > 50:
        ax.set_yscale("log")
    elif min_val <= 0 and (max_val - min_val) > 100:
        ax.set_yscale("symlog", linthresh=1.0)
    else:
        ax.set_yscale("linear")


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
    curve = np.asarray(result.convergence_curve, dtype=float)
    ax.plot(curve, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    _apply_robust_yscale(ax, curve)
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

    Robust against runs with varying iteration counts and negative/zero values.

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
    all_plotted_values = []

    for algo_name, results in results_dict.items():
        if not results:
            continue

        raw_curves = [r.convergence_curve for r in results if len(r.convergence_curve) > 0]
        if not raw_curves:
            continue

        # Handle uneven lengths by padding to maximum length
        max_len = max(len(c) for c in raw_curves)
        padded_curves = []
        for c in raw_curves:
            c_arr = np.asarray(c, dtype=float)
            if len(c_arr) < max_len:
                pad = np.full(max_len - len(c_arr), c_arr[-1] if len(c_arr) > 0 else np.nan)
                c_arr = np.concatenate([c_arr, pad])
            padded_curves.append(c_arr)

        curves = np.array(padded_curves)
        mean_curve = np.nanmean(curves, axis=0)
        std_curve = np.nanstd(curves, axis=0)
        iters = np.arange(len(mean_curve))

        all_plotted_values.extend(mean_curve)

        ax.plot(iters, mean_curve, linewidth=1.8, label=algo_name)
        ax.fill_between(
            iters,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    if all_plotted_values:
        _apply_robust_yscale(ax, np.array(all_plotted_values))
    ax.set_title(title or "Convergence Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig
