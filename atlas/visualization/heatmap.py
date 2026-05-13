"""Heatmap for algorithm × problem performance comparison."""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from atlas.core.result import Result


def plot_heatmap(
    all_results: Dict[str, Dict[str, List[Result]]],
    algo_names: Optional[List[str]] = None,
    problem_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Optional[tuple] = None,
    dpi: int = 150,
    save_path: Optional[str] = None,
    normalize: bool = True,
) -> plt.Figure:
    """Draw a heatmap of mean fitness (algorithms × problems).

    Each cell shows the normalised mean best fitness of an algorithm on a
    problem.  If *normalize* is ``True`` (default), values are normalised
    to [0, 1] per problem (column-wise min-max).

    Args:
        all_results: ``{problem_name: {algo_name: [Result, ...]}}``.
        algo_names: Ordered list of algorithm names.  Auto-detected if ``None``.
        problem_names: Ordered list of problem names.  Auto-detected if ``None``.
        title: Plot title.
        figsize: Figure size.
        dpi: Resolution.
        save_path: If provided, save figure to this path.
        normalize: Whether to column-normalise the matrix.

    Returns:
        The matplotlib Figure object.
    """
    if problem_names is None:
        problem_names = sorted(all_results.keys())
    if algo_names is None:
        first_prob = problem_names[0]
        algo_names = sorted(all_results[first_prob].keys())

    n_algos = len(algo_names)
    n_probs = len(problem_names)

    # Build raw matrix
    raw = np.zeros((n_algos, n_probs))
    for j, p_name in enumerate(problem_names):
        for i, a_name in enumerate(algo_names):
            results = all_results[p_name][a_name]
            raw[i, j] = np.mean([r.best_fitness for r in results])

    # Column-wise min-max normalisation
    if normalize:
        col_min = raw.min(axis=0)
        col_max = raw.max(axis=0)
        denom = col_max - col_min
        denom[denom < 1e-30] = 1.0
        data = (raw - col_min) / denom
    else:
        data = raw

    if figsize is None:
        figsize = (max(6, n_probs * 1.2), max(4, n_algos * 0.8))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(data, cmap="YlOrRd_r", aspect="auto", vmin=0, vmax=1 if normalize else None)

    # Annotate cells with raw values
    for i in range(n_algos):
        for j in range(n_probs):
            text = f"{raw[i, j]:.2e}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    ax.set_xticks(np.arange(n_probs))
    ax.set_yticks(np.arange(n_algos))
    ax.set_xticklabels(problem_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(algo_names, fontsize=9)
    ax.set_xlabel("Problem")
    ax.set_ylabel("Algorithm")
    ax.set_title(title or "Algorithm × Problem Performance (lower is better)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Normalised Fitness" if normalize else "Fitness")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig
