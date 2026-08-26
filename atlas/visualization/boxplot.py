"""Box-plot utilities for algorithm comparison."""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from atlas.core.result import Result


def plot_boxplot(
    results_dict: Dict[str, List[Result]],
    title: Optional[str] = None,
    figsize: tuple = (8, 5),
    dpi: int = 150,
    save_path: Optional[str] = None,
    log_scale: bool = True,
) -> plt.Figure:
    """Draw a standard box-plot comparing final fitness across algorithms.

    Args:
        results_dict: ``{algo_name: [Result, ...]}`` mapping.
        title: Plot title.
        figsize: Figure size.
        dpi: Resolution.
        save_path: If provided, save figure to this path.
        log_scale: Attempt log/symlog scale on y-axis if appropriate.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    labels = []
    data = []
    all_vals = []
    for algo_name, results in results_dict.items():
        labels.append(algo_name)
        vals = [r.best_fitness for r in results if np.isfinite(r.best_fitness)]
        data.append(vals)
        all_vals.extend(vals)

    if not data or not all_vals:
        return fig

    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 6},
    )

    # Color each box differently
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(data), 1)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    arr = np.array(all_vals)
    min_val, max_val = np.min(arr), np.max(arr)
    if log_scale:
        if min_val > 0 and (max_val / max(min_val, 1e-300)) > 50:
            ax.set_yscale("log")
        elif min_val <= 0 and (max_val - min_val) > 100:
            ax.set_yscale("symlog", linthresh=1.0)

    ax.set_ylabel("Final Best Fitness")
    ax.set_title(title or "Algorithm Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig
