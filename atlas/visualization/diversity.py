"""Population diversity tracking and exploration-exploitation analysis.

Provides :class:`DiversityTracker` for recording population diversity during
optimization runs, and plotting utilities for visualizing diversity curves and
exploration versus exploitation percentages.

Reference:
    Hussain, K., Salleh, M. N. M., Cheng, S., & Shi, Y. (2019).
    Metaheuristic research: a comprehensive survey.
    *Artificial Intelligence Review*, 52(4), 2191-2233.
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


class DiversityTracker:
    """Tracks population diversity across iterations.

    Diversity metric (Hussain et al. 2019):
        Div(t) = (1 / (N * D)) * sum_i sum_j |x_ij(t) - mean_j(t)|

    Attributes:
        _diversity_curve: Internal list storing diversity values per iteration.
    """

    def __init__(self) -> None:
        """Initialize an empty DiversityTracker."""
        self._diversity_curve: List[float] = []

    def update(self, population: np.ndarray) -> None:
        """Record diversity of current population. Call once per iteration.

        Args:
            population: 2-D array of shape ``(pop_size, dim)`` representing
                the current population coordinates.
        """
        pop = np.asarray(population, dtype=float)
        if pop.ndim != 2 or pop.shape[0] == 0 or pop.shape[1] == 0:
            self._diversity_curve.append(0.0)
            return

        mean_j = np.mean(pop, axis=0)
        div = float(np.mean(np.abs(pop - mean_j)))
        self._diversity_curve.append(div)

    def get_diversity_curve(self) -> List[float]:
        """Get the recorded diversity curve.

        Returns:
            List of diversity values for each tracked iteration.
        """
        return list(self._diversity_curve)

    def get_exploration_curve(self) -> List[float]:
        """Compute the exploration percentage curve.

        Formula:
            %Exploration = Div(t) / max(Div) * 100.

        When all diversity values are 0 (or when empty), exploration is 0.

        Returns:
            List of exploration percentage values in [0, 100].
        """
        if not self._diversity_curve:
            return []

        max_div = max(self._diversity_curve)
        if max_div <= 0.0:
            return [0.0] * len(self._diversity_curve)

        return [float((d / max_div) * 100.0) for d in self._diversity_curve]

    def get_exploitation_curve(self) -> List[float]:
        """Compute the exploitation percentage curve.

        Formula:
            %Exploitation = 100 - %Exploration.

        Exploration and exploitation percentages strictly sum to 100.0.

        Returns:
            List of exploitation percentage values in [0, 100].
        """
        exploration = self.get_exploration_curve()
        return [float(100.0 - exp) for exp in exploration]


def plot_diversity(
    tracker: DiversityTracker,
    title: Optional[str] = None,
    figsize: tuple = (9, 4),
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Plot raw diversity curve with fill_between shading.

    Args:
        tracker: :class:`DiversityTracker` containing recorded diversity values.
        title: Optional plot title. Defaults to ``'Population Diversity'``.
        figsize: Figure dimensions (width, height) in inches.
        save_path: Optional file path to save the generated figure.
        dpi: Resolution of the figure in dots per inch.

    Returns:
        The matplotlib Figure object.
    """
    curve = tracker.get_diversity_curve()
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if len(curve) > 0:
        iters = np.arange(len(curve))
        arr = np.asarray(curve, dtype=float)
        ax.plot(iters, arr, label="Diversity", color="#1f77b4", linewidth=1.5)
        ax.fill_between(iters, 0, arr, color="#1f77b4", alpha=0.2)
        ax.set_xlim(0, max(1, len(curve) - 1))
        max_y = float(np.max(arr)) * 1.05 if float(np.max(arr)) > 0 else 1.0
        ax.set_ylim(bottom=0.0, top=max_y)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Diversity (Div)")
    ax.set_title(title or "Population Diversity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_exploration_exploitation(
    tracker: DiversityTracker,
    title: Optional[str] = None,
    figsize: tuple = (9, 4),
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Plot stacked area chart: %Exploration (blue) + %Exploitation (orange) = 100%.

    Args:
        tracker: :class:`DiversityTracker` containing recorded diversity values.
        title: Optional plot title. Defaults to ``'Exploration vs Exploitation'``.
        figsize: Figure dimensions (width, height) in inches.
        save_path: Optional file path to save the generated figure.
        dpi: Resolution of the figure in dots per inch.

    Returns:
        The matplotlib Figure object.
    """
    exploration = tracker.get_exploration_curve()
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if len(exploration) > 0:
        iters = np.arange(len(exploration))
        exp_arr = np.asarray(exploration, dtype=float)

        ax.fill_between(
            iters,
            0,
            exp_arr,
            label="Exploration (%)",
            color="#1f77b4",
            alpha=0.6,
        )
        ax.fill_between(
            iters,
            exp_arr,
            100.0,
            label="Exploitation (%)",
            color="#ff7f0e",
            alpha=0.6,
        )
        ax.plot(iters, exp_arr, color="#1f77b4", linewidth=1.2)
        ax.set_xlim(0, max(1, len(exploration) - 1))
        ax.legend(loc="upper right")

    ax.set_ylim(0, 100)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Percentage (%)")
    ax.set_title(title or "Exploration vs Exploitation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig
