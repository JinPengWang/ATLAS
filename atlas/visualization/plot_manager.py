"""Unified plot manager for ATLAS experiments."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt

from atlas.core.result import Result


class PlotManager:
    """Central manager for generating and saving plots.

    Provides a consistent interface for creating all supported plot types
    and optionally saving them.

    Args:
        style: Matplotlib style name (default ``'seaborn-v0_8-whitegrid'``).
    """

    def __init__(self, style: str = "seaborn-v0_8-whitegrid") -> None:
        self.style = style
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("ggplot")

    # ------------------------------------------------------------------
    # Convergence
    # ------------------------------------------------------------------
    def plot_convergence(
        self,
        results_dict: Dict[str, List[Result]],
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        dpi: int = 150,
        figsize: tuple = (10, 6),
    ) -> Optional[plt.Figure]:
        """Generate a convergence comparison plot.

        Args:
            results_dict: ``{algo_name: [Result, ...]}``.
            title: Plot title.
            save_path: File path to save, or ``None``.
            dpi: Resolution.
            figsize: Figure size.

        Returns:
            The Figure object, or ``None`` if input is empty.
        """
        if not results_dict:
            return None
        from atlas.visualization.convergence import plot_convergence_comparison

        fig = plot_convergence_comparison(
            results_dict, title=title, figsize=figsize, dpi=dpi, save_path=save_path,
        )
        return fig

    # ------------------------------------------------------------------
    # Box-plot
    # ------------------------------------------------------------------
    def plot_boxplot(
        self,
        results_dict: Dict[str, List[Result]],
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        dpi: int = 150,
        figsize: tuple = (8, 5),
    ) -> Optional[plt.Figure]:
        """Generate a box-plot comparison.

        Args:
            results_dict: ``{algo_name: [Result, ...]}``.
            title: Plot title.
            save_path: File path to save, or ``None``.
            dpi: Resolution.
            figsize: Figure size.

        Returns:
            The Figure object, or ``None`` if input is empty.
        """
        if not results_dict:
            return None
        from atlas.visualization.boxplot import plot_boxplot

        fig = plot_boxplot(
            results_dict, title=title, figsize=figsize, dpi=dpi, save_path=save_path,
        )
        return fig

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------
    def plot_heatmap(
        self,
        all_results: Dict[str, Dict[str, List[Result]]],
        algo_names: Optional[List[str]] = None,
        problem_names: Optional[List[str]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> Optional[plt.Figure]:
        """Generate a heatmap of algorithm × problem performance.

        Args:
            all_results: ``{problem: {algo: [Result, ...]}}``.
            algo_names: Algorithm names (auto-detected if ``None``).
            problem_names: Problem names (auto-detected if ``None``).
            title: Plot title.
            save_path: File path to save, or ``None``.
            dpi: Resolution.

        Returns:
            The Figure object, or ``None`` if input is empty.
        """
        if not all_results:
            return None
        from atlas.visualization.heatmap import plot_heatmap

        fig = plot_heatmap(
            all_results,
            algo_names=algo_names,
            problem_names=problem_names,
            title=title,
            dpi=dpi,
            save_path=save_path,
        )
        return fig

    # ------------------------------------------------------------------
    # Fitness landscape
    # ------------------------------------------------------------------
    def plot_landscape(
        self,
        problem: Any,
        trajectory: Optional[Any] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        dpi: int = 150,
        figsize: tuple = (8, 6),
        plot_3d: bool = False,
    ) -> Optional[plt.Figure]:
        """Generate a fitness landscape plot for a 2-D problem.

        Args:
            problem: A 2-D :class:`BaseProblem` instance.
            trajectory: Optional search trajectory array.
            title: Plot title.
            save_path: File path to save, or ``None``.
            dpi: Resolution.
            figsize: Figure size.
            plot_3d: Draw a 3-D surface.

        Returns:
            The Figure object.
        """
        from atlas.visualization.fitness_landscape import plot_fitness_landscape

        fig = plot_fitness_landscape(
            problem,
            trajectory=trajectory,
            title=title,
            save_path=save_path,
            dpi=dpi,
            figsize=figsize,
            plot_3d=plot_3d,
        )
        return fig

    # ------------------------------------------------------------------
    # Diversity & Exploration-Exploitation
    # ------------------------------------------------------------------
    def plot_diversity(
        self,
        tracker: Any,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        dpi: int = 150,
        figsize: tuple = (9, 4),
    ) -> Optional[plt.Figure]:
        """Generate a population diversity curve plot.

        Args:
            tracker: :class:`DiversityTracker` instance.
            title: Plot title.
            save_path: File path to save, or ``None``.
            dpi: Resolution.
            figsize: Figure size.

        Returns:
            The Figure object.
        """
        from atlas.visualization.diversity import plot_diversity

        return plot_diversity(
            tracker,
            title=title,
            save_path=save_path,
            dpi=dpi,
            figsize=figsize,
        )

    def plot_exploration_exploitation(
        self,
        tracker: Any,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        dpi: int = 150,
        figsize: tuple = (9, 4),
    ) -> Optional[plt.Figure]:
        """Generate an exploration vs exploitation stacked area plot.

        Args:
            tracker: :class:`DiversityTracker` instance.
            title: Plot title.
            save_path: File path to save, or ``None``.
            dpi: Resolution.
            figsize: Figure size.

        Returns:
            The Figure object.
        """
        from atlas.visualization.diversity import plot_exploration_exploitation

        return plot_exploration_exploitation(
            tracker,
            title=title,
            save_path=save_path,
            dpi=dpi,
            figsize=figsize,
        )

