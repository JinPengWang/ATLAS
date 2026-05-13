"""Visualisation sub-package.

Provides convergence curves, box-plots, heatmaps, fitness landscape plots,
and a unified :class:`PlotManager`.
"""

from atlas.visualization.boxplot import plot_boxplot
from atlas.visualization.convergence import (
    plot_convergence_comparison,
    plot_convergence_single,
)
from atlas.visualization.fitness_landscape import plot_fitness_landscape
from atlas.visualization.heatmap import plot_heatmap
from atlas.visualization.plot_manager import PlotManager

__all__ = [
    "PlotManager",
    "plot_convergence_single",
    "plot_convergence_comparison",
    "plot_boxplot",
    "plot_heatmap",
    "plot_fitness_landscape",
]
