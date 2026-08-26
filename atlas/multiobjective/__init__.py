"""Multi-Objective Optimization (MOO) framework."""

from __future__ import annotations

from atlas.multiobjective.algorithms import NSGA2
from atlas.multiobjective.base_mo_algorithm import BaseMOAlgorithm
from atlas.multiobjective.base_mo_problem import BaseMOProblem
from atlas.multiobjective.metrics import (
    calculate_gd,
    calculate_hypervolume,
    calculate_igd,
    calculate_spacing,
)
from atlas.multiobjective.mo_result import MOResult
from atlas.multiobjective.pareto import crowding_distance, dominates, non_dominated_sort
from atlas.multiobjective.problems import DTLZ1, DTLZ2, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from atlas.multiobjective.visualization import (
    plot_mo_comparison,
    plot_pareto_front_2d,
    plot_pareto_front_3d,
)

__all__ = [
    # Core
    "BaseMOProblem",
    "BaseMOAlgorithm",
    "MOResult",
    # Pareto
    "dominates",
    "non_dominated_sort",
    "crowding_distance",
    # Algorithms
    "NSGA2",
    # Problems
    "ZDT1",
    "ZDT2",
    "ZDT3",
    "ZDT4",
    "ZDT6",
    "DTLZ1",
    "DTLZ2",
    # Metrics
    "calculate_hypervolume",
    "calculate_igd",
    "calculate_gd",
    "calculate_spacing",
    # Visualization
    "plot_pareto_front_2d",
    "plot_pareto_front_3d",
    "plot_mo_comparison",
]
