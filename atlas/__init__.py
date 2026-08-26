"""ATLAS – All-in-one Toolkit for Learning and Applying metaheuristicS.

A modular, extensible platform for metaheuristic algorithm research.
"""

__version__ = "0.1.0"

# Trigger auto-registration of all algorithms and problems
import atlas.algorithms  # noqa: F401
import atlas.problems  # noqa: F401

from atlas.core import (
    BaseAlgorithm,
    BaseProblem,
    Experiment,
    ExperimentConfig,
    Result,
)
from atlas.report import generate_html_report
from atlas.utils import (
    ExperimentSaver,
    get_algorithm,
    get_problem,
    list_algorithms,
    list_problems,
)

__all__ = [
    "BaseAlgorithm",
    "BaseProblem",
    "Experiment",
    "ExperimentConfig",
    "Result",
    "ExperimentSaver",
    "get_algorithm",
    "get_problem",
    "list_algorithms",
    "list_problems",
    "generate_html_report",
]
