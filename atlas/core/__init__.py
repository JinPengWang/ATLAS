"""Core sub-package: base classes, experiment runner, and result structures."""

from atlas.core.base_algorithm import BaseAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.core.experiment import Experiment, ExperimentConfig
from atlas.core.result import Result

__all__ = [
    "BaseAlgorithm",
    "BaseProblem",
    "Experiment",
    "ExperimentConfig",
    "Result",
]
