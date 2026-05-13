"""Iteration-based algorithm variants.

These algorithms stop after a fixed number of iterations (``max_iter``).
"""

from atlas.algorithms.iteration.aco import ACO
from atlas.algorithms.iteration.de import DE
from atlas.algorithms.iteration.ga import GA
from atlas.algorithms.iteration.pso import PSO
from atlas.algorithms.iteration.sa import SA
from atlas.algorithms.iteration.woa import WOA

__all__ = ["ACO", "DE", "GA", "PSO", "SA", "WOA"]
