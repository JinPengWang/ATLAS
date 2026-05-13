"""Algorithms sub-package.

Importing this package automatically registers all built-in algorithms
via the :mod:`atlas.utils.registry` mechanism.  To add a new algorithm,
create a module in this directory and add a corresponding import line here.
"""

from atlas.algorithms.aco import ACO
from atlas.algorithms.de import DE
from atlas.algorithms.ga import GA
from atlas.algorithms.pso import PSO
from atlas.algorithms.sa import SA
from atlas.algorithms.woa import WOA

__all__ = ["ACO", "DE", "GA", "PSO", "SA", "WOA"]
