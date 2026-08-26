"""Iteration-based algorithm variants.

These algorithms stop after a fixed number of iterations (``max_iter``).
"""

from atlas.algorithms.iteration.abc import ABC
from atlas.algorithms.iteration.aco import ACO
from atlas.algorithms.iteration.cmaes import CMAES
from atlas.algorithms.iteration.de import DE
from atlas.algorithms.iteration.dp import DP
from atlas.algorithms.iteration.ga import GA
from atlas.algorithms.iteration.gwo import GWO
from atlas.algorithms.iteration.hho import HHO
from atlas.algorithms.iteration.lea import LEA
from atlas.algorithms.iteration.lgc import LGC
from atlas.algorithms.iteration.lshade import LSHADE
from atlas.algorithms.iteration.ppo import PPO
from atlas.algorithms.iteration.pso import PSO
from atlas.algorithms.iteration.psa import PSA
from atlas.algorithms.iteration.sa import SA
from atlas.algorithms.iteration.sma import SMA
from atlas.algorithms.iteration.tjo import TJO
from atlas.algorithms.iteration.woa import WOA

__all__ = [
    "ABC",
    "ACO",
    "CMAES",
    "DE",
    "DP",
    "GA",
    "GWO",
    "HHO",
    "LEA",
    "LGC",
    "LSHADE",
    "PPO",
    "PSO",
    "PSA",
    "SA",
    "SMA",
    "TJO",
    "WOA",
]
