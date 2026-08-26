"""NFE-based algorithm variants.

These algorithms stop after a maximum number of function evaluations
(``max_nfe``) rather than a fixed iteration count.
"""

from atlas.algorithms.nfe.abc_nfe import ABCNfe
from atlas.algorithms.nfe.aco import ACO_NFE
from atlas.algorithms.nfe.cmaes_nfe import CMAESNfe
from atlas.algorithms.nfe.de import DE_NFE
from atlas.algorithms.nfe.dp import DP_NFE
from atlas.algorithms.nfe.ga import GA_NFE
from atlas.algorithms.nfe.gwo_nfe import GWONfe
from atlas.algorithms.nfe.hho_nfe import HHONfe
from atlas.algorithms.nfe.lea import LEA_NFE
from atlas.algorithms.nfe.lgc import LGC_NFE
from atlas.algorithms.nfe.lshade_nfe import LShadeNFE
from atlas.algorithms.nfe.ppo import PPO_NFE
from atlas.algorithms.nfe.pso import PSO_NFE
from atlas.algorithms.nfe.psa import PSA_NFE
from atlas.algorithms.nfe.sa import SA_NFE
from atlas.algorithms.nfe.sma_nfe import SMANfe
from atlas.algorithms.nfe.tjo import TJO_NFE
from atlas.algorithms.nfe.woa import WOA_NFE

__all__ = [
    "ABCNfe",
    "ACO_NFE",
    "CMAESNfe",
    "DE_NFE",
    "DP_NFE",
    "GA_NFE",
    "GWONfe",
    "HHONfe",
    "LEA_NFE",
    "LGC_NFE",
    "LShadeNFE",
    "PPO_NFE",
    "PSO_NFE",
    "PSA_NFE",
    "SA_NFE",
    "SMANfe",
    "TJO_NFE",
    "WOA_NFE",
]
