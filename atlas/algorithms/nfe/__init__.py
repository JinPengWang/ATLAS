"""NFE-based algorithm variants.

These algorithms stop after a maximum number of function evaluations
(``max_nfe``) rather than a fixed iteration count.
"""

from atlas.algorithms.nfe.aco import ACO_NFE
from atlas.algorithms.nfe.de import DE_NFE
from atlas.algorithms.nfe.ga import GA_NFE
from atlas.algorithms.nfe.pso import PSO_NFE
from atlas.algorithms.nfe.sa import SA_NFE
from atlas.algorithms.nfe.woa import WOA_NFE

__all__ = ["ACO_NFE", "DE_NFE", "GA_NFE", "PSO_NFE", "SA_NFE", "WOA_NFE"]
