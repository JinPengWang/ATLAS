"""Algorithms sub-package.

Importing this package automatically registers all built-in algorithms
via the :mod:`atlas.utils.registry` mechanism.

Two variants exist for each algorithm:

* **Iteration-based** (``atlas.algorithms.iteration``) – stops after
  ``max_iter`` iterations.
* **NFE-based** (``atlas.algorithms.nfe``) – stops after
  ``max_nfe`` function evaluations.
"""

from atlas.algorithms.iteration import (
    ABC,
    ACO,
    CMAES,
    DE,
    DP,
    GA,
    GWO,
    HHO,
    LEA,
    LGC,
    LSHADE,
    PPO,
    PSO,
    PSA,
    SA,
    SMA,
    TJO,
    WOA,
)
from atlas.algorithms.nfe import (
    ABCNfe,
    ACO_NFE,
    CMAESNfe,
    DE_NFE,
    DP_NFE,
    GA_NFE,
    GWONfe,
    HHONfe,
    LEA_NFE,
    LGC_NFE,
    LShadeNFE,
    PPO_NFE,
    PSO_NFE,
    PSA_NFE,
    SA_NFE,
    SMANfe,
    TJO_NFE,
    WOA_NFE,
)

__all__ = [
    # Iteration-based
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
    # NFE-based
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
