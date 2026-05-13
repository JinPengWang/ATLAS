"""Algorithms sub-package.

Importing this package automatically registers all built-in algorithms
via the :mod:`atlas.utils.registry` mechanism.

Two variants exist for each algorithm:

* **Iteration-based** (``atlas.algorithms.iteration``) – stops after
  ``max_iter`` iterations.
* **NFE-based** (``atlas.algorithms.nfe``) – stops after
  ``max_nfe`` function evaluations.
"""

from atlas.algorithms.iteration import ACO, DE, GA, PSO, SA, WOA
from atlas.algorithms.nfe import (
    ACO_NFE,
    DE_NFE,
    GA_NFE,
    PSO_NFE,
    SA_NFE,
    WOA_NFE,
)

__all__ = [
    # Iteration-based
    "ACO",
    "DE",
    "GA",
    "PSO",
    "SA",
    "WOA",
    # NFE-based
    "ACO_NFE",
    "DE_NFE",
    "GA_NFE",
    "PSO_NFE",
    "SA_NFE",
    "WOA_NFE",
]
