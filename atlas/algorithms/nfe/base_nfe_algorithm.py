"""Backward compatibility module for NFE-based algorithm variants.

In ATLAS >= 0.2, all algorithms natively support both iteration-based
and NFE-based stopping criteria via :class:`atlas.core.BaseAlgorithm`.
"""

from __future__ import annotations

from atlas.core.base_algorithm import BaseAlgorithm

# BaseNFEAlgorithm is now an alias to BaseAlgorithm
BaseNFEAlgorithm = BaseAlgorithm

