"""Custom problems sub-package.

This package contains user-defined optimisation problems.  Add your own
problem modules here and import them in this file to register them.
"""

from atlas.problems.custom.example_custom import PressureVessel

__all__ = ["PressureVessel"]
