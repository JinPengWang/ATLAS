"""Pure-Python CEC benchmark suites (zero C compiler dependency)."""

from __future__ import annotations

from atlas.problems.cec_pure.cec2017_pure import (
    CEC2017F1,
    CEC2017F3,
    CEC2017F4,
    CEC2017F5,
    CEC2017F6,
    CEC2017F7,
    CEC2017F8,
    CEC2017F9,
    CEC2017F10,
    PureCEC2017Problem,
)

__all__ = [
    "PureCEC2017Problem",
    "CEC2017F1",
    "CEC2017F3",
    "CEC2017F4",
    "CEC2017F5",
    "CEC2017F6",
    "CEC2017F7",
    "CEC2017F8",
    "CEC2017F9",
    "CEC2017F10",
]
