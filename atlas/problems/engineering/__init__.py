"""Engineering optimization benchmark problems."""

from __future__ import annotations

from atlas.problems.engineering.pressure_vessel_eng import PressureVesselEng
from atlas.problems.engineering.speed_reducer import SpeedReducer
from atlas.problems.engineering.spring_design import SpringDesign
from atlas.problems.engineering.truss3bar import Truss3Bar
from atlas.problems.engineering.welded_beam import WeldedBeam

__all__ = [
    "WeldedBeam",
    "SpringDesign",
    "SpeedReducer",
    "Truss3Bar",
    "PressureVesselEng",
]
