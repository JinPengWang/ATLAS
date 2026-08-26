"""Unit tests for engineering optimization problems."""

import numpy as np
import pytest

from atlas.problems.engineering.pressure_vessel_eng import PressureVesselEng
from atlas.problems.engineering.speed_reducer import SpeedReducer
from atlas.problems.engineering.spring_design import SpringDesign
from atlas.problems.engineering.truss3bar import Truss3Bar
from atlas.problems.engineering.welded_beam import WeldedBeam
from atlas.utils.registry import get_problem


@pytest.mark.parametrize(
    "name",
    [
        "welded_beam",
        "spring_design",
        "speed_reducer",
        "truss_3bar",
        "pressure_vessel_eng",
    ],
)
def test_problem_registered(name):
    assert get_problem(name) is not None


@pytest.mark.parametrize(
    "name",
    [
        "welded_beam",
        "spring_design",
        "speed_reducer",
        "truss_3bar",
        "pressure_vessel_eng",
    ],
)
def test_evaluate_finite(name):
    prob = get_problem(name)()
    lb, ub = prob.get_bounds()
    x = (lb + ub) / 2.0
    val = prob.evaluate_with_penalty(x)
    assert np.isfinite(val)


@pytest.mark.parametrize(
    "name",
    [
        "welded_beam",
        "spring_design",
        "speed_reducer",
        "truss_3bar",
        "pressure_vessel_eng",
    ],
)
def test_bounds_shape_matches_dim(name):
    prob = get_problem(name)()
    lb, ub = prob.get_bounds()
    assert len(lb) == prob.get_dim()
    assert len(ub) == prob.get_dim()
    assert np.all(lb < ub)


@pytest.mark.parametrize(
    "name,expected_dim",
    [
        ("welded_beam", 4),
        ("spring_design", 3),
        ("speed_reducer", 7),
        ("truss_3bar", 2),
        ("pressure_vessel_eng", 4),
    ],
)
def test_correct_dimension(name, expected_dim):
    prob = get_problem(name)()
    assert prob.get_dim() == expected_dim


def test_welded_beam_has_constraints():
    prob = WeldedBeam()
    lb, ub = prob.get_bounds()
    x = (lb + ub) / 2.0
    g = prob.get_constraints(x)
    assert len(g) == 7


def test_spring_design_known_optimum():
    prob = SpringDesign()
    opt_loc = prob.get_optimum_location()
    assert opt_loc is not None
    val = prob.evaluate(opt_loc)
    assert val < 0.013


def test_speed_reducer_known_optimum():
    prob = SpeedReducer()
    opt_loc = prob.get_optimum_location()
    assert opt_loc is not None
    val = prob.evaluate(opt_loc)
    assert val < 3050.0


def test_truss3bar_known_optimum():
    prob = Truss3Bar()
    opt_loc = prob.get_optimum_location()
    assert opt_loc is not None
    val = prob.evaluate(opt_loc)
    assert val < 270.0


def test_pressure_vessel_eng_known_optimum():
    prob = PressureVesselEng()
    opt_loc = prob.get_optimum_location()
    assert opt_loc is not None
    val = prob.evaluate(opt_loc)
    assert val < 6100.0
