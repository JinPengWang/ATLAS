"""Unit tests for the interactive HTML experiment report generator."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from atlas.algorithms.iteration.de import DE
from atlas.algorithms.iteration.pso import PSO
from atlas.core.result import Result
from atlas.problems.benchmark.multimodal import Rastrigin
from atlas.problems.benchmark.unimodal import Sphere
from atlas.report import generate_html_report
from atlas.report.html_report import (
    _build_boxplot_figure,
    _build_convergence_figure,
    _build_heatmap_figure,
    _build_summary_table_html,
    _format_value,
)


@pytest.fixture
def sample_sphere_results() -> Dict[str, Dict[str, List[Result]]]:
    """Generate small fast sample results on Sphere for testing."""
    problem = Sphere(dim=5)
    pso_runs = [
        PSO(problem=problem, max_iter=10, pop_size=5, seed=42 + i).run(run_id=i)
        for i in range(2)
    ]
    de_runs = [
        DE(problem=problem, max_iter=10, pop_size=5, seed=42 + i).run(run_id=i)
        for i in range(2)
    ]
    return {
        "sphere": {
            "pso": pso_runs,
            "de": de_runs,
        }
    }


def test_html_report_creates_file(tmp_path: Path, sample_sphere_results: Dict[str, Dict[str, List[Result]]]) -> None:
    """Test that report file is created and contains basic HTML tags and problem name."""
    out_file = tmp_path / "report.html"
    returned_path = generate_html_report(sample_sphere_results, output_path=out_file)

    assert returned_path.exists()
    assert returned_path == out_file
    content = out_file.read_text(encoding="utf-8")
    assert "<html" in content.lower()
    assert "sphere" in content.lower()


def test_html_report_contains_algorithms(tmp_path: Path, sample_sphere_results: Dict[str, Dict[str, List[Result]]]) -> None:
    """Test that custom title and all algorithm names are present in report."""
    out_file = tmp_path / "custom_report.html"
    custom_title = "My Custom Benchmark Title"
    generate_html_report(sample_sphere_results, output_path=out_file, title=custom_title)

    content = out_file.read_text(encoding="utf-8")
    assert custom_title in content
    assert "pso" in content.lower()
    assert "de" in content.lower()


def test_html_report_creates_parent_dirs(tmp_path: Path, sample_sphere_results: Dict[str, Dict[str, List[Result]]]) -> None:
    """Test that deeply nested parent directories are automatically created."""
    deep_path = tmp_path / "deeply" / "nested" / "output" / "dir" / "report.html"
    returned_path = generate_html_report(sample_sphere_results, output_path=deep_path)

    assert returned_path.exists()
    assert returned_path.is_file()
    content = deep_path.read_text(encoding="utf-8")
    assert "<html" in content.lower()


def test_html_report_single_algorithm(tmp_path: Path) -> None:
    """Test that generating a report with a single algorithm does not crash."""
    problem = Sphere(dim=5)
    pso_runs = [
        PSO(problem=problem, max_iter=10, pop_size=5, seed=42 + i).run(run_id=i)
        for i in range(2)
    ]
    single_results = {"sphere": {"pso": pso_runs}}

    out_file = tmp_path / "single_algo_report.html"
    returned_path = generate_html_report(single_results, output_path=out_file)

    assert returned_path.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "<html" in content.lower()
    assert "pso" in content.lower()


def test_html_report_multi_problem_and_heatmap(tmp_path: Path) -> None:
    """Test multi-problem and multi-algorithm execution including heatmap."""
    sphere = Sphere(dim=5)
    rastrigin = Rastrigin(dim=5)

    all_results = {
        "sphere": {
            "pso": [PSO(problem=sphere, max_iter=10, pop_size=5, seed=1).run(run_id=0)],
            "de": [DE(problem=sphere, max_iter=10, pop_size=5, seed=1).run(run_id=0)],
        },
        "rastrigin": {
            "pso": [PSO(problem=rastrigin, max_iter=10, pop_size=5, seed=1).run(run_id=0)],
            "de": [DE(problem=rastrigin, max_iter=10, pop_size=5, seed=1).run(run_id=0)],
        },
    }

    out_file = tmp_path / "multi_report.html"
    generate_html_report(all_results, output_path=out_file)

    content = out_file.read_text(encoding="utf-8")
    assert "sphere" in content.lower()
    assert "rastrigin" in content.lower()
    assert "heatmap" in content.lower()


def test_html_report_empty_results(tmp_path: Path) -> None:
    """Test that empty results do not raise an error and produce valid HTML."""
    out_file = tmp_path / "empty_report.html"
    generate_html_report({}, output_path=out_file)

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "<html" in content.lower()


def test_html_report_missing_convergence(tmp_path: Path) -> None:
    """Test handling of results without convergence curves."""
    res = Result(
        algorithm_name="mock_algo",
        problem_name="mock_prob",
        run_id=0,
        seed=42,
        best_fitness=1.23,
        best_solution=np.zeros(2),
        convergence_curve=[],
    )
    all_results = {"mock_prob": {"mock_algo": [res]}}
    out_file = tmp_path / "mock_report.html"
    generate_html_report(all_results, output_path=out_file)

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "mock_prob" in content
    assert "mock_algo" in content


def test_helper_functions() -> None:
    """Test helper functions formatting and figure builders."""
    assert _format_value(0.0) == "0.0000"
    assert _format_value(float("nan")) == "N/A"
    assert "e" in _format_value(1e-6)
    assert _format_value(3.14159) == "3.1416"

    # Test summary table with empty data
    empty_html = _build_summary_table_html({})
    assert "No experiment results" in empty_html

    # Test heatmap builder with 1 algorithm returns None
    res = Result(
        algorithm_name="pso",
        problem_name="sphere",
        run_id=0,
        seed=1,
        best_fitness=0.1,
        best_solution=np.zeros(2),
    )
    assert _build_heatmap_figure({"sphere": {"pso": [res]}}) is None
