"""Unit tests for the Experiment scheduler and manager."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from atlas.core.experiment import Experiment, ExperimentConfig
from atlas.core.result import Result


class TestExperimentConfig:
    def test_from_yaml_content(self):
        yaml_content = """
        algorithms:
          - pso
          - de
        problems:
          - sphere
        dims:
          - 10
          - 30
        max_iter: 100
        runs: 5
        n_jobs: 2
        """
        cfg = ExperimentConfig.from_yaml(yaml_content)
        assert cfg.algorithms == ["pso", "de"]
        assert cfg.problems == ["sphere"]
        assert cfg.dims == [10, 30]
        assert cfg.max_iter == 100
        assert cfg.runs == 5
        assert cfg.n_jobs == 2

    def test_from_yaml_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("algorithms: ['ga']\nproblems: ['ackley']\nmax_iter: 50\n")
            f_path = f.name

        try:
            cfg = ExperimentConfig.from_yaml(f_path)
            assert cfg.algorithms == ["ga"]
            assert cfg.problems == ["ackley"]
            assert cfg.max_iter == 50
        finally:
            Path(f_path).unlink(missing_ok=True)


class TestExperimentExecution:
    def test_cartesian_product_multidim(self):
        """Test multi-dimensional Cartesian product execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ExperimentConfig(
                algorithms=["pso"],
                problems=["sphere"],
                dims=[5, 10],
                max_iter=10,
                runs=2,
                save_results=False,
                generate_stats=False,
                base_dir=tmpdir,
            )
            exp = Experiment(cfg)
            results = exp.run_all()

            assert "sphere_D5" in results
            assert "sphere_D10" in results
            assert len(results["sphere_D5"]["pso"]) == 2
            assert len(results["sphere_D10"]["pso"]) == 2

    def test_multiprocessing_parallel(self):
        """Test parallel execution with n_jobs=2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ExperimentConfig(
                algorithms=["pso", "de"],
                problems=["sphere"],
                dims=[5],
                max_iter=20,
                runs=4,
                n_jobs=2,
                save_results=True,
                generate_stats=True,
                base_dir=tmpdir,
            )
            exp = Experiment(cfg)
            results = exp.run_all()

            assert "sphere" in results
            assert len(results["sphere"]["pso"]) == 4
            assert len(results["sphere"]["de"]) == 4

            # Check that statistical reports were generated
            stat_dir = list(Path(tmpdir).glob("experiment_*/statistical_reports"))[0]
            assert stat_dir.exists()
            assert (stat_dir / "summary_statistics.csv").exists()
            assert (stat_dir / "benchmark_table.tex").exists()
            assert (stat_dir / "summary_statistics.md").exists()
