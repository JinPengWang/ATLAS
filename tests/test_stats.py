"""Unit tests for statistical hypothesis testing and table export."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atlas.stats.hypothesis_testing import (
    FriedmanTestResult,
    PairwiseTestResult,
    calculate_average_ranks,
    friedman_test,
    wilcoxon_signed_rank_test,
)
from atlas.stats.table_exporter import (
    export_csv_summary,
    export_latex_table,
    export_markdown_table,
)


class TestWilcoxonSignedRankTest:
    def test_sample_a_better_than_b(self):
        sample_a = [0.01, 0.02, 0.015, 0.03, 0.012, 0.018, 0.022, 0.014, 0.016, 0.019]
        sample_b = [1.5, 2.3, 1.8, 2.1, 1.9, 2.4, 2.0, 1.7, 2.2, 2.5]
        res = wilcoxon_signed_rank_test(sample_a, sample_b, alpha=0.05)

        assert isinstance(res, PairwiseTestResult)
        assert res.p_value < 0.05
        assert res.significance == "+"

    def test_sample_a_worse_than_b(self):
        sample_a = [10.0, 12.0, 11.5, 13.0, 10.5, 11.8, 12.2, 10.9, 11.2, 12.5]
        sample_b = [0.1, 0.2, 0.15, 0.3, 0.12, 0.18, 0.22, 0.14, 0.16, 0.19]
        res = wilcoxon_signed_rank_test(sample_a, sample_b, alpha=0.05)

        assert isinstance(res, PairwiseTestResult)
        assert res.p_value < 0.05
        assert res.significance == "-"

    def test_identical_samples(self):
        sample = [1.0, 2.0, 3.0, 4.0, 5.0]
        res = wilcoxon_signed_rank_test(sample, sample, alpha=0.05)

        assert res.p_value == 1.0
        assert res.significance == "="

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="Sample lengths must match"):
            wilcoxon_signed_rank_test([1.0, 2.0], [1.0, 2.0, 3.0])


class TestFriedmanTest:
    def test_friedman_ranking(self):
        # 4 problems, 3 algorithms
        # Algo_0 always best (1), Algo_1 middle (2), Algo_2 worst (3)
        perf_matrix = np.array([
            [1.0, 10.0, 100.0],
            [0.5, 5.0, 50.0],
            [2.0, 20.0, 200.0],
            [1.5, 15.0, 150.0],
        ])
        algo_names = ["BestAlgo", "MidAlgo", "WorstAlgo"]

        ranks = calculate_average_ranks(perf_matrix, algorithm_names=algo_names, higher_is_better=False)
        assert ranks["BestAlgo"] == pytest.approx(1.0)
        assert ranks["MidAlgo"] == pytest.approx(2.0)
        assert ranks["WorstAlgo"] == pytest.approx(3.0)

        res = friedman_test(perf_matrix, algorithm_names=algo_names, alpha=0.05)
        assert isinstance(res, FriedmanTestResult)
        assert res.is_significant is True
        assert res.rankings_order[0][0] == "BestAlgo"
        assert res.rankings_order[-1][0] == "WorstAlgo"


class TestTableExporter:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "problem": ["sphere", "sphere", "rastrigin", "rastrigin"],
            "algorithm": ["pso", "de", "pso", "de"],
            "mean": [1.2e-4, 5.4e-6, 12.5, 0.45],
            "std": [3.1e-5, 1.2e-6, 2.1, 0.08],
        })

    def test_latex_table_export(self, sample_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "table.tex"
            latex_str = export_latex_table(sample_df, output_path=out_file, highlight_best=True)

            assert "\\begin{table*}" in latex_str
            assert "\\textbf{" in latex_str
            assert out_file.exists()
            assert len(out_file.read_text(encoding="utf-8")) > 0

    def test_markdown_table_export(self, sample_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "table.md"
            md_str = export_markdown_table(sample_df, output_path=out_file)

            assert "| problem" in md_str
            assert out_file.exists()

    def test_csv_summary_export(self, sample_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "summary.csv"
            export_csv_summary(sample_df, out_file)

            assert out_file.exists()
            loaded = pd.read_csv(out_file)
            assert len(loaded) == 4
