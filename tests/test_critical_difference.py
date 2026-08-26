"""Unit tests for Critical Difference diagram and Nemenyi post-hoc analysis."""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import atlas.stats
from atlas.stats.critical_difference import (
    _Q_ALPHA_TABLE,
    calculate_critical_difference,
    nemenyi_post_hoc,
    plot_critical_difference,
)


class TestCriticalDifferenceCalculation:
    """Tests for calculate_critical_difference and q_alpha lookup."""

    def test_cd_value_k3_N30_alpha05(self):
        """k=3, N=30, alpha=0.05 should yield CD ≈ 0.605 (error < 0.01)."""
        cd = calculate_critical_difference(k=3, N=30, alpha=0.05)
        # q = 2.343, CD = 2.343 * sqrt(3*4 / (6*30)) = 2.343 * sqrt(12/180) ≈ 0.605
        assert cd == pytest.approx(0.605, abs=0.01)

    def test_cd_value_k4_N10_alpha05(self):
        """k=4, N=10, alpha=0.05 should use q=2.569 from table."""
        cd = calculate_critical_difference(k=4, N=10, alpha=0.05)
        expected = 2.569 * np.sqrt(4 * 5 / (6 * 10))
        assert cd == pytest.approx(expected, rel=1e-5)

    def test_cd_value_alpha10_table(self):
        """alpha=0.10 lookup from table."""
        cd = calculate_critical_difference(k=5, N=20, alpha=0.10)
        expected = 2.460 * np.sqrt(5 * 6 / (6 * 20))
        assert cd == pytest.approx(expected, rel=1e-5)

    def test_cd_fallback_scipy_for_large_k(self):
        """Fallback to scipy studentized_range when k not in table (e.g. k=15)."""
        cd = calculate_critical_difference(k=15, N=30, alpha=0.05)
        assert isinstance(cd, float)
        assert cd > 0

    def test_cd_fallback_scipy_for_custom_alpha(self):
        """Fallback to scipy studentized_range when alpha not in table (e.g. alpha=0.01)."""
        cd = calculate_critical_difference(k=4, N=20, alpha=0.01)
        assert isinstance(cd, float)
        assert cd > 0

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match=r"k must be >= 2"):
            calculate_critical_difference(k=1, N=30)

    def test_invalid_N_raises(self):
        with pytest.raises(ValueError, match=r"N must be >= 1"):
            calculate_critical_difference(k=3, N=0)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\)"):
            calculate_critical_difference(k=3, N=30, alpha=0.0)
        with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\)"):
            calculate_critical_difference(k=3, N=30, alpha=1.0)
        with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\)"):
            calculate_critical_difference(k=3, N=30, alpha=-0.05)


class TestNemenyiPostHoc:
    """Tests for nemenyi_post_hoc pairwise significance test."""

    def test_nemenyi_significantly_different(self):
        """Algorithms with |rank_i - rank_j| > CD must appear in significant pairs."""
        ranks = {"AlgoA": 1.0, "AlgoB": 1.5, "AlgoC": 3.0}
        cd = 1.0
        sig_pairs = nemenyi_post_hoc(ranks, cd)

        # |AlgoA - AlgoC| = 2.0 > 1.0 -> significant
        # |AlgoB - AlgoC| = 1.5 > 1.0 -> significant
        # |AlgoA - AlgoB| = 0.5 <= 1.0 -> not significant
        assert ("AlgoA", "AlgoC") in sig_pairs or ("AlgoC", "AlgoA") in sig_pairs
        assert ("AlgoB", "AlgoC") in sig_pairs or ("AlgoC", "AlgoB") in sig_pairs
        assert len(sig_pairs) == 2

    def test_nemenyi_not_significant(self):
        """Algorithms with |rank_i - rank_j| <= CD must NOT appear in significant pairs."""
        ranks = {"AlgoA": 1.0, "AlgoB": 1.5, "AlgoC": 1.8}
        cd = 1.0
        sig_pairs = nemenyi_post_hoc(ranks, cd)

        # All differences <= 1.0, so no significant pairs
        assert len(sig_pairs) == 0
        assert ("AlgoA", "AlgoB") not in sig_pairs
        assert ("AlgoA", "AlgoC") not in sig_pairs
        assert ("AlgoB", "AlgoC") not in sig_pairs

    def test_negative_cd_raises(self):
        with pytest.raises(ValueError, match=r"cd must be non-negative"):
            nemenyi_post_hoc({"AlgoA": 1.0, "AlgoB": 2.0}, cd=-0.5)


class TestPlotCriticalDifference:
    """Tests for plot_critical_difference diagram generation."""

    def test_plot_critical_difference_returns_figure(self):
        """4 algorithms should generate and return a valid matplotlib Figure."""
        ranks = {
            "DE": 1.25,
            "PSO": 1.85,
            "GA": 2.90,
            "ACO": 4.00,
        }
        fig = plot_critical_difference(ranks, n_datasets=30, alpha=0.05, title="Benchmark CD Diagram")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_saves_to_png_svg_pdf(self):
        """Verify export to PNG, SVG, and PDF formats via save_path."""
        ranks = {
            "DE": 1.2,
            "PSO": 1.8,
            "GA": 2.7,
            "ACO": 3.5,
            "SA": 4.8,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            for ext in [".png", ".svg", ".pdf"]:
                out_path = Path(tmpdir) / f"cd_plot{ext}"
                fig = plot_critical_difference(
                    ranks,
                    n_datasets=25,
                    alpha=0.05,
                    save_path=str(out_path),
                )
                plt.close(fig)
                assert out_path.exists()
                assert out_path.stat().st_size > 0

    def test_plot_insufficient_algorithms_raises(self):
        with pytest.raises(ValueError, match=r"At least 2 algorithms"):
            plot_critical_difference({"DE": 1.0}, n_datasets=10)

    def test_plot_invalid_n_datasets_raises(self):
        with pytest.raises(ValueError, match=r"n_datasets must be >= 1"):
            plot_critical_difference({"DE": 1.0, "PSO": 2.0}, n_datasets=0)


class TestModuleExports:
    """Tests for atlas.stats module exports."""

    def test_stats_package_exports(self):
        """Verify critical difference symbols are exported at package level."""
        assert hasattr(atlas.stats, "calculate_critical_difference")
        assert hasattr(atlas.stats, "nemenyi_post_hoc")
        assert hasattr(atlas.stats, "plot_critical_difference")
        assert "calculate_critical_difference" in atlas.stats.__all__
        assert "nemenyi_post_hoc" in atlas.stats.__all__
        assert "plot_critical_difference" in atlas.stats.__all__
