"""Critical Difference (CD) diagram generation and Nemenyi post-hoc analysis.

This module provides tools for computing the critical difference (CD) using
the Nemenyi test and plotting Critical Difference diagrams to visualize
statistically significant differences among multiple optimization algorithms.

References:
    Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets.
    *Journal of Machine Learning Research*, 7, 1-30.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import studentized_range

# Nemenyi q_alpha critical values lookup table
_Q_ALPHA_TABLE: Dict[float, Dict[int, float]] = {
    0.05: {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
    },
    0.10: {
        2: 1.645,
        3: 2.052,
        4: 2.291,
        5: 2.460,
        6: 2.589,
        7: 2.693,
        8: 2.780,
        9: 2.855,
        10: 2.920,
    },
}


def calculate_critical_difference(k: int, N: int, alpha: float = 0.05) -> float:
    """Compute Nemenyi critical difference.

    The critical difference is calculated as:
        CD = q_alpha * sqrt(k * (k + 1) / (6 * N))

    If (alpha, k) is found in `_Q_ALPHA_TABLE`, the tabulated critical value is
    used. Otherwise, it falls back to scipy's `studentized_range` distribution:
        q_alpha = studentized_range.ppf(1 - alpha, k, df=inf) / sqrt(2)

    Args:
        k: Number of algorithms being compared (must be >= 2).
        N: Number of datasets/benchmark problems (must be >= 1).
        alpha: Significance level (default: 0.05, must be in (0, 1)).

    Returns:
        Critical difference (CD) value as a float.

    Raises:
        ValueError: If k < 2, N < 1, or alpha is not in the interval (0, 1).
    """
    if k < 2:
        raise ValueError(f"Number of algorithms k must be >= 2, got {k}.")
    if N < 1:
        raise ValueError(f"Number of datasets/problems N must be >= 1, got {N}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(
            f"Significance level alpha must be in (0, 1), got {alpha}."
        )

    q_alpha: Optional[float] = None
    for tbl_alpha, k_dict in _Q_ALPHA_TABLE.items():
        if np.isclose(alpha, tbl_alpha, atol=1e-5):
            if k in k_dict:
                q_alpha = k_dict[k]
            break

    if q_alpha is None:
        # Fall back to scipy studentized_range
        # For Nemenyi test: q_alpha = q_tukey(1 - alpha, k, df=inf) / sqrt(2)
        q_alpha = float(studentized_range.ppf(1.0 - alpha, k, np.inf) / np.sqrt(2.0))

    cd = float(q_alpha * np.sqrt((k * (k + 1)) / (6.0 * N)))
    return cd


def nemenyi_post_hoc(ranks: Dict[str, float], cd: float) -> List[Tuple[str, str]]:
    """Identify pairs of algorithms with statistically significant rank differences.

    Two algorithms i and j have a statistically significant difference if
    |rank_i - rank_j| > cd.

    Args:
        ranks: Dictionary mapping algorithm names to their average ranks.
        cd: Critical difference threshold (must be non-negative).

    Returns:
        List of tuples (algo_a, algo_b) representing pairs of algorithms where
        the absolute difference between their average ranks exceeds cd.

    Raises:
        ValueError: If cd is negative.
    """
    if cd < 0:
        raise ValueError(f"Critical difference cd must be non-negative, got {cd}.")

    significant_pairs: List[Tuple[str, str]] = []
    algo_names = list(ranks.keys())
    n = len(algo_names)
    for i in range(n):
        for j in range(i + 1, n):
            a = algo_names[i]
            b = algo_names[j]
            if abs(ranks[a] - ranks[b]) > cd:
                significant_pairs.append((a, b))
    return significant_pairs


def _find_non_significant_cliques(
    sorted_ranks: List[float], cd: float
) -> List[Tuple[int, int]]:
    """Find all maximal contiguous cliques where (max_rank - min_rank) <= cd.

    Args:
        sorted_ranks: List of ranks sorted in non-decreasing order.
        cd: Critical difference threshold.

    Returns:
        List of (start_idx, end_idx) index pairs for maximal groups of size >= 2.
    """
    n = len(sorted_ranks)
    candidates: List[Tuple[int, int]] = []
    for i in range(n):
        j = i
        while j + 1 < n and (sorted_ranks[j + 1] - sorted_ranks[i]) <= cd:
            j += 1
        if j > i:
            candidates.append((i, j))

    # Filter out subsets
    maximal: List[Tuple[int, int]] = []
    for i, j in candidates:
        if not any((oi, oj) != (i, j) and oi <= i and oj >= j for oi, oj in candidates):
            maximal.append((i, j))
    return maximal


def plot_critical_difference(
    ranks: Dict[str, float],
    n_datasets: int,
    alpha: float = 0.05,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 4),
    dpi: int = 300,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Draw a Critical Difference diagram showing average ranks and connecting bars.

    The diagram illustrates:
    - A horizontal axis from 1 (best rank on left) to k (worst rank on right).
    - Algorithms plotted at their average rank positions with alternating top/bottom
      stems and labels to prevent visual overlap.
    - Bold horizontal bars connecting groups of algorithms between which differences
      are not statistically significant (|rank_i - rank_j| <= CD).
    - A CD scale bar in the top-left corner.

    Args:
        ranks: Dictionary mapping algorithm names to their average ranks.
        n_datasets: Number of datasets/problems used in evaluation (must be >= 1).
        alpha: Significance level for the Nemenyi test (default: 0.05).
        title: Optional title for the diagram.
        figsize: Figure dimensions (width, height) in inches (default: (10, 4)).
        dpi: Resolution for the figure (default: 300).
        save_path: Optional file path to save the figure (supports .png, .pdf, .svg).

    Returns:
        Matplotlib Figure object containing the Critical Difference diagram.

    Raises:
        ValueError: If fewer than 2 algorithms are provided, or n_datasets < 1.
    """
    if len(ranks) < 2:
        raise ValueError(
            f"At least 2 algorithms are required to plot CD diagram, got {len(ranks)}."
        )
    if n_datasets < 1:
        raise ValueError(f"Number of datasets n_datasets must be >= 1, got {n_datasets}.")

    k = len(ranks)
    cd = calculate_critical_difference(k=k, N=n_datasets, alpha=alpha)

    # Sort algorithms by average rank ascending (1 = best)
    sorted_items = sorted(ranks.items(), key=lambda item: item[1])
    algo_names = [item[0] for item in sorted_items]
    algo_ranks = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Base coordinates
    axis_y = 0.0
    tick_height = 0.06
    stem_length = 0.40

    # Draw main horizontal axis line
    ax.plot([1.0, float(k)], [axis_y, axis_y], color="black", linewidth=1.5, zorder=2)

    # Draw ticks and numbers 1..k on the axis
    for tick in range(1, k + 1):
        ax.plot(
            [tick, tick],
            [axis_y - tick_height / 2.0, axis_y + tick_height / 2.0],
            color="black",
            linewidth=1.2,
            zorder=2,
        )
        ax.text(
            tick,
            axis_y - 0.10,
            str(tick),
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="#333333",
        )

    # Find non-significant groups (cliques)
    cliques = _find_non_significant_cliques(algo_ranks, cd)

    # Assign vertical bar levels for non-significant groups to prevent overlap
    clique_spans = [(algo_ranks[i], algo_ranks[j]) for i, j in cliques]
    clique_spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))

    bar_levels: List[Tuple[Tuple[float, float], int]] = []
    for c_start, c_end in clique_spans:
        lvl = 0
        while True:
            collision = any(
                lvl == ex_lvl and not (c_end < ex_start - 0.02 or c_start > ex_end + 0.02)
                for (ex_start, ex_end), ex_lvl in bar_levels
            )
            if not collision:
                break
            lvl += 1
        bar_levels.append(((c_start, c_end), lvl))

    # Draw non-significant group connecting bars
    bar_base_y = 0.12
    bar_step_y = 0.08
    for (span_start, span_end), lvl in bar_levels:
        bar_y = bar_base_y + lvl * bar_step_y
        ax.plot(
            [span_start, span_end],
            [bar_y, bar_y],
            color="#2c3e50",
            linewidth=3.5,
            solid_capstyle="round",
            zorder=4,
        )

    # Colors for markers
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for idx, (name, rank) in enumerate(zip(algo_names, algo_ranks)):
        color = palette[idx % len(palette)]
        # Marker on the rank axis
        ax.plot(rank, axis_y, marker="o", color=color, markersize=7, zorder=5)

        # Alternating top / bottom
        is_top = (idx % 2 == 0)
        if is_top:
            max_bar_offset = (len(bar_levels) * bar_step_y) if bar_levels else 0.0
            stem_top = axis_y + stem_length + max_bar_offset * 0.5
            ax.plot(
                [rank, rank],
                [axis_y, stem_top],
                color="gray",
                linestyle="--",
                linewidth=1.0,
                zorder=1,
            )
            ax.text(
                rank,
                stem_top + 0.04,
                f"{name} ({rank:.2f})",
                rotation=45,
                ha="left",
                va="bottom",
                fontsize=9.5,
                fontweight="semibold",
                color="#1a1a1a",
            )
        else:
            stem_bottom = axis_y - stem_length
            ax.plot(
                [rank, rank],
                [axis_y, stem_bottom],
                color="gray",
                linestyle="--",
                linewidth=1.0,
                zorder=1,
            )
            ax.text(
                rank,
                stem_bottom - 0.04,
                f"{name} ({rank:.2f})",
                rotation=-45,
                ha="left",
                va="top",
                fontsize=9.5,
                fontweight="semibold",
                color="#1a1a1a",
            )

    # Draw CD scale bar in the top-left corner
    cd_bar_start_x = 1.0
    cd_bar_end_x = 1.0 + cd
    cd_bar_y = 0.95 + (len(bar_levels) * bar_step_y if bar_levels else 0.0)

    ax.plot(
        [cd_bar_start_x, cd_bar_end_x],
        [cd_bar_y, cd_bar_y],
        color="#c0392b",
        linewidth=2.5,
        zorder=6,
    )
    cd_cap = 0.04
    ax.plot(
        [cd_bar_start_x, cd_bar_start_x],
        [cd_bar_y - cd_cap, cd_bar_y + cd_cap],
        color="#c0392b",
        linewidth=2.0,
        zorder=6,
    )
    ax.plot(
        [cd_bar_end_x, cd_bar_end_x],
        [cd_bar_y - cd_cap, cd_bar_y + cd_cap],
        color="#c0392b",
        linewidth=2.0,
        zorder=6,
    )

    cd_mid_x = (cd_bar_start_x + cd_bar_end_x) / 2.0
    ax.text(
        cd_mid_x,
        cd_bar_y + 0.06,
        f"CD = {cd:.3f} (α={alpha})",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#c0392b",
    )

    # Determine axis limits
    x_min = min(0.5, 1.0 - 0.2)
    x_max = max(k + 0.8, cd_bar_end_x + 0.5)
    y_min = -0.95
    y_max = cd_bar_y + 0.30

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    fig.tight_layout()

    if save_path:
        out_path = Path(save_path)
        if out_path.parent:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    return fig
