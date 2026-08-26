"""Statistical hypothesis testing module for benchmark comparisons.

Implements non-parametric statistical tests commonly recommended for
comparing metaheuristic and evolutionary optimization algorithms:
- Wilcoxon signed-rank test (pairwise comparison)
- Friedman test (multiple-algorithm comparison)
- Average ranking and post-hoc analysis

References:
    Derrac, J., García, S., Molina, D., & Herrera, F. (2011).
    A practical tutorial on the use of nonparametric statistical tests as
    methodologies for comparison in evolutionary and swarm intelligence algorithms.
    *Swarm and Evolutionary Computation*, 1(1), 3-18.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats


@dataclass
class PairwiseTestResult:
    """Result of a pairwise Wilcoxon signed-rank test."""
    stat: float
    p_value: float
    significance: str  # '+', '-', or '=' (relative to baseline / algo_a)
    alpha: float
    summary: str


@dataclass
class FriedmanTestResult:
    """Result of a Friedman ranking test."""
    stat: float
    p_value: float
    is_significant: bool
    alpha: float
    average_ranks: Dict[str, float]
    rankings_order: List[Tuple[str, float]]


def wilcoxon_signed_rank_test(
    sample_a: Union[List[float], np.ndarray],
    sample_b: Union[List[float], np.ndarray],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> PairwiseTestResult:
    """Perform Wilcoxon signed-rank test between two paired algorithm runs.

    Args:
        sample_a: Fitness values of algorithm A across runs (or problems).
        sample_b: Fitness values of algorithm B across identical runs (or problems).
        alpha: Significance level (default 0.05).
        alternative: "two-sided", "less", or "greater".

    Returns:
        :class:`PairwiseTestResult` with test statistic, p-value, and comparison sign.
        Sign meaning (for minimization):
          '+' : Algorithm A is significantly better than Algorithm B (A < B).
          '-' : Algorithm A is significantly worse than Algorithm B (A > B).
          '=' : No statistically significant difference.
    """
    arr_a = np.asarray(sample_a, dtype=float)
    arr_b = np.asarray(sample_b, dtype=float)

    if len(arr_a) != len(arr_b):
        raise ValueError(f"Sample lengths must match: {len(arr_a)} vs {len(arr_b)}")

    diff = arr_a - arr_b

    # All identical differences
    if np.allclose(diff, 0.0):
        return PairwiseTestResult(
            stat=0.0,
            p_value=1.0,
            significance="=",
            alpha=alpha,
            summary=f"Identical distributions (p=1.0000 >= {alpha})",
        )

    try:
        res = stats.wilcoxon(arr_a, arr_b, alternative=alternative, zero_method="pratt")
        stat = float(res.statistic)
        p_val = float(res.pvalue)
    except Exception:
        # Fallback if zero differences cause issues in older scipy
        non_zero_diff = diff[~np.isclose(diff, 0.0)]
        if len(non_zero_diff) == 0:
            return PairwiseTestResult(0.0, 1.0, "=", alpha, "Identical samples")
        res = stats.wilcoxon(non_zero_diff, alternative=alternative)
        stat = float(res.statistic)
        p_val = float(res.pvalue)

    # Determine direction for minimization:
    # If p < alpha: check if median(a) < median(b)
    if p_val < alpha:
        med_a = np.median(arr_a)
        med_b = np.median(arr_b)
        if med_a < med_b:
            sign = "+"
        elif med_a > med_b:
            sign = "-"
        else:
            mean_a = np.mean(arr_a)
            mean_b = np.mean(arr_b)
            sign = "+" if mean_a < mean_b else ("-" if mean_a > mean_b else "=")
    else:
        sign = "="

    summary = (
        f"p-value={p_val:.4e} ({'significant' if sign != '=' else 'not significant'} at alpha={alpha})"
    )

    return PairwiseTestResult(
        stat=stat,
        p_value=p_val,
        significance=sign,
        alpha=alpha,
        summary=summary,
    )


def calculate_average_ranks(
    performance_matrix: Union[Dict[str, List[float]], np.ndarray],
    algorithm_names: Optional[List[str]] = None,
    higher_is_better: bool = False,
) -> Dict[str, float]:
    """Compute Friedman average ranks for multiple algorithms across problems/instances.

    Args:
        performance_matrix: Either a 2D array where rows=problems, columns=algorithms,
            or a dictionary {algo_name: [scores_across_problems]}.
        algorithm_names: Names corresponding to columns (if 2D array passed).
        higher_is_better: True if maximizing, False if minimizing.

    Returns:
        Dictionary mapping algorithm name to its average rank (1.0 is best).
    """
    if isinstance(performance_matrix, dict):
        algo_names = list(performance_matrix.keys())
        matrix = np.column_stack([performance_matrix[k] for k in algo_names])
    else:
        matrix = np.asarray(performance_matrix, dtype=float)
        algo_names = (
            algorithm_names
            if algorithm_names is not None
            else [f"Algo_{i}" for i in range(matrix.shape[1])]
        )

    n_problems, n_algos = matrix.shape

    # Compute ranks per row (problem)
    ranks = np.zeros((n_problems, n_algos), dtype=float)
    for i in range(n_problems):
        row = matrix[i, :]
        if higher_is_better:
            ranks[i, :] = stats.rankdata(-row, method="average")
        else:
            ranks[i, :] = stats.rankdata(row, method="average")

    avg_ranks = np.mean(ranks, axis=0)
    return {name: float(rank) for name, rank in zip(algo_names, avg_ranks)}


def friedman_test(
    performance_matrix: Union[Dict[str, List[float]], np.ndarray],
    algorithm_names: Optional[List[str]] = None,
    alpha: float = 0.05,
    higher_is_better: bool = False,
) -> FriedmanTestResult:
    """Perform Friedman test across multiple algorithms on a set of benchmark problems.

    Args:
        performance_matrix: 2D array (rows=problems, cols=algorithms) or dict of score lists.
        algorithm_names: Names of the algorithms if matrix is a 2D array.
        alpha: Significance level.
        higher_is_better: Whether higher values mean better performance.

    Returns:
        :class:`FriedmanTestResult` containing test statistics and sorted average ranks.
    """
    if isinstance(performance_matrix, dict):
        algo_names = list(performance_matrix.keys())
        samples = [performance_matrix[k] for k in algo_names]
        matrix = np.column_stack(samples)
    else:
        matrix = np.asarray(performance_matrix, dtype=float)
        algo_names = (
            algorithm_names
            if algorithm_names is not None
            else [f"Algo_{i}" for i in range(matrix.shape[1])]
        )
        samples = [matrix[:, j] for j in range(matrix.shape[1])]

    # Calculate scipy friedmanchisquare
    stat_res = stats.friedmanchisquare(*samples)
    stat = float(stat_res.statistic)
    p_val = float(stat_res.pvalue)

    avg_ranks = calculate_average_ranks(matrix, algorithm_names=algo_names, higher_is_better=higher_is_better)
    sorted_ranks = sorted(avg_ranks.items(), key=lambda x: x[1])

    return FriedmanTestResult(
        stat=stat,
        p_value=p_val,
        is_significant=bool(p_val < alpha),
        alpha=alpha,
        average_ranks=avg_ranks,
        rankings_order=sorted_ranks,
    )
