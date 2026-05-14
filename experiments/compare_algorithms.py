"""Multi-algorithm comparison on a set of benchmark functions.

Generates convergence curves, box-plots, and a heatmap in one run.

Usage::

    python experiments/compare_algorithms.py
    python experiments/compare_algorithms.py --algorithms pso de woa --dim 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import atlas  # noqa: E402
from experiments.run_benchmark import run_benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="ATLAS multi-algorithm comparison")
    parser.add_argument(
        "--algorithms", nargs="+",
        default=["pso", "ga", "de", "dp", "tjo", "lgc", "ppo", "lea", "psa", "woa", "sa", "aco"],
        help="Algorithms to compare",
    )
    parser.add_argument(
        "--problems", nargs="+",
        default=["sphere", "rosenbrock", "rastrigin", "ackley", "griewank"],
        help="Benchmark problems",
    )
    parser.add_argument("--dim", type=int, nargs="+", default=[30])
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pop_size", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_benchmark(
        algorithms=args.algorithms,
        problems=args.problems,
        dims=args.dim,
        max_iter=args.max_iter,
        runs=args.runs,
        seed=args.seed,
        pop_size=args.pop_size,
        plots=["convergence", "boxplot", "heatmap"],
        save_results=True,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
