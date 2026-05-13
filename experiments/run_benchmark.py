"""Run benchmark experiments from the command line.

Usage examples::

    # Run PSO, GA, DE on Sphere and Rastrigin with 30 dimensions
    python experiments/run_benchmark.py --algorithms pso ga de \\
        --problems sphere rastrigin --dim 30 --max_iter 500 --runs 30

    # Quick test with fewer runs
    python experiments/run_benchmark.py --algorithms pso --problems sphere \\
        --dim 10 --max_iter 100 --runs 5 --seed 0

You can also import and call ``run_benchmark`` from a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Ensure project root is on sys.path when running as a script
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import atlas  # noqa: E402  – triggers registration
from atlas.core.experiment import Experiment, ExperimentConfig  # noqa: E402
from atlas.core.result import Result  # noqa: E402
from atlas.utils.logger import get_logger  # noqa: E402
from atlas.utils.problem_groups import expand_problem_names, get_all_supported_dims  # noqa: E402


def print_summary_table(
    all_results: Dict[str, Dict[str, List[Result]]],
) -> None:
    """Print a formatted result summary table to the terminal."""
    header = f"{'Problem':<20s} {'Algorithm':<10s} {'Mean':>14s} {'Std':>14s} {'Best':>14s} {'Worst':>14s}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for prob_name, algos in sorted(all_results.items()):
        for algo_name, results in sorted(algos.items()):
            vals = np.array([r.best_fitness for r in results])
            print(
                f"{prob_name:<20s} {algo_name:<10s} "
                f"{np.mean(vals):>14.6e} {np.std(vals):>14.6e} "
                f"{np.min(vals):>14.6e} {np.max(vals):>14.6e}"
            )
    print(sep)


def run_benchmark(
    algorithms: List[str],
    problems: List[str],
    dims: Optional[List[int]] = None,
    max_iter: int = 500,
    runs: int = 30,
    seed: int = 42,
    pop_size: Optional[int] = None,
    algo_params: Optional[Dict[str, Dict]] = None,
    plots: Optional[List[str]] = None,
    save_results: bool = True,
    verbose: bool = False,
    base_dir: str = "results",
    stopping_criterion: str = "iterations",
    max_nfe: int = 10000,
) -> Dict[str, Dict[str, List[Result]]]:
    """Run a benchmark experiment (also usable from Jupyter).

    Args:
        algorithms: List of algorithm names (e.g. ``["pso", "ga_nfe"]``).
        problems: List of problem names.
        dims: Dimension per problem (broadcasts if single element). Use ``[0]`` or ``[-1]`` for all supported dims.
        max_iter: Iterations per run (for iteration-based algorithms).
        runs: Number of independent runs.
        seed: Base random seed.
        pop_size: Global population size override.
        algo_params: Per-algorithm keyword arguments.
        plots: List of plot types to enable (e.g. ``["convergence", "boxplot"]``).
        save_results: Persist results to disk.
        verbose: Print per-iteration info.
        base_dir: Root output directory.
        stopping_criterion: ``"iterations"`` or ``"nfe"``.
        max_nfe: Maximum function evaluations (for NFE-based algorithms).

    Returns:
        Nested dict ``{problem: {algo: [Result, ...]}}``.
    """
    # Expand CEC suite group names (e.g. "cec2017" -> all cec2017_f* functions)
    problems = expand_problem_names(problems)

    # Handle "all" dimensions: use sentinel value -1
    if dims is not None and len(dims) == 1 and dims[0] == -1:
        dims = get_all_supported_dims(problems)
        print(f"Testing all supported dimensions: {dims}")

    if dims is None:
        dims = [30]
    if algo_params is None:
        algo_params = {}
    if plots is None:
        plots = ["convergence", "boxplot", "heatmap"]

    enable_plots = {
        "convergence": "convergence" in plots,
        "boxplot": "boxplot" in plots,
        "heatmap": "heatmap" in plots,
        "fitness_landscape": "fitness_landscape" in plots,
    }

    config = ExperimentConfig(
        algorithms=algorithms,
        problems=problems,
        dims=dims,
        max_iter=max_iter,
        runs=runs,
        seed=seed,
        pop_size=pop_size,
        algo_params=algo_params,
        enable_plots=enable_plots,
        save_results=save_results,
        verbose=verbose,
        base_dir=base_dir,
        stopping_criterion=stopping_criterion,
        max_nfe=max_nfe,
    )

    experiment = Experiment(config)
    all_results = experiment.run_all()

    # Print summary table
    print_summary_table(all_results)

    return all_results


# ---- CLI entry point ------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ATLAS – Run metaheuristic benchmark experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--algorithms", nargs="+", default=["pso"],
        help="Algorithm names (default: pso)",
    )
    parser.add_argument(
        "--problems", nargs="+", default=["sphere"],
        help="Problem names (default: sphere)",
    )
    parser.add_argument(
        "--dim", type=str, nargs="+", default=["30"],
        help="Dimension(s) or 'all' for all supported dims (default: 30)",
    )
    parser.add_argument("--max_iter", type=int, default=500, help="Max iterations")
    parser.add_argument(
        "--stopping_criterion", choices=["iterations", "nfe"], default="iterations",
        help="Stopping criterion: 'iterations' or 'nfe' (default: iterations)",
    )
    parser.add_argument("--max_nfe", type=int, default=10000, help="Max function evaluations (for NFE-based)")
    parser.add_argument("--runs", type=int, default=30, help="Number of runs")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--pop_size", type=int, default=None, help="Population size override")
    parser.add_argument(
        "--plots", nargs="+", default=["convergence", "boxplot", "heatmap"],
        help="Plot types to generate",
    )
    parser.add_argument(
        "--no_save_results", action="store_true", default=False,
        help="Disable saving results to disk",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-iteration info")
    parser.add_argument("--base_dir", type=str, default="results", help="Output directory")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    # Parse dim arguments: convert "all" to [-1] sentinel, otherwise convert to int
    if len(args.dim) == 1 and args.dim[0].lower() == "all":
        dims = [-1]
    else:
        dims = [int(d) for d in args.dim]
    run_benchmark(
        algorithms=args.algorithms,
        problems=args.problems,
        dims=dims,
        max_iter=args.max_iter,
        runs=args.runs,
        seed=args.seed,
        pop_size=args.pop_size,
        plots=args.plots,
        save_results=not args.no_save_results,
        verbose=args.verbose,
        base_dir=args.base_dir,
        stopping_criterion=args.stopping_criterion,
        max_nfe=args.max_nfe,
    )


if __name__ == "__main__":
    main()
