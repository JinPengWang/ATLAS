"""Run a custom optimisation problem with ATLAS.

This script demonstrates how to:
1. Define a custom problem (here: Pressure Vessel Design).
2. Run multiple algorithms on it.
3. Compare results and generate plots.

Usage::

    python experiments/run_custom.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

import atlas  # noqa: E402 – triggers registration
from atlas.core.experiment import Experiment, ExperimentConfig  # noqa: E402
from atlas.problems.custom.example_custom import PressureVessel  # noqa: E402
from atlas.utils.logger import get_logger  # noqa: E402


def main() -> None:
    logger = get_logger("run_custom")
    logger.info("=== ATLAS – Custom Problem Demo ===")
    logger.info("Problem: Pressure Vessel Design")

    # The PressureVessel problem is already registered.
    # We can reference it by name or use the class directly.
    config = ExperimentConfig(
        algorithms=["pso", "de", "ga"],
        problems=["pressure_vessel"],
        dims=[4],  # fixed 4-D problem
        max_iter=300,
        runs=10,
        seed=42,
        pop_size=50,
        algo_params={
            "pso": {"w": 0.6, "c1": 1.8, "c2": 1.8},
            "de": {"F": 0.7, "CR": 0.9, "strategy": "best/1/bin"},
            "ga": {"crossover_prob": 0.85, "mutation_prob": 0.02},
        },
        enable_plots={
            "convergence": True,
            "boxplot": True,
            "heatmap": False,
            "fitness_landscape": False,
        },
        save_results=True,
        verbose=True,
        base_dir="results",
    )

    experiment = Experiment(config)
    all_results = experiment.run_all()

    # Print summary
    print("\n=== Summary ===")
    for prob_name, algos in all_results.items():
        for algo_name, results in algos.items():
            vals = np.array([r.best_fitness for r in results])
            print(
                f"  {algo_name:<8s} on {prob_name}: "
                f"mean={np.mean(vals):.4f}, best={np.min(vals):.4f}"
            )

    # Compare with known optimum
    problem = PressureVessel()
    opt = problem.get_optimum()
    if opt is not None:
        print(f"\n  Known optimum: {opt:.6f}")
        for algo_name, results in all_results.get("pressure_vessel", {}).items():
            vals = np.array([r.best_fitness for r in results])
            gap = (np.min(vals) - opt) / abs(opt) * 100
            print(f"  {algo_name:<8s} gap to optimum: {gap:.2f}%")


if __name__ == "__main__":
    main()
