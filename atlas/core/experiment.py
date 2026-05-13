"""Experiment scheduler and manager."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
from tqdm import tqdm

from atlas.core.base_algorithm import BaseAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.core.result import Result
from atlas.utils.logger import get_logger
from atlas.utils.registry import get_algorithm, get_problem
from atlas.utils.saver import ExperimentSaver


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run.

    Attributes:
        algorithms: List of algorithm names or classes.
        problems: List of problem names or classes.
        dims: Dimensionality for each problem (broadcast if single int).
        max_iter: Maximum iterations per run.
        runs: Number of independent runs.
        seed: Base random seed.
        pop_size: Population size override (``None`` = algorithm default).
        algo_params: Per-algorithm parameter overrides.
        enable_plots: Dict of plot-name -> bool flags.
        save_results: Whether to persist results to disk.
        verbose: Print per-iteration info.
        base_dir: Root output directory.
    """

    algorithms: List[str] = field(default_factory=lambda: ["pso"])
    problems: List[str] = field(default_factory=lambda: ["sphere"])
    dims: List[int] = field(default_factory=lambda: [30])
    max_iter: int = 500
    runs: int = 30
    seed: int = 42
    pop_size: Optional[int] = None
    algo_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    enable_plots: Dict[str, bool] = field(
        default_factory=lambda: {
            "convergence": True,
            "boxplot": True,
            "heatmap": True,
        }
    )
    save_results: bool = True
    verbose: bool = False
    base_dir: str = "results"


class Experiment:
    """Orchestrates the execution of algorithm × problem combinations.

    The typical usage is::

        cfg = ExperimentConfig(algorithms=["pso", "ga"], problems=["sphere"])
        exp = Experiment(cfg)
        all_results = exp.run_all()

    Args:
        config: An :class:`ExperimentConfig` instance.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.logger = get_logger("atlas.experiment")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_all(self) -> Dict[str, Dict[str, List[Result]]]:
        """Run every algorithm–problem combination.

        Returns:
            Nested dict ``{problem_name: {algo_name: [Result, ...]}}``.
        """
        cfg = self.config
        dims = self._broadcast_dims()
        all_results: Dict[str, Dict[str, List[Result]]] = {}

        for prob_name, dim in zip(cfg.problems, dims):
            problem = self._make_problem(prob_name, dim)
            all_results[prob_name] = {}

            for algo_name in cfg.algorithms:
                self.logger.info(
                    "Running %s on %s (dim=%d, runs=%d, max_iter=%d)",
                    algo_name, prob_name, dim, cfg.runs, cfg.max_iter,
                )
                results = self._run_single(algo_name, problem, prob_name)
                all_results[prob_name][algo_name] = results

                # Summary statistics
                fitness_vals = np.array([r.best_fitness for r in results])
                self.logger.info(
                    "  %s on %s -> mean=%.4e std=%.4e best=%.4e",
                    algo_name, prob_name,
                    np.mean(fitness_vals), np.std(fitness_vals),
                    np.min(fitness_vals),
                )

                # Optionally save
                if cfg.save_results:
                    saver = ExperimentSaver(prob_name, algo_name, cfg.base_dir)
                    saver.save_csv(results)
                    saver.save_summary(results)
                    saver.save_convergence_csv(results)
                    saver.save_config({
                        "algorithm": algo_name,
                        "problem": prob_name,
                        "dim": dim,
                        "max_iter": cfg.max_iter,
                        "runs": cfg.runs,
                        "seed": cfg.seed,
                        "algo_params": cfg.algo_params.get(algo_name, {}),
                    })

        # Optional: multi-algorithm plots
        if cfg.save_results and len(cfg.algorithms) > 1:
            self._generate_comparative_plots(all_results, dims)

        return all_results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _broadcast_dims(self) -> List[int]:
        dims = self.config.dims
        probs = self.config.problems
        if len(dims) == 1:
            return dims * len(probs)
        if len(dims) != len(probs):
            raise ValueError(
                f"Length of dims ({len(dims)}) must be 1 or match "
                f"number of problems ({len(probs)})."
            )
        return dims

    def _make_problem(self, name: str, dim: int) -> BaseProblem:
        cls = get_problem(name)
        return cls(dim=dim)

    def _make_algorithm(
        self,
        algo_name: str,
        problem: BaseProblem,
        seed: int,
    ) -> BaseAlgorithm:
        cls = get_algorithm(algo_name)
        params = dict(self.config.algo_params.get(algo_name, {}))
        if self.config.pop_size is not None:
            params.setdefault("pop_size", self.config.pop_size)
        return cls(
            problem=problem,
            max_iter=self.config.max_iter,
            seed=seed,
            verbose=self.config.verbose,
            **params,
        )

    def _run_single(
        self,
        algo_name: str,
        problem: BaseProblem,
        prob_name: str,
    ) -> List[Result]:
        results: List[Result] = []
        for run_idx in tqdm(
            range(self.config.runs),
            desc=f"{algo_name:>6s} | {prob_name}",
            ncols=80,
        ):
            seed = self.config.seed + run_idx
            algo = self._make_algorithm(algo_name, problem, seed)
            result = algo.run(run_id=run_idx)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Comparative plots
    # ------------------------------------------------------------------
    def _generate_comparative_plots(
        self,
        all_results: Dict[str, Dict[str, List[Result]]],
        dims: List[int],
    ) -> None:
        from atlas.visualization.plot_manager import PlotManager

        cfg = self.config
        pm = PlotManager()

        # Convergence comparison
        if cfg.enable_plots.get("convergence", False):
            for prob_name in cfg.problems:
                fig = pm.plot_convergence(
                    all_results[prob_name],
                    title=f"Convergence on {prob_name}",
                )
                if fig is not None and cfg.save_results:
                    saver = ExperimentSaver(prob_name, "comparison", cfg.base_dir)
                    saver.save_plot(fig, "convergence_comparison")

        # Box-plot comparison
        if cfg.enable_plots.get("boxplot", False):
            for prob_name in cfg.problems:
                fig = pm.plot_boxplot(
                    all_results[prob_name],
                    title=f"Results on {prob_name}",
                )
                if fig is not None and cfg.save_results:
                    saver = ExperimentSaver(prob_name, "comparison", cfg.base_dir)
                    saver.save_plot(fig, "boxplot_comparison")

        # Heatmap
        if cfg.enable_plots.get("heatmap", False):
            fig = pm.plot_heatmap(
                all_results,
                algo_names=cfg.algorithms,
                problem_names=cfg.problems,
            )
            if fig is not None and cfg.save_results:
                saver = ExperimentSaver("all_problems", "comparison", cfg.base_dir)
                saver.save_plot(fig, "heatmap")
