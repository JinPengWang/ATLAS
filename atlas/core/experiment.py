"""Experiment scheduler and manager with multi-processing and statistical reporting."""

from __future__ import annotations

import concurrent.futures
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import yaml
from tqdm import tqdm

from atlas.core.base_algorithm import BaseAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.core.result import Result
from atlas.stats.hypothesis_testing import calculate_average_ranks, friedman_test, wilcoxon_signed_rank_test
from atlas.stats.table_exporter import export_csv_summary, export_latex_table, export_markdown_table
from atlas.utils.logger import get_logger
from atlas.utils.problem_groups import expand_problem_names
from atlas.utils.registry import get_algorithm, get_problem
from atlas.utils.saver import ExperimentSaver


def _execute_single_run(
    algo_name: str,
    prob_name: str,
    dim: int,
    run_idx: int,
    seed: int,
    max_iter: int,
    max_nfe: int,
    stopping_criterion: str,
    pop_size: Optional[int],
    verbose: bool,
    algo_params: Dict[str, Any],
) -> Result:
    """Top-level picklable worker function for multiprocessing on Windows/Linux."""
    cls_prob = get_problem(prob_name)
    problem = cls_prob(dim=dim)
    cls_algo = get_algorithm(algo_name)

    params = dict(algo_params)
    if pop_size is not None:
        params.setdefault("pop_size", pop_size)
    if stopping_criterion == "nfe" or algo_name.endswith("_nfe"):
        params["max_nfe"] = max_nfe
        params["stopping_criterion"] = "nfe"

    algo = cls_algo(
        problem=problem,
        max_iter=max_iter,
        seed=seed,
        verbose=verbose,
        **params,
    )
    return algo.run(run_id=run_idx)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run.

    Attributes:
        algorithms: List of algorithm names or classes.
        problems: List of problem names or classes.
        dims: Dimensionalities to test (runs Cartesian product if multiple).
        max_iter: Maximum iterations per run.
        runs: Number of independent runs.
        seed: Base random seed.
        pop_size: Population size override (``None`` = algorithm default).
        stopping_criterion: "iterations" or "nfe".
        max_nfe: Function evaluation budget when stopping_criterion="nfe".
        n_jobs: Number of parallel worker processes (1 = sequential).
        algo_params: Per-algorithm parameter overrides.
        enable_plots: Dict of plot-name -> bool flags.
        generate_stats: Whether to compute Wilcoxon/Friedman tests & export LaTeX/MD tables.
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
    stopping_criterion: str = "iterations"  # "iterations" or "nfe"
    max_nfe: int = 10000
    n_jobs: int = 1
    algo_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    enable_plots: Dict[str, bool] = field(
        default_factory=lambda: {
            "convergence": True,
            "boxplot": True,
            "heatmap": True,
        }
    )
    generate_stats: bool = True
    save_results: bool = True
    verbose: bool = False
    base_dir: str = "results"

    @classmethod
    def from_yaml(cls, path_or_content: Union[str, Path]) -> "ExperimentConfig":
        """Load experiment configuration from YAML file or YAML string."""
        p = Path(path_or_content)
        if p.exists() and p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            data = yaml.safe_load(str(path_or_content))
        return cls(**(data or {}))

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)


class Experiment:
    """Orchestrates the execution of algorithm × problem × dimension combinations.

    Supports multi-processing (`n_jobs > 1`) and automatic generation of
    statistical test results (Wilcoxon, Friedman) and LaTeX / Markdown tables.

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
        """Run every algorithm–problem–dimension combination.

        Returns:
            Nested dict ``{problem_key: {algo_name: [Result, ...]}}``.
        """
        cfg = self.config
        cfg.problems = expand_problem_names(cfg.problems)
        
        # Build Cartesian product list of (prob_name, dim)
        combos: List[Tuple[str, int, str]] = []
        is_single_dim = len(cfg.dims) == 1
        for prob_name in cfg.problems:
            for dim in cfg.dims:
                key = prob_name if is_single_dim else f"{prob_name}_D{dim}"
                combos.append((prob_name, dim, key))

        all_results: Dict[str, Dict[str, List[Result]]] = {}

        for prob_name, dim, key in combos:
            all_results[key] = {}

            for algo_name in cfg.algorithms:
                self.logger.info(
                    "Running %s on %s (dim=%d, runs=%d, max_iter=%d, n_jobs=%d)",
                    algo_name, prob_name, dim, cfg.runs, cfg.max_iter, cfg.n_jobs,
                )
                results = self._run_single(algo_name, prob_name, dim)
                all_results[key][algo_name] = results

                # Summary statistics
                fitness_vals = np.array([r.best_fitness for r in results])
                self.logger.info(
                    "  %s on %s (D=%d) -> mean=%.4e std=%.4e best=%.4e",
                    algo_name, prob_name, dim,
                    np.mean(fitness_vals), np.std(fitness_vals),
                    np.min(fitness_vals),
                )

                # Optionally save per-algorithm outputs
                if cfg.save_results:
                    saver = ExperimentSaver(key, algo_name, cfg.base_dir)
                    saver.save_csv(results)
                    saver.save_summary(results)
                    saver.save_convergence_csv(results)
                    saver.save_config({
                        "algorithm": algo_name,
                        "problem": prob_name,
                        "dim": dim,
                        "max_iter": cfg.max_iter,
                        "stopping_criterion": cfg.stopping_criterion,
                        "max_nfe": cfg.max_nfe,
                        "runs": cfg.runs,
                        "seed": cfg.seed,
                        "algo_params": cfg.algo_params.get(algo_name, {}),
                    })

        # Save statistical reports & academic tables
        if cfg.save_results and cfg.generate_stats:
            self._generate_statistical_reports(all_results)

        # Generate plots
        if cfg.save_results:
            self._generate_plots(all_results)

        return all_results

    # ------------------------------------------------------------------
    # Execution Helpers
    # ------------------------------------------------------------------
    def _run_single(
        self,
        algo_name: str,
        prob_name: str,
        dim: int,
    ) -> List[Result]:
        cfg = self.config
        algo_params = cfg.algo_params.get(algo_name, {})

        if cfg.n_jobs > 1:
            tasks = [
                (
                    algo_name,
                    prob_name,
                    dim,
                    run_idx,
                    cfg.seed + run_idx,
                    cfg.max_iter,
                    cfg.max_nfe,
                    cfg.stopping_criterion,
                    cfg.pop_size,
                    cfg.verbose,
                    algo_params,
                )
                for run_idx in range(cfg.runs)
            ]
            with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.n_jobs) as executor:
                futures = [executor.submit(_execute_single_run, *t) for t in tasks]
                results = [
                    f.result()
                    for f in tqdm(
                        futures,
                        desc=f"{algo_name:>6s} | {prob_name} (D={dim}) [Parallel]",
                        ncols=80,
                    )
                ]
                results.sort(key=lambda r: r.run_id)
                return results

        # Sequential execution
        results: List[Result] = []
        for run_idx in tqdm(
            range(cfg.runs),
            desc=f"{algo_name:>6s} | {prob_name} (D={dim})",
            ncols=80,
        ):
            seed = cfg.seed + run_idx
            res = _execute_single_run(
                algo_name,
                prob_name,
                dim,
                run_idx,
                seed,
                cfg.max_iter,
                cfg.max_nfe,
                cfg.stopping_criterion,
                cfg.pop_size,
                cfg.verbose,
                algo_params,
            )
            results.append(res)
        return results

    # ------------------------------------------------------------------
    # Statistical Analysis & Academic Table Generation
    # ------------------------------------------------------------------
    def _generate_statistical_reports(
        self,
        all_results: Dict[str, Dict[str, List[Result]]],
    ) -> None:
        """Compute statistical tests and export LaTeX, Markdown, and CSV tables."""
        cfg = self.config
        import pandas as pd

        rows = []
        for prob_key, algos_dict in all_results.items():
            for algo_name, res_list in algos_dict.items():
                fits = [r.best_fitness for r in res_list]
                times = [r.wall_time_sec for r in res_list]
                rows.append({
                    "problem": prob_key,
                    "algorithm": algo_name,
                    "mean": float(np.mean(fits)),
                    "std": float(np.std(fits)),
                    "median": float(np.median(fits)),
                    "best": float(np.min(fits)),
                    "worst": float(np.max(fits)),
                    "iqr": float(np.percentile(fits, 75) - np.percentile(fits, 25)),
                    "mean_time_sec": float(np.mean(times)),
                })

        if not rows:
            return

        df_summary = pd.DataFrame(rows)
        out_dir = Path(cfg.base_dir) / "statistical_reports"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Export CSV summary
        export_csv_summary(df_summary, out_dir / "summary_statistics.csv")

        # 2. Export LaTeX table
        export_latex_table(
            df_summary,
            output_path=out_dir / "benchmark_table.tex",
            caption=f"Optimization results across benchmark functions ({cfg.runs} runs)",
        )

        # 3. Export Markdown table
        export_markdown_table(df_summary, output_path=out_dir / "summary_statistics.md")

        # 4. Multi-algorithm Friedman test & Wilcoxon pairwise test (if > 1 algorithm)
        if len(cfg.algorithms) > 1 and len(all_results) > 1:
            perf_dict: Dict[str, List[float]] = {algo: [] for algo in cfg.algorithms}
            for prob_key, algos_dict in all_results.items():
                for algo in cfg.algorithms:
                    if algo in algos_dict:
                        perf_dict[algo].append(float(np.mean([r.best_fitness for r in algos_dict[algo]])))

            try:
                friedman_res = friedman_test(perf_dict)
                with open(out_dir / "friedman_test.txt", "w", encoding="utf-8") as f:
                    f.write("=== Friedman Ranking Test ===\n")
                    f.write(f"Statistic (Chi-square): {friedman_res.stat:.4f}\n")
                    f.write(f"p-value: {friedman_res.p_value:.4e}\n")
                    f.write(f"Significant at alpha={friedman_res.alpha}: {friedman_res.is_significant}\n\n")
                    f.write("Average Ranks (lower is better):\n")
                    for algo, rank in friedman_res.rankings_order:
                        f.write(f"  {algo:<15s}: {rank:.4f}\n")
            except Exception as e:
                self.logger.warning("Friedman test could not be computed: %s", e)

    # ------------------------------------------------------------------
    # Plot Generation
    # ------------------------------------------------------------------
    def _generate_plots(
        self,
        all_results: Dict[str, Dict[str, List[Result]]],
    ) -> None:
        from atlas.visualization.plot_manager import PlotManager

        cfg = self.config
        pm = PlotManager()
        is_comparison = len(cfg.algorithms) > 1

        for prob_key in all_results.keys():
            # Convergence plot
            if cfg.enable_plots.get("convergence", False):
                fig = pm.plot_convergence(
                    all_results[prob_key],
                    title=f"Convergence on {prob_key}",
                )
                if fig is not None:
                    label = "comparison" if is_comparison else cfg.algorithms[0]
                    saver = ExperimentSaver(prob_key, label, cfg.base_dir)
                    saver.save_plot(fig, f"convergence_{prob_key}")

            # Box-plot
            if cfg.enable_plots.get("boxplot", False):
                fig = pm.plot_boxplot(
                    all_results[prob_key],
                    title=f"Results on {prob_key}",
                )
                if fig is not None:
                    label = "comparison" if is_comparison else cfg.algorithms[0]
                    saver = ExperimentSaver(prob_key, label, cfg.base_dir)
                    saver.save_plot(fig, f"boxplot_{prob_key}")

        # Heatmap
        if cfg.enable_plots.get("heatmap", False) and is_comparison:
            fig = pm.plot_heatmap(
                all_results,
                algo_names=cfg.algorithms,
                problem_names=list(all_results.keys()),
            )
            if fig is not None:
                saver = ExperimentSaver("all_problems", "comparison", cfg.base_dir)
                saver.save_plot(fig, "heatmap")
