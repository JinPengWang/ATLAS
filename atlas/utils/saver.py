"""Result saving utilities for ATLAS experiments."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from atlas.core.result import Result


class ExperimentSaver:
    """Manages the output directory structure and persists experiment artefacts.

    Directory layout::

        results/{problem_name}/{timestamp}_{exp_label}/
            config.yaml
            raw_results.csv
            summary.csv
            figures/
            logs/

    Args:
        problem_name: Name of the optimisation problem.
        exp_label: Short human-readable label for the experiment.
        base_dir: Root results directory (default ``results``).
    """

    def __init__(
        self,
        problem_name: str,
        exp_label: str = "exp",
        base_dir: str = "results",
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(base_dir) / problem_name / f"{timestamp}_{exp_label}"
        self.figures_dir = self.exp_dir / "figures"
        self.logs_dir = self.exp_dir / "logs"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def save_config(self, config: Dict[str, Any]) -> Path:
        """Save experiment configuration as YAML.

        Args:
            config: Dictionary of configuration parameters.

        Returns:
            Path to the saved YAML file.
        """
        path = self.exp_dir / "config.yaml"
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(config, fh, default_flow_style=False, allow_unicode=True)
        return path

    # ------------------------------------------------------------------
    # Raw results
    # ------------------------------------------------------------------
    def save_csv(self, results: List[Result]) -> Path:
        """Save per-run final fitness values and iteration histories to CSV.

        Args:
            results: List of :class:`Result` objects (one per run).

        Returns:
            Path to the saved CSV file.
        """
        rows = []
        for res in results:
            row: Dict[str, Any] = {
                "algorithm": res.algorithm_name,
                "problem": res.problem_name,
                "run": res.run_id,
                "best_fitness": res.best_fitness,
                "seed": res.seed,
            }
            row.update(res.best_solution_dict())
            rows.append(row)
        df = pd.DataFrame(rows)
        path = self.exp_dir / "raw_results.csv"
        df.to_csv(path, index=False)
        return path

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    def save_summary(self, results: List[Result]) -> Path:
        """Save statistical summary (mean, std, best, worst, median).

        Args:
            results: List of :class:`Result` objects.

        Returns:
            Path to the saved CSV file.
        """
        fitness_vals = np.array([r.best_fitness for r in results])
        summary = {
            "algorithm": [results[0].algorithm_name],
            "problem": [results[0].problem_name],
            "runs": [len(results)],
            "mean": [np.mean(fitness_vals)],
            "std": [np.std(fitness_vals)],
            "best": [np.min(fitness_vals)],
            "worst": [np.max(fitness_vals)],
            "median": [np.median(fitness_vals)],
        }
        df = pd.DataFrame(summary)
        path = self.exp_dir / "summary.csv"
        df.to_csv(path, index=False)
        return path

    # ------------------------------------------------------------------
    # Convergence histories
    # ------------------------------------------------------------------
    def save_convergence_csv(self, results: List[Result]) -> Path:
        """Save convergence histories (one column per run) to CSV.

        Args:
            results: List of :class:`Result` objects.

        Returns:
            Path to the saved CSV file.
        """
        data = {}
        for res in results:
            col = f"run_{res.run_id}"
            data[col] = res.convergence_curve
        df = pd.DataFrame(data)
        path = self.exp_dir / "convergence.csv"
        df.to_csv(path, index=False)
        return path

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    def save_plot(self, fig, name: str, dpi: int = 150) -> Path:
        """Save a matplotlib figure to the figures sub-directory.

        Args:
            fig: A :class:`matplotlib.figure.Figure` instance.
            name: Filename (without extension).
            dpi: Resolution.

        Returns:
            Path to the saved PNG file.
        """
        path = self.figures_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        return path

    def save_json(self, data: Any, name: str) -> Path:
        """Save arbitrary JSON-serialisable data.

        Args:
            data: JSON-serialisable Python object.
            name: Filename (without extension).

        Returns:
            Path to the saved JSON file.
        """
        path = self.exp_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
        return path
