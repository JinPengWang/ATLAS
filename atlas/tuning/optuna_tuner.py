"""Automated hyperparameter tuner for ATLAS algorithms using Optuna."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from atlas.core.base_problem import BaseProblem
from atlas.tuning.space import HyperparameterSpace
from atlas.utils.registry import get_algorithm, get_problem


@dataclass
class TuningResult:
    """Outcome of an automated hyperparameter tuning experiment."""

    best_params: Dict[str, Any]
    best_value: float
    study: Any  # optuna.Study
    n_trials: int
    algorithm_name: str
    problem_names: List[str]


class AlgorithmTuner:
    """Automated Hyperparameter Tuner for metaheuristic algorithms.

    Uses Optuna's Bayesian optimization (TPE / CMA-ES / Random) to search
    the optimal hyperparameter configurations for an algorithm on a given
    problem or across a suite of problems.

    Args:
        algorithm: Name or class of the algorithm to tune.
        space: HyperparameterSpace instance defining the search space.
        problems: A single BaseProblem or list of BaseProblems / problem names.
        n_trials: Number of tuning trials to perform.
        n_runs_per_trial: Number of repeated runs per configuration (reduces stochastic noise).
        max_iter: Max iterations for algorithm runs during evaluation.
        pop_size: Population size (can also be included in space to tune dynamically).
        sampler: Optuna sampler name ('tpe', 'random', 'cmaes'). Default is 'tpe'.
        seed: Random seed for reproducible tuning.
        timeout: Maximum duration in seconds for tuning.
    """

    def __init__(
        self,
        algorithm: Union[str, type],
        space: HyperparameterSpace,
        problems: Union[BaseProblem, str, List[Union[BaseProblem, str]]],
        n_trials: int = 50,
        n_runs_per_trial: int = 3,
        max_iter: int = 200,
        pop_size: int = 30,
        sampler: str = "tpe",
        seed: int = 42,
        timeout: Optional[float] = None,
    ) -> None:
        self.algo_cls = get_algorithm(algorithm) if isinstance(algorithm, str) else algorithm
        self.algo_name = algorithm if isinstance(algorithm, str) else self.algo_cls.__name__
        self.space = space

        # Parse problems
        if not isinstance(problems, list):
            problems = [problems]
        self.problem_instances: List[BaseProblem] = []
        self.problem_names: List[str] = []
        for p in problems:
            if isinstance(p, str):
                inst = get_problem(p)()
                self.problem_instances.append(inst)
                self.problem_names.append(p)
            else:
                self.problem_instances.append(p)
                self.problem_names.append(p.get_name())

        self.n_trials = n_trials
        self.n_runs_per_trial = max(1, n_runs_per_trial)
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.sampler_type = sampler.lower()
        self.seed = seed
        self.timeout = timeout

    def _create_study(self) -> Any:
        try:
            import optuna
        except ImportError as e:
            raise ImportError(
                "Optuna is required for automated hyperparameter tuning. "
                "Install with `pip install optuna`."
            ) from e

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if self.sampler_type == "random":
            sampler = optuna.samplers.RandomSampler(seed=self.seed)
        elif self.sampler_type == "cmaes":
            sampler = optuna.samplers.CmaEsSampler(seed=self.seed)
        else:  # default 'tpe'
            sampler = optuna.samplers.TPESampler(seed=self.seed)

        return optuna.create_study(direction="minimize", sampler=sampler)

    def tune(self, show_progress_bar: bool = False) -> TuningResult:
        """Run the hyperparameter tuning study.

        Returns:
            :class:`TuningResult` object containing best parameters and statistics.
        """
        study = self._create_study()

        def objective(trial: Any) -> float:
            sampled_params = self.space.sample_trial(trial)
            scores = []

            for prob in self.problem_instances:
                for run_id in range(self.n_runs_per_trial):
                    algo = self.algo_cls(
                        problem=prob,
                        max_iter=self.max_iter,
                        pop_size=self.pop_size,
                        seed=self.seed + run_id * 1000 + trial.number,
                        **sampled_params,
                    )
                    res = algo.run(run_id=run_id)
                    scores.append(res.best_fitness)

            return float(np.mean(scores))

        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=show_progress_bar,
        )

        return TuningResult(
            best_params=dict(study.best_params),
            best_value=float(study.best_value),
            study=study,
            n_trials=len(study.trials),
            algorithm_name=self.algo_name,
            problem_names=self.problem_names,
        )

    def plot_optimization_history(
        self,
        result: TuningResult,
        figsize: tuple = (8, 5),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot the objective value history over trials."""
        fig, ax = plt.subplots(figsize=figsize)
        trial_numbers = [t.number for t in result.study.trials if t.value is not None]
        trial_values = [t.value for t in result.study.trials if t.value is not None]

        best_so_far = []
        current_best = float("inf")
        for v in trial_values:
            current_best = min(current_best, v)
            best_so_far.append(current_best)

        ax.scatter(trial_numbers, trial_values, color="royalblue", alpha=0.6, label="Trial Value")
        ax.plot(trial_numbers, best_so_far, color="crimson", linewidth=2.0, label="Best Value")

        ax.set_xlabel("Trial")
        ax.set_ylabel("Mean Objective Value")
        ax.set_title(f"Tuning History — {result.algorithm_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig
