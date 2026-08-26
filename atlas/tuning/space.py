"""Hyperparameter search space definitions for automated tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Union


@dataclass
class Hyperparameter:
    """Definition of a single tunable hyperparameter."""

    name: str
    type: str  # "float", "int", "categorical"
    low: Optional[Union[float, int]] = None
    high: Optional[Union[float, int]] = None
    step: Optional[Union[float, int]] = None
    log: bool = False
    choices: Optional[Sequence[Any]] = None

    def suggest(self, trial: Any) -> Any:
        """Sample a value from an Optuna trial."""
        if self.type == "float":
            return trial.suggest_float(
                self.name,
                float(self.low),
                float(self.high),
                step=self.step,
                log=self.log,
            )
        elif self.type == "int":
            return trial.suggest_int(
                self.name,
                int(self.low),
                int(self.high),
                step=int(self.step) if self.step else 1,
                log=self.log,
            )
        elif self.type == "categorical":
            if self.choices is None:
                raise ValueError(f"Choices must be provided for categorical parameter '{self.name}'")
            return trial.suggest_categorical(self.name, list(self.choices))
        else:
            raise ValueError(f"Unknown hyperparameter type '{self.type}' for parameter '{self.name}'")


@dataclass
class HyperparameterSpace:
    """Container for multiple hyperparameter definitions."""

    parameters: List[Hyperparameter] = field(default_factory=list)

    def add_float(
        self,
        name: str,
        low: float,
        high: float,
        step: Optional[float] = None,
        log: bool = False,
    ) -> HyperparameterSpace:
        """Add a continuous float hyperparameter."""
        self.parameters.append(
            Hyperparameter(name=name, type="float", low=low, high=high, step=step, log=log)
        )
        return self

    def add_int(
        self,
        name: str,
        low: int,
        high: int,
        step: Optional[int] = None,
        log: bool = False,
    ) -> HyperparameterSpace:
        """Add an integer hyperparameter."""
        self.parameters.append(
            Hyperparameter(name=name, type="int", low=low, high=high, step=step, log=log)
        )
        return self

    def add_categorical(self, name: str, choices: Sequence[Any]) -> HyperparameterSpace:
        """Add a categorical hyperparameter."""
        self.parameters.append(
            Hyperparameter(name=name, type="categorical", choices=list(choices))
        )
        return self

    def sample_trial(self, trial: Any) -> dict[str, Any]:
        """Sample all hyperparameters from an Optuna trial."""
        return {p.name: p.suggest(trial) for p in self.parameters}

    def __len__(self) -> int:
        return len(self.parameters)
