# How to Add a New Algorithm

This guide walks you through adding a custom metaheuristic to ATLAS.
Follow the 5 steps below.

---

## Step 1: Create a new file

Create a new Python file in `atlas/algorithms/`. For example:

```
atlas/algorithms/my_algo.py
```

## Step 2: Copy the template

Paste the following skeleton into your new file:

```python
"""My custom algorithm.

Reference:
    Author (Year). Paper title. Journal.
"""

from __future__ import annotations
from typing import Any
import numpy as np

from atlas.core.base_algorithm import BaseAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_algorithm


@register_algorithm("my_algo")
class MyAlgo(BaseAlgorithm):
    """Short description of the algorithm.

    Args:
        problem: Optimisation problem instance.
        max_iter: Maximum iterations.
        pop_size: Population size.
        seed: Random seed.
        verbose: Print progress.
        # Add your custom parameters here
    """

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 500,
        pop_size: int = 30,
        seed: int = 42,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(problem, max_iter, pop_size, seed, verbose, **kwargs)
        # Extract custom parameters from kwargs with defaults
        # self.my_param = kwargs.get("my_param", 0.5)

    def get_name(self) -> str:
        return "MyAlgo"

    def initialize(self) -> None:
        """Set up the initial population and evaluate fitness.

        You must set:
        - self.population: shape (pop_size, dim)
        - self.fitness:    shape (pop_size,)
        - self.g_best_x:   best solution found
        - self.g_best_f:   best fitness found
        """
        self.population = self.rng.uniform(
            self.lb, self.ub, size=(self.pop_size, self.dim)
        )
        self.fitness = np.array([
            self.problem.evaluate_with_penalty(self.population[i])
            for i in range(self.pop_size)
        ])
        self._update_global_best()

    def iterate(self, iter_idx: int) -> float:
        """Perform one iteration.

        Args:
            iter_idx: Current iteration index (0-based).

        Returns:
            Best fitness found so far.
        """
        # --- Your algorithm logic here ---
        # For each individual, generate a new candidate, evaluate it,
        # and decide whether to replace the current solution.

        # Example: random perturbation (replace with real logic)
        for i in range(self.pop_size):
            candidate = self.population[i] + self.rng.normal(0, 0.1, self.dim)
            candidate = self._clip(candidate)
            f = self.problem.evaluate_with_penalty(candidate)
            if f < self.fitness[i]:
                self.population[i] = candidate
                self.fitness[i] = f

        self._update_global_best()
        return self.g_best_f
```

## Step 3: Implement `initialize()` and `iterate()`

**`initialize()`** – called once before the iteration loop:
- Create an initial population of shape `(pop_size, dim)`.
- Evaluate each individual's fitness.
- Set `self.g_best_x` and `self.g_best_f` using `self._update_global_best()`.

**`iterate(iter_idx)`** – called once per iteration:
- Implement your algorithm's main logic.
- Update `self.population` and `self.fitness`.
- Call `self._update_global_best()` at the end.
- Return `self.g_best_f`.

**Available helpers on `self`:**

| Attribute / Method | Description |
|---|---|
| `self.rng` | `np.random.Generator` seeded with `self.seed` |
| `self.lb`, `self.ub` | Lower/upper bounds (1-D arrays) |
| `self.dim` | Problem dimensionality |
| `self.pop_size` | Population size |
| `self.max_iter` | Maximum iterations |
| `self.problem.evaluate_with_penalty(x)` | Evaluate a solution |
| `self._clip(x)` | Clip solution to bounds |
| `self._update_global_best()` | Scan fitness array for best |
| `self.g_best_x`, `self.g_best_f` | Current global best |

## Step 4: Register the algorithm

The `@register_algorithm("my_algo")` decorator in Step 2 already handles
registration. Just make sure the name is unique (check with
`atlas.list_algorithms()`).

## Step 5: Add the import

Open `atlas/algorithms/__init__.py` and add one line:

```python
from atlas.algorithms.my_algo import MyAlgo
```

This triggers auto-registration when the package is imported.

---

## Verify

```bash
python -c "import atlas; print(atlas.list_algorithms())"
```

You should see `'my_algo'` in the list.
