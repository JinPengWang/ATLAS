# How to Add a New Benchmark Function

This guide shows how to add a new test function to ATLAS.

---

## Step 1: Create or edit a file

Add your function to an existing file in `atlas/problems/benchmark/` (e.g.
`unimodal.py` or `multimodal.py`), or create a new `.py` file in that
directory.

## Step 2: Copy the template

```python
"""My benchmark function.

Reference:
    Author (Year). Paper title. Journal.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_problem


@register_problem("my_func")
class MyFunc(BaseProblem):
    """Short description.

    f(x) = ...

    * Type: Unimodal / Multimodal, Separable / Non-separable
    * Default dim: 30
    * Bounds: [lb, ub]^d
    * Global optimum: f(x*) = ... at x* = ...
    """

    def __init__(self, dim: int = 30, **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        self._lb = np.full(dim, -100.0)   # adjust per dimension
        self._ub = np.full(dim, 100.0)

    def evaluate(self, x: np.ndarray) -> float:
        # Implement the mathematical formula here
        return float(np.sum(x ** 2))

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lb.copy(), self._ub.copy()

    def get_name(self) -> str:
        return "my_func"

    def get_optimum(self) -> Optional[float]:
        return 0.0

    def get_optimum_location(self) -> Optional[np.ndarray]:
        return np.zeros(self.dim)
```

## Step 3: Implement the required methods

| Method | Purpose |
|---|---|
| `evaluate(x)` | The objective function. Receives a 1-D array, returns a float. |
| `get_bounds()` | Returns `(lb, ub)` as numpy arrays. |
| `get_name()` | A short lowercase string identifier (e.g. `"my_func"`). |
| `get_optimum()` | Known global optimum value, or `None`. |
| `get_optimum_location()` | Known global optimum position, or `None`. |

## Step 4: Register and import

The `@register_problem("my_func")` decorator handles registration.

If you created a **new file**, add an import in the appropriate
`__init__.py`:

- `atlas/problems/benchmark/__init__.py` for benchmark functions
- `atlas/problems/__init__.py` if it is a top-level problem

```python
from atlas.problems.benchmark.my_module import MyFunc
```

---

## Verify

```bash
python -c "import atlas; print(atlas.list_problems())"
```

You should see `'my_func'` in the list.
