# ATLAS – All-in-one Toolkit for Learning and Applying metaheuristicS

> A modular, extensible Python platform for metaheuristic algorithm research,
> benchmarking, and comparison.

---

## 1. Introduction

ATLAS is a research-oriented framework designed for students, researchers, and
practitioners who work with metaheuristic optimisation algorithms.  It provides:

- **6 built-in algorithms** (PSO, GA, DE, SA, WOA, ACO) with sensible defaults.
- **11 benchmark functions** (unimodal + multimodal) plus a custom problem
  template.
- **4 visualisation types** – convergence curves, box-plots, heatmaps, and
  fitness landscape plots.
- **Plugin architecture** – add a new algorithm, problem, or plot by creating a
  single file and one import line.
- **Full reproducibility** – every run is seeded; all parameters are logged.
- **One-command experiments** – run from the terminal or from Jupyter.

### Key Design Principles

| Principle | How ATLAS achieves it |
|---|---|
| Extensibility | Decorator-based auto-registration (`@register_algorithm`, `@register_problem`) |
| Ease of use | Single `Experiment` class orchestrates everything; `run_benchmark.py` provides a CLI |
| Reproducibility | Per-run seeds, config logging, deterministic NumPy RNG |
| Comparison | Multi-algorithm × multi-problem experiments with automatic heatmap generation |

---

## 2. Quick Start

### 2.1 Installation

```bash
# Clone or download the project
cd E:\A_Works\workspace\ATLAS

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# (Optional) Install ATLAS in development mode
pip install -e .
```

### 2.2 Run Your First Experiment

```bash
# Run PSO on the Sphere function (30D, 500 iterations, 30 runs)
python experiments/run_benchmark.py --algorithms pso --problems sphere --dim 30 --max_iter 500 --runs 30

# Compare three algorithms on two problems
python experiments/run_benchmark.py --algorithms pso ga de --problems sphere rastrigin --dim 30 --runs 10

# Full comparison of all six algorithms
python experiments/compare_algorithms.py --runs 20
```

### 2.3 Run from Jupyter

```python
import sys; sys.path.insert(0, r"E:\A_Works\workspace\ATLAS")
from experiments.run_benchmark import run_benchmark

results = run_benchmark(
    algorithms=["pso", "de", "woa"],
    problems=["sphere", "ackley"],
    dims=[20],
    max_iter=300,
    runs=10,
    seed=42,
)
```

---

## 3. Project Structure

```
ATLAS/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── setup.py                         # Optional pip-installable package
├── config/
│   └── default_config.yaml          # Documented default experiment config
├── atlas/
│   ├── __init__.py                  # Top-level imports & auto-registration
│   ├── core/
│   │   ├── base_algorithm.py        # Abstract base class for algorithms
│   │   ├── base_problem.py          # Abstract base class for problems
│   │   ├── experiment.py            # Experiment scheduler & manager
│   │   └── result.py                # Result data structure
│   ├── algorithms/
│   │   ├── __init__.py              # Imports all algorithms (triggers registration)
│   │   ├── pso.py                   # Particle Swarm Optimisation
│   │   ├── ga.py                    # Genetic Algorithm (real-coded)
│   │   ├── de.py                    # Differential Evolution
│   │   ├── sa.py                    # Simulated Annealing
│   │   ├── woa.py                   # Whale Optimisation Algorithm
│   │   └── aco.py                   # Ant Colony Optimisation (continuous)
│   ├── problems/
│   │   ├── __init__.py              # Imports all problems (triggers registration)
│   │   ├── benchmark/
│   │   │   ├── unimodal.py          # Sphere, Rosenbrock, Schwefel222, Quartic
│   │   │   └── multimodal.py        # Rastrigin, Ackley, Griewank, Levy, Schwefel, Michalewicz
│   │   └── custom/
│   │       └── example_custom.py    # Pressure Vessel Design (engineering problem)
│   ├── visualization/
│   │   ├── __init__.py              # Exports all plot functions
│   │   ├── convergence.py           # Convergence curves (mean ± std)
│   │   ├── boxplot.py               # Standard box-plots
│   │   ├── heatmap.py               # Algorithm × Problem performance heatmap
│   │   ├── fitness_landscape.py     # 2-D contour/surface + trajectory overlay
│   │   └── plot_manager.py          # Unified PlotManager class
│   └── utils/
│       ├── __init__.py              # Exports registry, saver, logger
│       ├── registry.py              # Auto-registration decorators
│       ├── saver.py                 # ExperimentSaver (CSV, YAML, figures)
│       └── logger.py                # Logging helpers
├── experiments/
│   ├── run_benchmark.py             # CLI + function for benchmark experiments
│   ├── run_custom.py                # Demo: custom problem (Pressure Vessel)
│   └── compare_algorithms.py        # Multi-algorithm comparison script
├── results/                         # Auto-generated output directory
│   └── .gitkeep
├── docs/
│   ├── how_to_add_algorithm.md      # Step-by-step guide
│   ├── how_to_add_benchmark.md      # Step-by-step guide
│   └── how_to_add_visualization.md  # Step-by-step guide
└── tests/
    ├── test_algorithms.py           # Algorithm unit tests
    └── test_problems.py             # Problem unit tests
```

---

## 4. Implemented Algorithms

| # | Name | Abbrev. | Key Parameters | Reference |
|---|---|---|---|---|
| 1 | Particle Swarm Optimisation | `pso` | `w=0.7, c1=1.5, c2=1.5` | Kennedy & Eberhart (1995) |
| 2 | Genetic Algorithm (real-coded) | `ga` | `crossover_prob=0.8, mutation_prob=0.01` | Holland (1992) |
| 3 | Differential Evolution | `de` | `F=0.8, CR=0.9, strategy='rand/1/bin'` | Storn & Price (1997) |
| 4 | Simulated Annealing | `sa` | `T_init=1000, T_min=1e-3, alpha=0.95` | Kirkpatrick et al. (1983) |
| 5 | Whale Optimisation Algorithm | `woa` | `b=1.0` | Mirjalili & Lewis (2016) |
| 6 | Ant Colony (continuous) | `aco` | `n_ants=20, q=0.1, xi=0.85` | Socha & Dorigo (2008) |

All parameters can be overridden via `algo_params` in the config or
`--algo_params` when using the CLI.

---

## 5. Implemented Benchmark Functions

### Unimodal

| Function | Dim | Bounds | Optimum | Characteristics |
|---|---|---|---|---|
| Sphere | 30 | [-100, 100] | 0 | Unimodal, separable |
| Rosenbrock | 30 | [-30, 30] | 0 | Unimodal, non-separable |
| Schwefel 2.22 | 30 | [-10, 10] | 0 | Unimodal, separable |
| Quartic | 30 | [-1.28, 1.28] | 0 | Unimodal, separable, noisy |

### Multimodal

| Function | Dim | Bounds | Optimum | Characteristics |
|---|---|---|---|---|
| Rastrigin | 30 | [-5.12, 5.12] | 0 | Multimodal, separable |
| Ackley | 30 | [-32.768, 32.768] | 0 | Multimodal, non-separable |
| Griewank | 30 | [-600, 600] | 0 | Multimodal, non-separable |
| Levy | 30 | [-10, 10] | 0 | Multimodal, non-separable |
| Schwefel | 30 | [-500, 500] | 0 | Multimodal, separable |
| Michalewicz | 30 | [0, π] | varies | Multimodal, separable |

### Engineering

| Problem | Dim | Bounds | Optimum |
|---|---|---|---|
| Pressure Vessel | 4 | see source | ≈ 6059.71 |

---

## 6. Visualisation

| Plot Type | Function | Description |
|---|---|---|
| Convergence Curve | `plot_convergence_comparison` | Mean ± std fitness over iterations with semi-transparent confidence band |
| Box-Plot | `plot_boxplot` | Standard box-plot of final fitness across runs; mean shown as diamond |
| Heatmap | `plot_heatmap` | Algorithm × Problem matrix; cells show normalised mean fitness |
| Fitness Landscape | `plot_fitness_landscape` | 2-D contour or 3-D surface with optional algorithm trajectory overlay |

All plots accept `save_path`, `dpi`, and `figsize` parameters.
Use the `PlotManager` class for a unified interface.

---

## 7. How to Add a New Algorithm

> **Full guide**: [docs/how_to_add_algorithm.md](docs/how_to_add_algorithm.md)

### Step 1 – Create the file

Create `atlas/algorithms/my_algo.py`.

### Step 2 – Copy the template

```python
"""My custom algorithm."""
from __future__ import annotations
from typing import Any
import numpy as np
from atlas.core.base_algorithm import BaseAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_algorithm


@register_algorithm("my_algo")
class MyAlgo(BaseAlgorithm):
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
        # self.my_param = kwargs.get("my_param", 0.5)

    def get_name(self) -> str:
        return "MyAlgo"

    def initialize(self) -> None:
        self.population = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([
            self.problem.evaluate_with_penalty(self.population[i])
            for i in range(self.pop_size)
        ])
        self._update_global_best()

    def iterate(self, iter_idx: int) -> float:
        # Replace with your algorithm logic
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

### Step 3 – Implement `initialize()` and `iterate()`

**`initialize()`** sets up the initial population. **`iterate()`** advances
one generation.

**Key attributes on `self`:**

| Attribute | Type | Description |
|---|---|---|
| `self.rng` | `np.random.Generator` | Seeded RNG |
| `self.lb`, `self.ub` | `np.ndarray` | Search bounds |
| `self.dim` | `int` | Problem dimensionality |
| `self.population` | `np.ndarray` | Shape `(pop_size, dim)` |
| `self.fitness` | `np.ndarray` | Shape `(pop_size,)` |
| `self.g_best_x` | `np.ndarray` | Best solution found |
| `self.g_best_f` | `float` | Best fitness found |

**Helper methods:** `_clip(x)`, `_update_global_best()`,
`problem.evaluate_with_penalty(x)`.

### Step 4 – Register

The `@register_algorithm("my_algo")` decorator handles everything.

### Step 5 – Add the import

Open `atlas/algorithms/__init__.py` and add:

```python
from atlas.algorithms.my_algo import MyAlgo
```

### Verify

```bash
python -c "import atlas; print(atlas.list_algorithms())"
# Should include 'my_algo'
```

---

## 8. How to Add a New Benchmark Function

> **Full guide**: [docs/how_to_add_benchmark.md](docs/how_to_add_benchmark.md)

### Step 1 – Create or edit a file

Add your function to `atlas/problems/benchmark/` or create a new `.py` file.

### Step 2 – Copy the template

```python
@register_problem("my_func")
class MyFunc(BaseProblem):
    def __init__(self, dim: int = 30, **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        self._lb = np.full(dim, -100.0)
        self._ub = np.full(dim, 100.0)

    def evaluate(self, x: np.ndarray) -> float:
        return float(np.sum(x ** 2))

    def get_bounds(self):
        return self._lb.copy(), self._ub.copy()

    def get_name(self) -> str:
        return "my_func"

    def get_optimum(self):
        return 0.0

    def get_optimum_location(self):
        return np.zeros(self.dim)
```

### Step 3 – Implement the methods

- `evaluate(x)` – the objective function.
- `get_bounds()` – lower and upper bounds as numpy arrays.
- `get_name()` – a short lowercase identifier.
- `get_optimum()` / `get_optimum_location()` – known optimal values.

### Step 4 – Import

Add an import in `atlas/problems/benchmark/__init__.py` (or
`atlas/problems/__init__.py` for custom problems).

### Verify

```bash
python -c "import atlas; print(atlas.list_problems())"
```

---

## 9. How to Add a New Visualisation

> **Full guide**: [docs/how_to_add_visualization.md](docs/how_to_add_visualization.md)

1. Create `atlas/visualization/my_plot.py`.
2. Implement a function `plot_my_visualisation(results_dict, ...)` that
   returns a `matplotlib.Figure`.
3. Add a wrapper method in `PlotManager`.
4. Export from `atlas/visualization/__init__.py`.

---

## 10. Custom Problem Example

See `experiments/run_custom.py` for a complete example using the
Pressure Vessel Design problem.

```python
from experiments.run_benchmark import run_benchmark

results = run_benchmark(
    algorithms=["pso", "de", "ga"],
    problems=["pressure_vessel"],
    dims=[4],
    max_iter=300,
    runs=10,
    seed=42,
    algo_params={
        "pso": {"w": 0.6, "c1": 1.8, "c2": 1.8},
        "de":  {"F": 0.7, "CR": 0.9, "strategy": "best/1/bin"},
    },
)
```

To define your own problem, see `atlas/problems/custom/example_custom.py`
and [docs/how_to_add_benchmark.md](docs/how_to_add_benchmark.md).

---

## 11. Results Directory Structure

```
results/
└── {problem_name}/
    └── {YYYYMMDD_HHMMSS}_{exp_label}/
        ├── config.yaml              # Full experiment configuration
        ├── raw_results.csv          # Per-run best fitness and solution
        ├── convergence.csv          # Per-iteration best-so-far (one column per run)
        ├── summary.csv              # Statistical summary (mean, std, best, worst, median)
        ├── figures/
        │   ├── convergence_curve.png
        │   ├── boxplot.png
        │   └── heatmap.png
        └── logs/
            └── experiment.log
```

| File | Contents |
|---|---|
| `config.yaml` | Every parameter used in the experiment (YAML format) |
| `raw_results.csv` | One row per run: algorithm, problem, run_id, best_fitness, seed, x0…xN |
| `convergence.csv` | One column per run: best-so-far fitness at each iteration |
| `summary.csv` | Mean, std, min, max, median of final fitness across runs |
| `figures/` | Auto-generated plots (convergence, boxplot, heatmap) |
| `logs/` | Detailed execution log |

---

## 12. FAQ

**Q: How do I change the random seed?**
A: Use `--seed 123` on the CLI, or set `seed=123` in `ExperimentConfig`.
Run `i` uses `seed + i`.

**Q: How do I add constraints to a problem?**
A: Override `evaluate_with_penalty()` in your problem class. See
`atlas/problems/custom/example_custom.py` for an example.

**Q: Can I run only specific plots?**
A: Yes. Use `--plots convergence boxplot` (omit heatmap), or set
`enable_plots` in the config YAML.

**Q: How do I use a different population size?**
A: Set `--pop_size 100` (global override) or specify per-algorithm in
`algo_params`.

**Q: The algorithm I need is not included.**
A: Follow the guide in Section 7.  You only need to create one file and
add one import line.

**Q: How do I get non-log-scale box-plots?**
A: Call `plot_boxplot(..., log_scale=False)` directly, or modify the
`PlotManager` wrapper.

**Q: Can I use this for multi-objective problems?**
A: Not yet. ATLAS currently supports single-objective optimisation only.

---

## 13. Contributing & Code Style

1. **Format**: Use `black` (line length 88) and `isort`.
2. **Linting**: Run `ruff check atlas/` before committing.
3. **Type hints**: All public functions must have type annotations.
4. **Docstrings**: Google style with `Args`, `Returns`, `Raises` sections.
5. **Tests**: Add a test in `tests/` for any new algorithm or problem.
6. **Naming**: snake_case for functions/variables, PascalCase for classes.
7. **Dependencies**: Add any new package to `requirements.txt` with a
   version range.

### Running Tests

```bash
pytest tests/ -v
```

---

## License

This project is provided for educational and research purposes.
Feel free to modify and distribute.
