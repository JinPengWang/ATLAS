# ATLAS – All-in-one Toolkit for Learning and Applying metaheuristicS

> A modular, extensible Python platform for metaheuristic algorithm research,
> benchmarking, and comparison.

---

## 1. Introduction

ATLAS is a research-oriented framework designed for students, researchers, and
practitioners who work with metaheuristic optimisation algorithms. It provides:

- **12 built-in algorithms** – 6 iteration-based + 6 NFE-based variants (PSO, GA, DE, SA, WOA, ACO).
- **155 benchmark functions** – 11 classic functions + 144 CEC competition benchmarks (CEC 2005–2022).
- **Dual stopping criteria** – iteration-based or function-evaluation-based (NFE).
- **CEC suite grouping** – run an entire CEC suite with a single name (e.g. `--problems cec2017`).
- **4 visualisation types** – convergence curves, box-plots, heatmaps, and fitness landscape plots.
- **Plugin architecture** – add a new algorithm, problem, or plot by creating a single file and one import line.
- **Full reproducibility** – every run is seeded; all parameters are logged.
- **One-command experiments** – run from the terminal or from Jupyter.

### Key Design Principles

| Principle              | How ATLAS achieves it                                                                |
| ---------------------- | ------------------------------------------------------------------------------------ |
| Extensibility          | Decorator-based auto-registration (`@register_algorithm`, `@register_problem`)       |
| Dual stopping criteria | Iteration-based (`iteration/`) and NFE-based (`nfe/`) algorithm variants             |
| CEC benchmarks         | Full CEC 2005–2022 suite via `opfunu` with suite-level grouping                      |
| Ease of use            | Single `Experiment` class orchestrates everything; `run_benchmark.py` provides a CLI |
| Reproducibility        | Per-run seeds, config logging, deterministic NumPy RNG                               |
| Comparison             | Multi-algorithm × multi-problem experiments with automatic heatmap generation        |

---

## 2. Quick Start

### 2.1 Installation

```bash
# Clone or download the project
cd ATLAS

# Create a virtual environment (recommended)
conda create -n atlas python=3.8
conda activate atlas          # Windows
# source venv/bin/activate    # Linux/macOS

# Install dependencies (includes opfunu for CEC benchmarks)
pip install -r requirements.txt

# (Optional) Install ATLAS in development mode
pip install -e .

# (Optional) Install with CEC benchmark support only
pip install -e ".[cec]"
```

### 2.2 Run Your First Experiment

```bash
# Run PSO on the Sphere function (30D, 500 iterations, 30 runs)
python experiments/run_benchmark.py --algorithms pso --problems sphere --dim 30 --max_iter 500 --runs 30

# Compare three algorithms on two problems
python experiments/run_benchmark.py --algorithms pso ga de --problems sphere rastrigin --dim 30 --runs 10

# Full comparison of all six iteration-based algorithms
python experiments/compare_algorithms.py --runs 20
```

### 2.3 Comprehensive Example (All Parameters)

```bash
python experiments/run_benchmark.py \
    --algorithms pso ga de woa sa aco \
    --problems sphere rastrigin ackley cec2017_f1 cec2022_f1 \
    --dim 10 30 50 \
    --max_iter 1000 \
    --stopping_criterion iterations \
    --max_nfe 10000 \
    --runs 30 \
    --seed 42 \
    --pop_size 50 \
    --plots convergence boxplot heatmap \
    --save_results \
    --verbose \
    --base_dir results
```

### 2.3 NFE-Based Algorithms

Use NFE (Number of Function Evaluations) variants for fair comparison across algorithms with different population sizes:

```bash
# Run GA with NFE-based stopping (10000 function evaluations)
python experiments/run_benchmark.py --algorithms ga_nfe --problems sphere --dim 30 --max_nfe 10000 --runs 30

# Compare iteration-based vs NFE-based
python experiments/run_benchmark.py --algorithms ga ga_nfe --problems sphere --dim 30 --max_iter 500 --max_nfe 10000 --runs 10

# All NFE variants: pso_nfe, ga_nfe, de_nfe, sa_nfe, woa_nfe, aco_nfe
```

### 2.4 CEC Benchmark Suites

Run entire CEC competition suites with a single name:

```bash
# Run all CEC 2017 functions (F1-F29) with default dim=30
python experiments/run_benchmark.py --algorithms de --problems cec2017 --max_iter 500 --runs 30

# Test all supported dimensions for CEC 2017
python experiments/run_benchmark.py --algorithms de --problems cec2017 --dim all --max_iter 500 --runs 30

# Test specific dimensions
python experiments/run_benchmark.py --algorithms de --problems cec2017 --dim 10 30 50 100 --max_iter 500 --runs 30

# Mix suite and individual problems
python experiments/run_benchmark.py --algorithms pso --problems cec2017 sphere ackley --dim 30 --runs 10

# Multiple suites
python experiments/run_benchmark.py --algorithms ga --problems cec2017 cec2022 --dim 10 --max_iter 500

# NFE-based on CEC suite
python experiments/run_benchmark.py --algorithms de_nfe --problems cec2022 --dim 10 --max_nfe 10000 --runs 30
```

Available suite names and supported dimensions:

| Suite     | Functions   | Default Dim | Supported Dims                                |
| --------- | ----------- | ----------- | --------------------------------------------- |
| `cec2005` | F1–F25 (25) | 30          | 10, 30, 50                                    |
| `cec2013` | F1–F28 (28) | 30          | 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 |
| `cec2014` | F1–F30 (30) | 30          | 10, 20, 30, 50, 100                           |
| `cec2017` | F1–F29 (29) | 30          | 2, 10, 20, 30, 50, 100                        |
| `cec2019` | F1–F10 (10) | varies      | F1=9, F2=16, F3=18, F4-F10=10 (fixed)         |
| `cec2020` | F1–F10 (10) | 30          | 2, 5, 10, 15, 20, 30, 50, 100                 |
| `cec2022` | F1–F12 (12) | 10          | 2, 10, 20                                     |

### 2.5 Run from Jupyter

```python
import sys; sys.path.insert(0, r"E:\A_Works\workspace\ATLAS")
from experiments.run_benchmark import run_benchmark

# Classic benchmark
results = run_benchmark(
    algorithms=["pso", "de", "woa"],
    problems=["sphere", "ackley"],
    dims=[20],
    max_iter=300,
    runs=10,
    seed=42,
)

# CEC suite with NFE-based algorithms
results = run_benchmark(
    algorithms=["de_nfe", "ga_nfe"],
    problems=["cec2022"],
    dims=[10],
    max_nfe=10000,
    runs=30,
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
│   │   └── result.py                # Result data structure (with nfe field)
│   ├── algorithms/
│   │   ├── __init__.py              # Imports all algorithms (triggers registration)
│   │   ├── iteration/               # Iteration-based stopping
│   │   │   ├── __init__.py
│   │   │   ├── pso.py               # Particle Swarm Optimisation
│   │   │   ├── ga.py                # Genetic Algorithm (real-coded)
│   │   │   ├── de.py                # Differential Evolution
│   │   │   ├── sa.py                # Simulated Annealing
│   │   │   ├── woa.py               # Whale Optimisation Algorithm
│   │   │   └── aco.py               # Ant Colony Optimisation (continuous)
│   │   └── nfe/                     # NFE-based stopping
│   │       ├── __init__.py
│   │       ├── base_nfe_algorithm.py # Shared NFE base class with _evaluate() counter
│   │       ├── pso.py               # PSO (NFE-based)
│   │       ├── ga.py                # GA (NFE-based)
│   │       ├── de.py                # DE (NFE-based)
│   │       ├── sa.py                # SA (NFE-based)
│   │       ├── woa.py               # WOA (NFE-based)
│   │       └── aco.py               # ACO (NFE-based)
│   ├── problems/
│   │   ├── __init__.py              # Imports all problems (triggers registration)
│   │   ├── benchmark/
│   │   │   ├── unimodal.py          # Sphere, Rosenbrock, Schwefel222, Quartic
│   │   │   └── multimodal.py        # Rastrigin, Ackley, Griewank, Levy, Schwefel, Michalewicz
│   │   ├── cec/                     # CEC competition benchmarks (via opfunu)
│   │   │   ├── __init__.py          # Registers all CEC functions
│   │   │   ├── base_cec.py          # Generic CECProblem wrapper + make_cec_class()
│   │   │   ├── cec2005.py           # CEC 2005 (F1-F25)
│   │   │   ├── cec2013.py           # CEC 2013 (F1-F28)
│   │   │   ├── cec2014.py           # CEC 2014 (F1-F30)
│   │   │   ├── cec2017.py           # CEC 2017 (F1-F29)
│   │   │   ├── cec2019.py           # CEC 2019 (F1-F10)
│   │   │   ├── cec2020.py           # CEC 2020 (F1-F10)
│   │   │   └── cec2022.py           # CEC 2022 (F1-F12)
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
│       ├── logger.py                # Logging helpers
│       └── problem_groups.py        # CEC suite name expansion
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
    ├── test_algorithms.py           # Iteration-based algorithm unit tests
    ├── test_problems.py             # Problem unit tests
    ├── test_nfe_algorithms.py       # NFE-based algorithm unit tests
    └── test_cec_problems.py         # CEC benchmark unit tests
```

---

## 4. Implemented Algorithms

### Iteration-Based Algorithms

| #   | Name                           | Abbrev. | Key Parameters                           | Reference                 |
| --- | ------------------------------ | ------- | ---------------------------------------- | ------------------------- |
| 1   | Particle Swarm Optimisation    | `pso`   | `w=0.7, c1=1.5, c2=1.5`                  | Kennedy & Eberhart (1995) |
| 2   | Genetic Algorithm (real-coded) | `ga`    | `crossover_prob=0.8, mutation_prob=0.01` | Holland (1992)            |
| 3   | Differential Evolution         | `de`    | `F=0.8, CR=0.9, strategy='rand/1/bin'`   | Storn & Price (1997)      |
| 4   | Simulated Annealing            | `sa`    | `T_init=1000, T_min=1e-3, alpha=0.95`    | Kirkpatrick et al. (1983) |
| 5   | Whale Optimisation Algorithm   | `woa`   | `b=1.0`                                  | Mirjalili & Lewis (2016)  |
| 6   | Ant Colony (continuous)        | `aco`   | `n_ants=20, q=0.1, xi=0.85`              | Socha & Dorigo (2008)     |

### NFE-Based Algorithms

These variants stop after a fixed number of function evaluations (NFE) instead of iterations, enabling fair comparison across algorithms with different population sizes.

| #   | Name            | Abbrev.   | Key Parameters | Reference                 |
| --- | --------------- | --------- | -------------- | ------------------------- |
| 1   | PSO (NFE-based) | `pso_nfe` | Same as PSO    | Kennedy & Eberhart (1995) |
| 2   | GA (NFE-based)  | `ga_nfe`  | Same as GA     | Holland (1992)            |
| 3   | DE (NFE-based)  | `de_nfe`  | Same as DE     | Storn & Price (1997)      |
| 4   | SA (NFE-based)  | `sa_nfe`  | Same as SA     | Kirkpatrick et al. (1983) |
| 5   | WOA (NFE-based) | `woa_nfe` | Same as WOA    | Mirjalili & Lewis (2016)  |
| 6   | ACO (NFE-based) | `aco_nfe` | Same as ACO    | Socha & Dorigo (2008)     |

All parameters can be overridden via `algo_params` in the config or
`--algo_params` when using the CLI.

---

## 5. Implemented Benchmark Functions

### Classic Benchmarks (11 functions)

#### Unimodal

| Function      | Dim | Bounds        | Optimum | Characteristics            |
| ------------- | --- | ------------- | ------- | -------------------------- |
| Sphere        | 30  | [-100, 100]   | 0       | Unimodal, separable        |
| Rosenbrock    | 30  | [-30, 30]     | 0       | Unimodal, non-separable    |
| Schwefel 2.22 | 30  | [-10, 10]     | 0       | Unimodal, separable        |
| Quartic       | 30  | [-1.28, 1.28] | 0       | Unimodal, separable, noisy |

#### Multimodal

| Function    | Dim | Bounds            | Optimum | Characteristics           |
| ----------- | --- | ----------------- | ------- | ------------------------- |
| Rastrigin   | 30  | [-5.12, 5.12]     | 0       | Multimodal, separable     |
| Ackley      | 30  | [-32.768, 32.768] | 0       | Multimodal, non-separable |
| Griewank    | 30  | [-600, 600]       | 0       | Multimodal, non-separable |
| Levy        | 30  | [-10, 10]         | 0       | Multimodal, non-separable |
| Schwefel    | 30  | [-500, 500]       | 0       | Multimodal, separable     |
| Michalewicz | 30  | [0, π]            | varies  | Multimodal, separable     |

#### Engineering

| Problem         | Dim | Bounds     | Optimum   |
| --------------- | --- | ---------- | --------- |
| Pressure Vessel | 4   | see source | ≈ 6059.71 |

### CEC Competition Benchmarks (144 functions)

Requires `opfunu` package (`pip install opfunu`).

| Suite    | Functions   | Default Dim | Supported Dims                                | Description                             |
| -------- | ----------- | ----------- | --------------------------------------------- | --------------------------------------- |
| CEC 2005 | F1–F25 (25) | 30          | 10, 30, 50                                    | Shifted, rotated, hybrid functions      |
| CEC 2013 | F1–F28 (28) | 30          | 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 | Shifted, rotated, composition functions |
| CEC 2014 | F1–F30 (30) | 30          | 10, 20, 30, 50, 100                           | Shifted, rotated, composition functions |
| CEC 2017 | F1–F29 (29) | 30          | 2, 10, 20, 30, 50, 100                        | Shifted, rotated, hybrid, composition   |
| CEC 2019 | F1–F10 (10) | varies      | F1=9, F2=16, F3=18, F4-F10=10 (fixed)         | Fixed-dimension functions               |
| CEC 2020 | F1–F10 (10) | 30          | 2, 5, 10, 15, 20, 30, 50, 100                 | Shifted, rotated functions              |
| CEC 2022 | F1–F12 (12) | 10          | 2, 10, 20                                     | Shifted, rotated, hybrid functions      |

Individual functions are registered as `cec20XX_fY` (e.g. `cec2017_f1`). Use suite names (e.g. `cec2017`) to run all functions in a suite at once. Use `--dim all` to test all supported dimensions for a suite.

---

## 6. CLI Reference

### Main Arguments

| Argument               | Type | Default      | Description                                               |
| ---------------------- | ---- | ------------ | --------------------------------------------------------- |
| `--algorithms`         | str+ | `pso`        | Algorithm names (e.g. `pso ga de_nfe`)                    |
| `--problems`           | str+ | `sphere`     | Problem names or suite names (e.g. `sphere cec2017`)      |
| `--dim`                | str+ | `30`         | Dimension(s) per problem, or `all` for all supported dims |
| `--max_iter`           | int  | `500`        | Max iterations per run (iteration-based)                  |
| `--stopping_criterion` | str  | `iterations` | `iterations` or `nfe`                                     |
| `--max_nfe`            | int  | `10000`      | Max function evaluations (NFE-based)                      |
| `--runs`               | int  | `30`         | Number of independent runs                                |
| `--seed`               | int  | `42`         | Base random seed                                          |
| `--pop_size`           | int  | `None`       | Population size override                                  |
| `--plots`              | str+ | all          | Plot types: `convergence boxplot heatmap`                 |
| `--save_results`       | flag | `True`       | Save results to disk                                      |
| `--verbose`            | flag | `False`      | Print per-iteration info                                  |
| `--base_dir`           | str  | `results`    | Output directory                                          |

### Examples

```bash
# Classic benchmark
python experiments/run_benchmark.py --algorithms pso ga de --problems sphere rastrigin --dim 30 --runs 30

# NFE-based comparison
python experiments/run_benchmark.py --algorithms pso_nfe ga_nfe de_nfe --problems sphere --dim 30 --max_nfe 10000 --runs 30

# CEC suite with default dim (30)
python experiments/run_benchmark.py --algorithms de --problems cec2017 --max_iter 500 --runs 30

# CEC suite with all supported dimensions
python experiments/run_benchmark.py --algorithms de --problems cec2017 --dim all --max_iter 500 --runs 30

# CEC suite with specific dimensions
python experiments/run_benchmark.py --algorithms de --problems cec2017 --dim 10 30 50 100 --max_iter 500 --runs 30

# CEC suite with NFE-based stopping
python experiments/run_benchmark.py --algorithms de_nfe --problems cec2022 --dim 10 --max_nfe 10000 --runs 30

# Mixed problems
python experiments/run_benchmark.py --algorithms pso --problems sphere cec2017_f1 ackley --dim 30 --runs 10

# Full example with all parameters
python experiments/run_benchmark.py \
    --algorithms pso ga de woa sa aco \
    --problems sphere rastrigin ackley cec2017_f1 cec2022_f1 \
    --dim 10 30 50 \
    --max_iter 1000 \
    --stopping_criterion iterations \
    --max_nfe 10000 \
    --runs 30 \
    --seed 42 \
    --pop_size 50 \
    --plots convergence boxplot heatmap \
    --save_results \
    --verbose \
    --base_dir results
```

---

## 7. Visualisation

| Plot Type         | Function                      | Description                                                              |
| ----------------- | ----------------------------- | ------------------------------------------------------------------------ |
| Convergence Curve | `plot_convergence_comparison` | Mean ± std fitness over iterations with semi-transparent confidence band |
| Box-Plot          | `plot_boxplot`                | Standard box-plot of final fitness across runs; mean shown as diamond    |
| Heatmap           | `plot_heatmap`                | Algorithm × Problem matrix; cells show normalised mean fitness           |
| Fitness Landscape | `plot_fitness_landscape`      | 2-D contour or 3-D surface with optional algorithm trajectory overlay    |

All plots accept `save_path`, `dpi`, and `figsize` parameters.
Use the `PlotManager` class for a unified interface.

---

## 8. How to Add a New Algorithm

> **Full guide**: [docs/how_to_add_algorithm.md](docs/how_to_add_algorithm.md)

### Step 1 – Create the file

For iteration-based: `atlas/algorithms/iteration/my_algo.py`
For NFE-based: `atlas/algorithms/nfe/my_algo.py`

### Step 2 – Copy the template (iteration-based)

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

### Step 3 – For NFE-based variant

```python
"""My custom algorithm (NFE-based)."""
from atlas.algorithms.nfe.base_nfe_algorithm import BaseNFEAlgorithm
from atlas.utils.registry import register_algorithm


@register_algorithm("my_algo_nfe")
class MyAlgo_NFE(BaseNFEAlgorithm):
    def __init__(self, problem, max_iter=0, max_nfe=10000, pop_size=30, seed=42, verbose=False, **kwargs):
        super().__init__(problem, max_iter, max_nfe, pop_size, seed, verbose, **kwargs)

    def get_name(self) -> str:
        return "MyAlgo_NFE"

    def initialize(self) -> None:
        # Same as iteration-based, but use self._evaluate() instead of self.problem.evaluate_with_penalty()
        ...

    def iterate(self, iter_idx: int) -> float:
        # Use self._evaluate(x) to count function evaluations
        ...
```

### Step 4 – Register and import

The `@register_algorithm` decorator handles registration. Add the import in `atlas/algorithms/__init__.py`:

```python
from atlas.algorithms.iteration.my_algo import MyAlgo
from atlas.algorithms.nfe.my_algo import MyAlgo_NFE
```

### Verify

```bash
python -c "import atlas; print(atlas.list_algorithms())"
# Should include 'my_algo' and 'my_algo_nfe'
```

---

## 9. How to Add a New Benchmark Function

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

### Step 3 – Import

Add an import in `atlas/problems/benchmark/__init__.py` (or
`atlas/problems/__init__.py` for custom problems).

### Verify

```bash
python -c "import atlas; print(atlas.list_problems())"
```

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
    └── {algorithm_name}/
        ├── config.yaml              # Full experiment configuration
        ├── raw_results.csv          # Per-run best fitness, solution, nfe, iterations
        ├── convergence.csv          # Per-iteration best-so-far (one column per run)
        ├── summary.csv              # Statistical summary (mean, std, best, worst, median, mean_nfe)
        └── figures/
            ├── convergence_curve.png
            ├── boxplot.png
            └── heatmap.png
```

| File              | Contents                                                                                |
| ----------------- | --------------------------------------------------------------------------------------- |
| `config.yaml`     | Every parameter used in the experiment (YAML format)                                    |
| `raw_results.csv` | One row per run: algorithm, problem, run_id, best_fitness, seed, nfe, iterations, x0…xN |
| `convergence.csv` | One column per run: best-so-far fitness at each iteration                               |
| `summary.csv`     | Mean, std, min, max, median of final fitness + mean NFE across runs                     |
| `figures/`        | Auto-generated plots (convergence, boxplot, heatmap)                                    |

---

## 12. FAQ

**Q: How do I change the random seed?**
A: Use `--seed 123` on the CLI, or set `seed=123` in `ExperimentConfig`.
Run `i` uses `seed + i`.

**Q: What's the difference between iteration-based and NFE-based algorithms?**
A: Iteration-based algorithms (e.g. `pso`) stop after a fixed number of iterations. NFE-based algorithms (e.g. `pso_nfe`) stop after a fixed number of function evaluations, enabling fair comparison across algorithms with different population sizes.

**Q: How do I run an entire CEC suite?**
A: Use the suite name directly: `--problems cec2017`. This expands to all functions in the suite (e.g. `cec2017_f1` through `cec2017_f29`). Available suites: `cec2005`, `cec2013`, `cec2014`, `cec2017`, `cec2019`, `cec2020`, `cec2022`.

**Q: How do I test all supported dimensions for a CEC suite?**
A: Use `--dim all` (CLI) or `dims=[-1]` (Python API). This automatically detects all dimensions supported by the specified problems and tests each one. For example, `--problems cec2017 --dim all` will test dimensions 2, 10, 20, 30, 50, and 100.

**Q: Do I need `opfunu` for CEC benchmarks?**
A: Yes. Install it with `pip install opfunu` or `pip install -e ".[cec]"`. Without it, CEC problems won't be registered but the framework still works for classic benchmarks.

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
A: Follow the guide in Section 8. You only need to create one file and
add one import line.

**Q: How do I get non-log-scale box-plots?**
A: Call `plot_boxplot(..., log_scale=False)` directly, or modify the
`PlotManager` wrapper.

**Q: Can I use this for multi-objective problems?**
A: Not yet. ATLAS currently supports single-objective optimisation only.

---

## 13. Dependencies

| Package      | Version | Purpose                            |
| ------------ | ------- | ---------------------------------- |
| `numpy`      | >=1.24  | Numerical computing                |
| `matplotlib` | >=3.7   | Plotting                           |
| `pyyaml`     | >=6.0   | Config file parsing                |
| `tqdm`       | >=4.65  | Progress bars                      |
| `opfunu`     | >=1.0.0 | CEC benchmark functions (optional) |

Install all dependencies: `pip install -r requirements.txt`
Install with CEC support: `pip install -e ".[cec]"`

---

## 14. Contributing & Code Style

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
