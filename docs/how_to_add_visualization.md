# How to Add a New Visualisation

This guide shows how to add a custom plot type to ATLAS.

---

## Step 1: Create a new file

Create a new Python file in `atlas/visualization/`. For example:

```
atlas/visualization/my_plot.py
```

## Step 2: Implement the plotting function

Follow this template:

```python
"""My custom visualisation."""

from __future__ import annotations
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

from atlas.core.result import Result


def plot_my_visualisation(
    results_dict: Dict[str, List[Result]],
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
    dpi: int = 150,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Short description of the plot.

    Args:
        results_dict: ``{algo_name: [Result, ...]}``.
        title: Plot title.
        figsize: Figure size in inches.
        dpi: Resolution.
        save_path: If provided, save figure to this path.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # --- Your plotting logic here ---
    for algo_name, results in results_dict.items():
        vals = [r.best_fitness for r in results]
        ax.bar(algo_name, np.mean(vals))

    ax.set_title(title or "My Custom Plot")
    ax.set_ylabel("Fitness")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig
```

**Style guidelines:**

- Always accept `save_path`, `dpi`, and `figsize` parameters.
- Return the `Figure` object so the caller can decide whether to show or
  save it.
- Use `fig.tight_layout()` before saving.
- Use a clean style (the `PlotManager` sets the global style for you).

## Step 3: Register in `plot_manager.py`

Open `atlas/visualization/plot_manager.py` and add a new method:

```python
def plot_my_visualisation(
    self,
    results_dict: Dict[str, List[Result]],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 150,
    figsize: tuple = (10, 6),
) -> Optional[plt.Figure]:
    from atlas.visualization.my_plot import plot_my_visualisation
    return plot_my_visualisation(
        results_dict, title=title, figsize=figsize, dpi=dpi, save_path=save_path,
    )
```

## Step 4: Export from `__init__.py`

Add your function to `atlas/visualization/__init__.py`:

```python
from atlas.visualization.my_plot import plot_my_visualisation
```

---

## Usage

```python
from atlas.visualization.plot_manager import PlotManager

pm = PlotManager()
fig = pm.plot_my_visualisation(results_dict, save_path="output.png")
```
