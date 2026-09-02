"""Visualization tools for multi-objective optimization Pareto fronts."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import matplotlib
matplotlib.rcParams["mathtext.default"] = "regular"  # avoid bold math
import matplotlib.pyplot as plt
import numpy as np


def plot_pareto_front_2d(
    front: np.ndarray,
    true_pf: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    labels: Tuple[str, str] = ("$f_1$", "$f_2$"),
    figsize: Tuple[float, float] = (8, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a 2-D Pareto front with optional true Pareto front reference curve.

    Args:
        front: 2-D array of shape (N, 2) representing obtained non-dominated solutions.
        true_pf: Optional 2-D array of shape (K, 2) representing the true Pareto front.
        title: Figure title.
        labels: Tuple of (x-label, y-label).
        figsize: Figure size in inches.
        save_path: Optional path to save figure.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if true_pf is not None and len(true_pf) > 0:
        sorted_pf = true_pf[np.argsort(true_pf[:, 0])]
        ax.plot(
            sorted_pf[:, 0],
            sorted_pf[:, 1],
            "r--",
            linewidth=1.8,
            alpha=0.8,
            label="True Pareto Front",
        )

    if len(front) > 0:
        ax.scatter(
            front[:, 0],
            front[:, 1],
            c="royalblue",
            edgecolors="navy",
            s=40,
            alpha=0.85,
            label="Obtained Front",
            zorder=5,
        )

    ax.set_xlabel(labels[0], fontsize=11)
    ax.set_ylabel(labels[1], fontsize=11)
    ax.set_title(title or "Pareto Front Approximation", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_pareto_front_3d(
    front: np.ndarray,
    title: Optional[str] = None,
    labels: Tuple[str, str, str] = ("$f_1$", "$f_2$", "$f_3$"),
    figsize: Tuple[float, float] = (9, 7),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a 3-D Pareto front scatter plot."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if len(front) > 0:
        sc = ax.scatter(
            front[:, 0],
            front[:, 1],
            front[:, 2],
            c=front[:, 2],
            cmap="viridis",
            s=35,
            edgecolors="black",
            linewidths=0.3,
            alpha=0.85,
        )
        fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label=labels[2])

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(title or "3D Pareto Front Approximation")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_mo_comparison(
    fronts_dict: Dict[str, np.ndarray],
    true_pf: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    labels: Tuple[str, str] = ("$f_1$", "$f_2$"),
    figsize: Tuple[float, float] = (9, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Compare multiple multi-objective algorithms on the same 2-D Pareto plot."""
    fig, ax = plt.subplots(figsize=figsize)

    if true_pf is not None and len(true_pf) > 0:
        sorted_pf = true_pf[np.argsort(true_pf[:, 0])]
        ax.plot(
            sorted_pf[:, 0],
            sorted_pf[:, 1],
            "k--",
            linewidth=2.0,
            alpha=0.7,
            label="True Pareto Front",
        )

    markers = ["o", "s", "^", "D", "v", "p", "*"]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(fronts_dict), 10)))

    for i, (name, front) in enumerate(fronts_dict.items()):
        if len(front) > 0:
            m = markers[i % len(markers)]
            c = colors[i % len(colors)]
            ax.scatter(
                front[:, 0],
                front[:, 1],
                color=c,
                marker=m,
                s=35,
                alpha=0.8,
                label=name,
            )

    ax.set_xlabel(labels[0], fontsize=11)
    ax.set_ylabel(labels[1], fontsize=11)
    ax.set_title(title or "Multi-Objective Algorithm Comparison", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig
