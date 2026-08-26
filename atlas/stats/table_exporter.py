"""Academic table exporter for LaTeX, Markdown, and CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


def export_latex_table(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    caption: str = "Optimization performance comparison (Mean $\\pm$ Std)",
    label: str = "tab:benchmark_results",
    highlight_best: bool = True,
    higher_is_better: bool = False,
    float_format: str = "{:.2e}",
) -> str:
    """Export summary results DataFrame to academic LaTeX table format.

    Args:
        df: Aggregated results DataFrame. Expected to contain columns:
            problem (or problem_name), algorithm (or algorithm_name), mean, std.
            Or a pivot table with problems as rows, algorithms as columns (Mean ± Std strings).
        output_path: Optional path to save .tex file.
        caption: Table caption.
        label: LaTeX label.
        highlight_best: Whether to wrap the best result in \\textbf{...}.
        higher_is_better: Whether higher values mean better performance.
        float_format: Format string for floating numbers.

    Returns:
        LaTeX formatted string.
    """
    # Normalize DataFrame
    df_clean = df.copy()

    # If df is raw summary with problem, algorithm, mean, std
    if "mean" in df_clean.columns and "std" in df_clean.columns:
        prob_col = "problem" if "problem" in df_clean.columns else "problem_name"
        algo_col = "algorithm" if "algorithm" in df_clean.columns else "algorithm_name"
        dim_col = "dim" if "dim" in df_clean.columns else None

        # Pivot to problem x algorithm
        group_cols = [prob_col] + ([dim_col] if dim_col else [])
        
        # Build formatted cells and find best
        algos = df_clean[algo_col].unique()
        problems = df_clean[group_cols].drop_duplicates()
        
        rows = []
        for _, prob_row in problems.iterrows():
            row_dict = dict(prob_row)
            sub = df_clean
            for col in group_cols:
                sub = sub[sub[col] == prob_row[col]]
            
            # Find best mean in this row
            means = sub.set_index(algo_col)["mean"]
            best_val = means.max() if higher_is_better else means.min()
            
            for algo in algos:
                match = sub[sub[algo_col] == algo]
                if not match.empty:
                    m = match["mean"].values[0]
                    s = match["std"].values[0]
                    cell_str = f"{float_format.format(m)} $\\pm$ {float_format.format(s)}"
                    if highlight_best and np.isclose(m, best_val):
                        cell_str = f"\\textbf{{{cell_str}}}"
                    row_dict[algo] = cell_str
                else:
                    row_dict[algo] = "N/A"
            rows.append(row_dict)
        table_df = pd.DataFrame(rows)
    else:
        table_df = df_clean

    # Generate LaTeX
    num_cols = len(table_df.columns)
    col_align = "l" + "c" * (num_cols - 1)

    latex_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_align}}}",
        "\\hline\\hline",
        " & ".join(str(c).replace("_", "\\_") for c in table_df.columns) + " \\\\",
        "\\hline",
    ]

    for _, row in table_df.iterrows():
        line = " & ".join(str(v) for v in row.values) + " \\\\"
        latex_lines.append(line)

    latex_lines.extend([
        "\\hline\\hline",
        "\\end{tabular}%",
        "}",
        "\\end{table*}",
    ])

    latex_str = "\n".join(latex_lines)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(latex_str, encoding="utf-8")

    return latex_str


def export_markdown_table(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Export summary results DataFrame to Markdown table format.

    Args:
        df: Summary results DataFrame.
        output_path: Optional path to save .md file.

    Returns:
        Markdown table string.
    """
    md_str = df.to_markdown(index=False)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md_str, encoding="utf-8")

    return md_str


def export_csv_summary(
    df: pd.DataFrame,
    output_path: Union[str, Path],
) -> None:
    """Export summary DataFrame to CSV.

    Args:
        df: Summary results DataFrame.
        output_path: Target .csv file path.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
