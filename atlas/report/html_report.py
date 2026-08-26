"""Interactive HTML experiment report generator using Plotly.

Provides capabilities to export comprehensive, self-contained interactive
HTML reports for ATLAS benchmark and experiment results.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import plotly.graph_objects as go

from atlas.core.result import Result

# Standard aesthetic color palette for algorithms
PALETTE = [
    "#2563eb",  # Blue
    "#dc2626",  # Red
    "#16a34a",  # Green
    "#d97706",  # Amber
    "#9333ea",  # Purple
    "#0891b2",  # Cyan
    "#ea580c",  # Orange
    "#4f46e5",  # Indigo
    "#059669",  # Emerald
    "#c026d3",  # Fuchsia
]


def _hex_to_rgba(hex_code: str, alpha: float = 0.2) -> str:
    """Convert hex color code to RGBA string.

    Args:
        hex_code: Hex string (e.g. ``"#2563eb"``).
        alpha: Alpha channel transparency between 0.0 and 1.0.

    Returns:
        RGBA string representation (e.g. ``"rgba(37, 99, 235, 0.2)"``).
    """
    hex_code = hex_code.lstrip("#")
    if len(hex_code) == 6:
        r = int(hex_code[0:2], 16)
        g = int(hex_code[2:4], 16)
        b = int(hex_code[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    return f"rgba(100, 100, 100, {alpha})"


def _format_value(val: float) -> str:
    """Format floating point numbers cleanly for display.

    Args:
        val: Numeric value.

    Returns:
        Formatted string in scientific or standard decimal notation.
    """
    if np.isnan(val):
        return "N/A"
    if val == 0.0:
        return "0.0000"
    if abs(val) < 1e-3 or abs(val) >= 1e5:
        return f"{val:.4e}"
    return f"{val:.4f}"


def _build_summary_table_html(
    all_results: Dict[str, Dict[str, List[Result]]],
) -> str:
    """Build summary statistics HTML table.

    Computes problem, algorithm, mean, std, best, and worst fitness statistics.

    Args:
        all_results: Nested dictionary ``{problem: {algo: [Result, ...]}}``.

    Returns:
        HTML table string with styled table markup.
    """
    if not all_results:
        return "<p class='no-data'>No experiment results available to summarize.</p>"

    rows_html: List[str] = []

    for prob_name, algos_dict in all_results.items():
        if not algos_dict:
            continue

        # Find best mean across algorithms for this problem to highlight
        algo_means = {}
        for a_name, res_list in algos_dict.items():
            if res_list:
                fits = [r.best_fitness for r in res_list]
                algo_means[a_name] = np.mean(fits)

        best_mean_val = min(algo_means.values()) if algo_means else None

        for a_name, res_list in algos_dict.items():
            if not res_list:
                continue
            fits = np.array([r.best_fitness for r in res_list], dtype=float)
            mean_val = float(np.mean(fits))
            std_val = float(np.std(fits))
            best_val = float(np.min(fits))
            worst_val = float(np.max(fits))
            n_runs = len(fits)

            is_winner = best_mean_val is not None and np.isclose(mean_val, best_mean_val)
            row_class = " class='highlight-row'" if is_winner else ""
            winner_badge = " <span class='badge-best'>Best</span>" if is_winner and len(algos_dict) > 1 else ""

            rows_html.append(
                f"<tr{row_class}>"
                f"<td><strong>{prob_name}</strong></td>"
                f"<td><span class='algo-pill'>{a_name}</span>{winner_badge}</td>"
                f"<td>{n_runs}</td>"
                f"<td class='num-cell'><strong>{_format_value(mean_val)}</strong></td>"
                f"<td class='num-cell'>&plusmn; {_format_value(std_val)}</td>"
                f"<td class='num-cell text-success'>{_format_value(best_val)}</td>"
                f"<td class='num-cell text-danger'>{_format_value(worst_val)}</td>"
                f"</tr>"
            )

    if not rows_html:
        return "<p class='no-data'>No valid run results found.</p>"

    table_html = f"""
    <div class="table-responsive">
        <table class="report-table">
            <thead>
                <tr>
                    <th>Problem</th>
                    <th>Algorithm</th>
                    <th>Runs</th>
                    <th>Mean</th>
                    <th>Std</th>
                    <th>Best</th>
                    <th>Worst</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
    """
    return table_html


def _build_convergence_figure(
    prob_name: str,
    algos_dict: Dict[str, List[Result]],
    colors: Sequence[str],
) -> Optional[go.Figure]:
    """Generate interactive Plotly convergence curve for a problem.

    Plots mean curve with std band for each algorithm, with y-axis on log scale.

    Args:
        prob_name: Name of the optimization problem.
        algos_dict: Dict mapping algorithm names to lists of Result objects.
        colors: Color palette sequence.

    Returns:
        Plotly Figure object or None if no valid convergence curves exist.
    """
    fig = go.Figure()
    has_traces = False

    for idx, (algo_name, res_list) in enumerate(algos_dict.items()):
        curves = [r.convergence_curve for r in res_list if r.convergence_curve and len(r.convergence_curve) > 0]
        if not curves:
            continue

        min_len = min(len(c) for c in curves)
        if min_len == 0:
            continue

        arr = np.array([c[:min_len] for c in curves], dtype=float)
        x = list(range(1, min_len + 1))
        mean_curve = np.mean(arr, axis=0)
        std_curve = np.std(arr, axis=0)

        upper = mean_curve + std_curve
        lower = mean_curve - std_curve

        color = colors[idx % len(colors)]
        rgba_fill = _hex_to_rgba(color, alpha=0.18)

        # Protect against non-positive bounds on log scale
        if np.all(mean_curve > 0):
            lower_plot = np.maximum(lower, 1e-30)
            upper_plot = np.maximum(upper, 1e-30)
            mean_plot = np.maximum(mean_curve, 1e-30)
        else:
            lower_plot = lower
            upper_plot = upper
            mean_plot = mean_curve

        # Upper bound (invisible boundary line)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=upper_plot,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Lower bound with ribbon fill up to upper trace
        fig.add_trace(
            go.Scatter(
                x=x,
                y=lower_plot,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=rgba_fill,
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Mean curve line
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean_plot,
                mode="lines",
                name=algo_name,
                line=dict(color=color, width=2.5),
                hovertemplate=(
                    f"<b>{algo_name}</b><br>"
                    "Iteration: %{x}<br>"
                    "Mean Fitness: %{y:.4e}<extra></extra>"
                ),
            )
        )
        has_traces = True

    if not has_traces:
        return None

    fig.update_layout(
        title=dict(
            text=f"Convergence Curves — {prob_name}",
            font=dict(size=14, color="#1e293b"),
        ),
        xaxis=dict(
            title="Iteration",
            gridcolor="#e2e8f0",
            zerolinecolor="#cbd5e1",
        ),
        yaxis=dict(
            title="Best Fitness (log scale)",
            type="log",
            gridcolor="#e2e8f0",
            zerolinecolor="#cbd5e1",
        ),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
        margin=dict(l=60, r=20, t=50, b=40),
        height=380,
    )
    return fig


def _build_boxplot_figure(
    prob_name: str,
    algos_dict: Dict[str, List[Result]],
    colors: Sequence[str],
) -> Optional[go.Figure]:
    """Generate interactive Plotly box plot for algorithm results on a problem.

    Args:
        prob_name: Name of the optimization problem.
        algos_dict: Dict mapping algorithm names to lists of Result objects.
        colors: Color palette sequence.

    Returns:
        Plotly Figure object or None if no results.
    """
    fig = go.Figure()
    has_traces = False

    for idx, (algo_name, res_list) in enumerate(algos_dict.items()):
        if not res_list:
            continue
        fits = [r.best_fitness for r in res_list]
        color = colors[idx % len(colors)]

        fig.add_trace(
            go.Box(
                y=fits,
                name=algo_name,
                marker_color=color,
                boxpoints="all",
                jitter=0.25,
                pointpos=-1.5,
                marker=dict(size=5, opacity=0.75),
                hovertemplate=f"<b>{algo_name}</b><br>Fitness: %{{y:.4e}}<extra></extra>",
            )
        )
        has_traces = True

    if not has_traces:
        return None

    fig.update_layout(
        title=dict(
            text=f"Fitness Distribution (Box Plot) — {prob_name}",
            font=dict(size=14, color="#1e293b"),
        ),
        xaxis=dict(
            title="Algorithm",
            gridcolor="#e2e8f0",
        ),
        yaxis=dict(
            title="Best Fitness",
            gridcolor="#e2e8f0",
            zerolinecolor="#cbd5e1",
        ),
        template="plotly_white",
        margin=dict(l=60, r=20, t=50, b=40),
        height=380,
        showlegend=False,
    )
    return fig


def _build_heatmap_figure(
    all_results: Dict[str, Dict[str, List[Result]]],
) -> Optional[go.Figure]:
    """Generate interactive Plotly heatmap comparing algorithms across problems.

    Args:
        all_results: Nested dict of results.

    Returns:
        Plotly Figure object or None if less than 2 algorithms exist.
    """
    if not all_results:
        return None

    prob_names = sorted(list(all_results.keys()))
    algo_set = set()
    for algos_dict in all_results.values():
        algo_set.update(algos_dict.keys())
    algo_names = sorted(list(algo_set))

    # Heatmap is only relevant if there are multiple algorithms
    if len(algo_names) < 2:
        return None

    n_algos = len(algo_names)
    n_probs = len(prob_names)

    raw_matrix = np.zeros((n_algos, n_probs), dtype=float)

    for j, prob in enumerate(prob_names):
        for i, algo in enumerate(algo_names):
            res_list = all_results.get(prob, {}).get(algo, [])
            if res_list:
                raw_matrix[i, j] = float(np.mean([r.best_fitness for r in res_list]))
            else:
                raw_matrix[i, j] = np.nan

    # Column-wise min-max normalisation per problem
    col_min = np.nanmin(raw_matrix, axis=0)
    col_max = np.nanmax(raw_matrix, axis=0)
    denom = col_max - col_min
    denom[np.isclose(denom, 0.0) | (denom < 1e-30)] = 1.0
    norm_matrix = (raw_matrix - col_min) / denom

    text_matrix = [
        [_format_value(raw_matrix[i, j]) for j in range(n_probs)]
        for i in range(n_algos)
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=norm_matrix,
            x=prob_names,
            y=algo_names,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=12, color="#0f172a"),
            colorscale="YlOrRd_r",
            colorbar=dict(
                title=dict(
                    text="Normalized<br>Score (0=Best)",
                    font=dict(size=12),
                ),
                thickness=16,
            ),
            hovertemplate=(
                "<b>Problem:</b> %{x}<br>"
                "<b>Algorithm:</b> %{y}<br>"
                "<b>Mean Fitness:</b> %{text}<br>"
                "<b>Normalized:</b> %{z:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text="Algorithm Performance Comparison Heatmap (Mean Fitness)",
            font=dict(size=15, color="#1e293b"),
        ),
        xaxis=dict(title="Problem", side="bottom"),
        yaxis=dict(title="Algorithm"),
        template="plotly_white",
        margin=dict(l=80, r=40, t=50, b=50),
        height=max(320, 120 + n_algos * 45),
    )
    return fig


def generate_html_report(
    all_results: Dict[str, Dict[str, List[Result]]],
    output_path: Union[str, Path] = "atlas_report.html",
    title: str = "ATLAS Experiment Report",
) -> Path:
    """Generate an interactive HTML report from experiment results.

    The report includes:
    1. Summary statistics table with mean, std, best, and worst fitness.
    2. Interactive Plotly convergence curves (log scale with std error band).
    3. Interactive Plotly box plots of fitness distributions.
    4. Cross-problem heatmap comparison (when multiple algorithms are tested).

    The HTML is self-contained: the first Plotly figure includes the Plotly JS
    library via CDN (``include_plotlyjs='cdn'``), and all subsequent figures omit
    duplicate script injections (``include_plotlyjs=False``).

    Args:
        all_results: Nested dict of structure ``{problem_name: {algo_name: [Result, ...]}}``.
        output_path: Target path for the generated HTML file. Defaults to ``"atlas_report.html"``.
        title: Title for the HTML report. Defaults to ``"ATLAS Experiment Report"``.

    Returns:
        Path object pointing to the written HTML file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Compute high-level summary KPIs
    total_problems = len(all_results)
    all_algos = sorted(list({algo for prob in all_results.values() for algo in prob.keys()}))
    total_algorithms = len(all_algos)
    total_runs = sum(
        len(res_list)
        for prob in all_results.values()
        for res_list in prob.values()
    )
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Plotly figure conversion helper managing first-include CDN flag
    first_plot_done = False

    def render_plotly_div(fig: Optional[go.Figure]) -> str:
        nonlocal first_plot_done
        if fig is None:
            return "<div class='no-plot'>No plot data available.</div>"
        include_cdn = "cdn" if not first_plot_done else False
        first_plot_done = True
        return fig.to_html(full_html=False, include_plotlyjs=include_cdn)

    # 1. Build Summary Table
    table_section_html = _build_summary_table_html(all_results)

    # 2. Build Per-Problem Visualizations (Convergence + Box Plot)
    problem_sections_html: List[str] = []
    for prob_name, algos_dict in all_results.items():
        conv_fig = _build_convergence_figure(prob_name, algos_dict, PALETTE)
        box_fig = _build_boxplot_figure(prob_name, algos_dict, PALETTE)

        conv_html = render_plotly_div(conv_fig)
        box_html = render_plotly_div(box_fig)

        problem_sections_html.append(
            f"""
            <div class="card mb-4">
                <div class="card-header">
                    <h3>Problem: {prob_name}</h3>
                </div>
                <div class="card-body grid-2">
                    <div class="chart-container">{conv_html}</div>
                    <div class="chart-container">{box_html}</div>
                </div>
            </div>
            """
        )

    # 3. Build Heatmap (if multiple algorithms exist)
    heatmap_fig = _build_heatmap_figure(all_results)
    heatmap_section_html = ""
    if heatmap_fig is not None:
        heatmap_html = render_plotly_div(heatmap_fig)
        heatmap_section_html = f"""
        <div class="card mb-4">
            <div class="card-header">
                <h3>Cross-Algorithm Comparison Heatmap</h3>
            </div>
            <div class="card-body">
                <div class="chart-container">{heatmap_html}</div>
            </div>
        </div>
        """

    # Assemble complete self-contained HTML page
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --primary: #2563eb;
            --primary-light: #eff6ff;
            --border-color: #e2e8f0;
            --success-color: #16a34a;
            --danger-color: #dc2626;
            --radius: 10px;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.5;
            padding: 24px 16px;
        }}

        .container {{
            max-width: 1280px;
            margin: 0 auto;
        }}

        .header {{
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            color: #ffffff;
            padding: 32px 28px;
            border-radius: var(--radius);
            margin-bottom: 24px;
            box-shadow: var(--shadow);
        }}

        .header h1 {{
            font-size: 26px;
            font-weight: 700;
            margin-bottom: 8px;
        }}

        .header .meta {{
            color: #94a3b8;
            font-size: 13px;
        }}

        .kpi-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}

        .kpi-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            padding: 18px 20px;
            box-shadow: var(--shadow);
        }}

        .kpi-title {{
            font-size: 13px;
            text-transform: uppercase;
            font-weight: 600;
            color: var(--text-muted);
            letter-spacing: 0.5px;
        }}

        .kpi-value {{
            font-size: 28px;
            font-weight: 700;
            color: var(--primary);
            margin-top: 4px;
        }}

        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            overflow: hidden;
        }}

        .mb-4 {{
            margin-bottom: 24px;
        }}

        .card-header {{
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-color);
            background-color: #fcfcfd;
        }}

        .card-header h3 {{
            font-size: 17px;
            font-weight: 600;
            color: var(--text-main);
        }}

        .card-body {{
            padding: 20px;
        }}

        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
        }}

        @media (min-width: 900px) {{
            .grid-2 {{
                grid-template-columns: 1fr 1fr;
            }}
        }}

        .chart-container {{
            min-height: 380px;
            width: 100%;
        }}

        .table-responsive {{
            width: 100%;
            overflow-x: auto;
        }}

        .report-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            text-align: left;
        }}

        .report-table th {{
            background-color: #f1f5f9;
            color: #334155;
            font-weight: 600;
            padding: 12px 14px;
            border-bottom: 2px solid var(--border-color);
        }}

        .report-table td {{
            padding: 12px 14px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-main);
        }}

        .report-table tr:hover {{
            background-color: #f8fafc;
        }}

        .highlight-row {{
            background-color: #f0fdf4;
        }}

        .num-cell {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 13px;
        }}

        .text-success {{
            color: var(--success-color);
        }}

        .text-danger {{
            color: var(--danger-color);
        }}

        .algo-pill {{
            display: inline-block;
            background-color: var(--primary-light);
            color: var(--primary);
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 12px;
        }}

        .badge-best {{
            display: inline-block;
            background-color: #dcfce7;
            color: #15803d;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 11px;
            margin-left: 6px;
        }}

        .no-data, .no-plot {{
            color: var(--text-muted);
            font-style: italic;
            padding: 12px;
        }}

        .footer {{
            text-align: center;
            color: var(--text-muted);
            font-size: 13px;
            margin-top: 36px;
            padding-bottom: 24px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>{title}</h1>
            <div class="meta">Generated at {now_str} &bull; ATLAS Optimization Research Platform</div>
        </header>

        <section class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-title">Problems Tested</div>
                <div class="kpi-value">{total_problems}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Algorithms Evaluated</div>
                <div class="kpi-value">{total_algorithms}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Total Runs</div>
                <div class="kpi-value">{total_runs}</div>
            </div>
        </section>

        <section class="card mb-4">
            <div class="card-header">
                <h3>Executive Summary Statistics</h3>
            </div>
            <div class="card-body">
                {table_section_html}
            </div>
        </section>

        {heatmap_section_html}

        <section>
            {''.join(problem_sections_html)}
        </section>

        <footer class="footer">
            <p>ATLAS Optimization Research Platform &bull; Automated Interactive Report</p>
        </footer>
    </div>
</body>
</html>
"""

    out_file.write_text(full_html, encoding="utf-8")
    return out_file
