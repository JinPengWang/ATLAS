"""Streamlit Interactive Web Console for ATLAS Benchmark and Optimization Research."""

from __future__ import annotations

import io
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from atlas.core.experiment import Experiment, ExperimentConfig
from atlas.stats import (
    calculate_average_ranks,
    export_latex_table,
    export_markdown_table,
    friedman_test,
    plot_critical_difference,
)
from atlas.utils.registry import list_algorithms, list_problems


def render_dashboard() -> None:
    """Render the main Streamlit application layout and interactive elements."""
    try:
        import streamlit as st
        import plotly.graph_objects as go
    except ImportError:
        print("Streamlit and Plotly are required to run the ATLAS Web Dashboard.")
        return

    st.set_page_config(
        page_title="ATLAS Benchmark Dashboard",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🧬 ATLAS Metaheuristics & Benchmark Dashboard")
    st.markdown(
        "Interactive experimentation platform for algorithm comparison, "
        "statistical ranking, and publication-ready report generation."
    )

    # -----------------------------------------------------------------------
    # Sidebar Configuration
    # -----------------------------------------------------------------------
    st.sidebar.header("⚙️ Experiment Configuration")

    all_algos = sorted(list_algorithms())
    default_algos = [a for a in ["pso", "de", "gwo", "lshade", "woa"] if a in all_algos]
    if not default_algos:
        default_algos = all_algos[:3]

    selected_algos = st.sidebar.multiselect(
        "Algorithms to Benchmark",
        options=all_algos,
        default=default_algos,
    )

    all_probs = sorted(list_problems())
    default_probs = [
        p for p in ["sphere", "rastrigin", "rosenbrock", "welded_beam", "spring_design"] if p in all_probs
    ]
    if not default_probs:
        default_probs = all_probs[:3]

    selected_probs = st.sidebar.multiselect(
        "Benchmark Problems",
        options=all_probs,
        default=default_probs,
    )

    col1, col2 = st.sidebar.columns(2)
    dim = col1.number_input("Dimension (dim)", min_value=2, max_value=200, value=10)
    runs = col2.number_input("Runs per Problem", min_value=1, max_value=50, value=5)

    col3, col4 = st.sidebar.columns(2)
    max_iter = col3.number_input("Max Iterations", min_value=5, max_value=5000, value=100)
    pop_size = col4.number_input("Population Size", min_value=4, max_value=500, value=30)

    n_jobs = st.sidebar.slider("Parallel Workers (n_jobs)", min_value=1, max_value=16, value=1)
    seed = st.sidebar.number_input("Base Random Seed", value=42)

    run_btn = st.sidebar.button("🚀 Run Benchmark Experiment", type="primary", use_container_width=True)

    # -----------------------------------------------------------------------
    # Execution Logic
    # -----------------------------------------------------------------------
    if run_btn:
        if not selected_algos:
            st.error("Please select at least one algorithm.")
            return
        if not selected_probs:
            st.error("Please select at least one problem.")
            return

        config = ExperimentConfig(
            algorithms=selected_algos,
            problems=selected_probs,
            dims=[int(dim)],
            max_iter=int(max_iter),
            pop_size=int(pop_size),
            runs=int(runs),
            seed=int(seed),
            n_jobs=int(n_jobs),
        )

        with st.spinner("Running experiments in parallel..."):
            exp = Experiment(config)
            all_results = exp.run_all()
            st.session_state["results"] = all_results
            st.session_state["config"] = config
            st.success("Benchmark completed successfully!")

    # -----------------------------------------------------------------------
    # Results Display & Visualizations
    # -----------------------------------------------------------------------
    if "results" in st.session_state:
        all_results = st.session_state["results"]

        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Convergence Curves",
            "📊 Statistical Boxplots",
            "🏆 Friedman Ranking & CD Diagram",
            "💾 Data & LaTeX Export",
        ])

        # ----------------- Tab 1: Convergence -----------------
        with tab1:
            st.subheader("Convergence Profiles")
            for prob_key, algos in all_results.items():
                fig = go.Figure()
                for algo_name, res_list in algos.items():
                    curves = [r.convergence_curve for r in res_list if r.convergence_curve]
                    if not curves:
                        continue
                    max_len = max(len(c) for c in curves)
                    padded = [c + [c[-1]] * (max_len - len(c)) for c in curves]
                    arr = np.array(padded)
                    mean_c = np.mean(arr, axis=0)
                    std_c = np.std(arr, axis=0)
                    iters = list(range(len(mean_c)))

                    fig.add_trace(
                        go.Scatter(x=iters, y=mean_c.tolist(), name=algo_name, mode="lines")
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=iters + iters[::-1],
                            y=(mean_c + std_c).tolist() + (mean_c - std_c).tolist(),
                            fill="toself",
                            opacity=0.15,
                            showlegend=False,
                            name=f"{algo_name}_std",
                        )
                    )

                fig.update_layout(
                    title=f"Convergence on {prob_key}",
                    xaxis_title="Iteration",
                    yaxis_title="Best Fitness (Log Scale)",
                    yaxis_type="log",
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

        # ----------------- Tab 2: Boxplots -----------------
        with tab2:
            st.subheader("Performance Distributions Across Runs")
            for prob_key, algos in all_results.items():
                fig = go.Figure()
                for algo_name, res_list in algos.items():
                    vals = [r.best_fitness for r in res_list]
                    fig.add_trace(
                        go.Box(y=vals, name=algo_name, boxpoints="all", jitter=0.3)
                    )
                fig.update_layout(
                    title=f"Final Fitness Distribution on {prob_key}",
                    yaxis_title="Best Fitness",
                )
                st.plotly_chart(fig, use_container_width=True)

        # ----------------- Tab 3: Rankings & CD Diagram -----------------
        with tab3:
            st.subheader("Non-Parametric Statistical Ranking")
            prob_keys = list(all_results.keys())
            algo_keys = list(next(iter(all_results.values())).keys())

            if len(prob_keys) >= 2 and len(algo_keys) >= 2:
                # Build mean performance matrix
                matrix = []
                for pk in prob_keys:
                    row = [np.mean([r.best_fitness for r in all_results[pk][ak]]) for ak in algo_keys]
                    matrix.append(row)
                perf_matrix = np.array(matrix)

                ranks = calculate_average_ranks(perf_matrix, algorithm_names=algo_keys)
                f_res = friedman_test(perf_matrix, algorithm_names=algo_keys)

                c1, c2 = st.columns(2)
                c1.metric("Friedman Statistic", f"{f_res.stat:.4f}")
                c2.metric("p-value", f"{f_res.p_value:.4e}", delta="Significant" if f_res.is_significant else "Not Significant")

                # Rank dataframe
                df_ranks = pd.DataFrame(
                    [{"Algorithm": k, "Average Rank": v} for k, v in f_res.rankings_order]
                )
                st.dataframe(df_ranks, use_container_width=True)

                # Critical Difference Diagram
                st.subheader("Critical Difference (CD) Diagram")
                cd_fig = plot_critical_difference(ranks, n_datasets=len(prob_keys))
                st.pyplot(cd_fig)
            else:
                st.info("Run at least 2 algorithms on at least 2 problems to generate Friedman ranks and CD diagrams.")

        # ----------------- Tab 4: Data Export -----------------
        with tab4:
            st.subheader("Export Academic Tables")
            # Build summary DataFrame from nested results dict
            rows = []
            for _prob_key, _algos in all_results.items():
                for _algo_name, _res_list in _algos.items():
                    _fits = [r.best_fitness for r in _res_list]
                    rows.append({
                        "problem": _prob_key,
                        "algorithm": _algo_name,
                        "mean": float(np.mean(_fits)),
                        "std": float(np.std(_fits)),
                        "best": float(np.min(_fits)),
                        "worst": float(np.max(_fits)),
                    })
            _df_summary = pd.DataFrame(rows)
            latex_code = export_latex_table(_df_summary)
            md_code = export_markdown_table(_df_summary)

            st.text_area("LaTeX Table Code", latex_code, height=200)
            st.download_button(
                "📥 Download LaTeX Table (.tex)",
                latex_code,
                file_name="atlas_table.tex",
                mime="text/plain",
            )

            st.text_area("Markdown Table", md_code, height=200)
            st.download_button(
                "📥 Download Markdown (.md)",
                md_code,
                file_name="atlas_results.md",
                mime="text/markdown",
            )


if __name__ == "__main__":
    render_dashboard()
