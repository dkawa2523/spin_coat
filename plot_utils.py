"""Plotting helpers for spin coating runs."""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib

# Use non-interactive backend for CLI environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import csv
import time

from model import JaxSpinODESolver, NumpySpinODESolver, build_params_from_theta


def plot_predictions(
    cfg: Dict,
    sensor_names: List[str],
    time_grid: Dict[str, np.ndarray],
    solver_time_grid: Dict[str, np.ndarray],
    data: Dict[str, np.ndarray],
    theta: np.ndarray,
    output_dir: str,
) -> List[str]:
    """Plot measured vs predicted thickness for each sensor.

    Returns list of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    gp, sensors = build_params_from_theta(theta, cfg, sensor_names)
    backend = cfg["fit"].get("backend", "numpy").lower()
    if backend == "jax":
        try:
            solver = JaxSpinODESolver(gp, sensors)
        except ImportError:
            solver = NumpySpinODESolver(gp, sensors)
    else:
        solver = NumpySpinODESolver(gp, sensors)

    saved = []
    for name in sensor_names:
        meas_times = time_grid[name]
        solver_times = solver_time_grid[name]
        h_meas = data[name]
        if cfg["fit"]["initial_h"]["mode"] == "from_data":
            h0 = float(h_meas[0])
        else:
            h0 = float(cfg["fit"]["initial_h"]["fixed_value"])
        h_pred = solver.solve(name, solver_times, h0)
        h_pred_np = np.asarray(h_pred)
        if solver_times.shape != meas_times.shape or not np.allclose(solver_times, meas_times):
            h_pred_np = np.interp(meas_times, solver_times, h_pred_np)

        plt.figure(figsize=(6, 4))
        plt.plot(meas_times, h_meas, "o", label="data", markersize=4, alpha=0.7)
        plt.plot(meas_times, h_pred_np, "-", label="model", linewidth=1.6)
        plt.xlabel("Time [s]")
        plt.ylabel("Thickness [m]")
        plt.title(f"Sensor: {name}")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"optuna_fit_{name}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        saved.append(out_path)
    return saved


def save_optuna_diagnostics(study, output_dir: str) -> Dict[str, List]:
    """Save optimisation history, parallel coordinates, parameter importance, and top-10 trials."""
    from optuna.trial import TrialState
    from optuna.visualization import matplotlib as ovm
    from optuna.visualization import (
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_param_importances,
    )
    from optuna.importance import FanovaImportanceEvaluator

    os.makedirs(output_dir, exist_ok=True)
    saved_paths: Dict[str, List] = {"plots": [], "tables": [], "top_trials": []}

    # Optimisation history (matplotlib PNG + Plotly HTML with log-scale loss)
    try:
        ax = ovm.plot_optimization_history(study)
        ax.set_yscale("log")
        hist_path = os.path.join(output_dir, "optuna_history.png")
        ax.figure.savefig(hist_path, dpi=150)
        plt.close(ax.figure)
        saved_paths["plots"].append(hist_path)
        fig = plot_optimization_history(study)
        fig.update_yaxes(type="log")
        hist_html = os.path.join(output_dir, "optuna_history.html")
        fig.write_html(hist_html, include_plotlyjs="cdn")
        saved_paths["plots"].append(hist_html)
    except Exception as exc:
        print(f"Warning: failed to plot optimization history: {exc}")

    # Parallel coordinates (matplotlib PNG + Plotly HTML)
    try:
        ax = ovm.plot_parallel_coordinate(study)
        pc_path = os.path.join(output_dir, "optuna_parallel_coords.png")
        ax.figure.savefig(pc_path, dpi=150)
        plt.close(ax.figure)
        saved_paths["plots"].append(pc_path)
        fig = plot_parallel_coordinate(study)
        pc_html = os.path.join(output_dir, "optuna_parallel_coords.html")
        fig.write_html(pc_html, include_plotlyjs="cdn")
        saved_paths["plots"].append(pc_html)
    except Exception as exc:
        print(f"Warning: failed to plot parallel coordinates: {exc}")

    # Helper to filter out penalised/outlier trials for importance
    def _filter_trials_for_importance(trials):
        finite = [t for t in trials if t.state == TrialState.COMPLETE and np.isfinite(t.value)]
        if not finite:
            return []
        # Remove extreme penalties (e.g., our 1e30) and top 10% outliers to stabilise importance
        values = np.array([t.value for t in finite], dtype=float)
        penalty_cap = 1e28
        mask = values < penalty_cap
        finite = [t for t, keep in zip(finite, mask) if keep]
        if not finite:
            return []
        values = np.array([t.value for t in finite], dtype=float)
        if len(values) >= 10:
            q90 = np.quantile(values, 0.9)
            finite = [t for t in finite if t.value <= q90]
        return finite

    # Parameter importance (matplotlib PNG + Plotly HTML) using filtered trials and fANOVA
    try:
        completed = _filter_trials_for_importance(study.trials)
        if len(completed) >= 8:
            sub_study = study
            # Recreate a study-like object with filtered trials? Optuna plots can take study directly;
            # we temporarily monkey-patch trials for plotting.
            original_trials = sub_study.trials
            sub_study._trials = completed  # type: ignore
            try:
                evaluator = FanovaImportanceEvaluator()
                ax = ovm.plot_param_importances(sub_study, evaluator=evaluator)
                imp_path = os.path.join(output_dir, "optuna_param_importance.png")
                ax.figure.savefig(imp_path, dpi=150)
                plt.close(ax.figure)
                saved_paths["plots"].append(imp_path)
                fig = plot_param_importances(sub_study, evaluator=evaluator)
                imp_html = os.path.join(output_dir, "optuna_param_importance.html")
                fig.write_html(imp_html, include_plotlyjs="cdn")
                saved_paths["plots"].append(imp_html)
            finally:
                sub_study._trials = original_trials  # type: ignore
        else:
            print("Info: not enough stable trials for importance (need >=8 after filtering); skipping global importance plots.")
    except Exception as exc:
        print(f"Warning: failed to plot parameter importances: {exc}")

    # Per-sensor parameter importance (Plotly HTML)
    sensor_keys = {}
    for t in study.trials:
        for key in t.params.keys():
            if key.startswith("k_flow_"):
                name = key.replace("k_flow_", "")
                sensor_keys.setdefault(name, set()).add(key)
            if key.startswith("delta_E_"):
                name = key.replace("delta_E_", "")
                sensor_keys.setdefault(name, set()).add(key)
    for sensor, keys in sensor_keys.items():
        try:
            params = sorted(list(keys | {"log10_mu0", "m", "log10_E0", "alpha_E"}))
            filt = _filter_trials_for_importance(study.trials)
            if len(filt) >= 8:
                sub_study = study
                original_trials = sub_study.trials
                sub_study._trials = filt  # type: ignore
                try:
                    evaluator = FanovaImportanceEvaluator()
                    fig = plot_param_importances(sub_study, evaluator=evaluator, params=params)
                    html_path = os.path.join(output_dir, f"optuna_param_importance_{sensor}.html")
                    fig.write_html(html_path, include_plotlyjs="cdn")
                    saved_paths["plots"].append(html_path)
                finally:
                    sub_study._trials = original_trials  # type: ignore
            else:
                print(f"Info: not enough stable trials for importance of sensor {sensor} (need >=8 after filtering); skipping.")
        except Exception as exc:
            print(f"Warning: failed to plot param importance for sensor {sensor}: {exc}")

    # Trial duration timeline (Plotly HTML)
    try:
        durations = []
        trial_nums = []
        for t in study.trials:
            if t.datetime_start and t.datetime_complete:
                dur = (t.datetime_complete - t.datetime_start).total_seconds()
                durations.append(dur)
                trial_nums.append(t.number)
        if durations:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Bar(x=trial_nums, y=durations, name="trial duration [s]"))
            fig.update_layout(
                xaxis_title="Trial number",
                yaxis_title="Duration [s]",
                title="Optuna trial durations",
            )
            dur_html = os.path.join(output_dir, "optuna_trial_durations.html")
            fig.write_html(dur_html, include_plotlyjs="cdn")
            saved_paths["plots"].append(dur_html)
    except Exception as exc:
        print(f"Warning: failed to plot trial durations: {exc}")

    # Top 10 trials table (CSV)
    completed = [
        t for t in study.trials if t.state == TrialState.COMPLETE and np.isfinite(t.value)
    ]
    completed.sort(key=lambda t: t.value)
    top_trials = completed[:10]
    if top_trials:
        # Collect all parameter keys across top trials to stabilise column order
        param_keys = sorted({k for t in top_trials for k in t.params.keys()})
        csv_path = os.path.join(output_dir, "optuna_top10_trials.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "trial_number", "value"] + param_keys)
            for rank, t in enumerate(top_trials, start=1):
                row = [rank, t.number, t.value]
                for k in param_keys:
                    row.append(t.params.get(k, ""))
                writer.writerow(row)
        saved_paths["tables"].append(csv_path)
        # Collect structured data for console output
        for rank, t in enumerate(top_trials, start=1):
            saved_paths["top_trials"].append(
                {
                    "rank": rank,
                    "trial_number": t.number,
                    "value": t.value,
                    "params": t.params,
                }
            )
    else:
        print("Warning: no completed trials available to record top-10 table.")

    # Full trials table (CSV)
    if completed:
        all_param_keys = sorted({k for t in completed for k in t.params.keys()})
        csv_path = os.path.join(output_dir, "optuna_all_trials.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trial_number", "value", "state"] + all_param_keys)
            for t in completed:
                row = [t.number, t.value, t.state.name]
                for k in all_param_keys:
                    row.append(t.params.get(k, ""))
                writer.writerow(row)
        saved_paths["tables"].append(csv_path)

    return saved_paths
