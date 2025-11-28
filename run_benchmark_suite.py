"""Batch benchmark runner: evaluate multiple configs and write a summary report.

Usage example:
  ./.venv/bin/python run_benchmark_suite.py \
      --configs config_benchmark.yaml config_benchmark_euler_numpy.yaml \
      --trials 50 --timeout 0 --output bench_report.md

The script runs optimisation (Optuna or gradient) defined in each YAML, logs
best loss, completed trials/iters, duration, backend/integrator, and writes
tables for later comparison.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from io_utils import (
    build_solver_time_grid,
    configure_jax_platform,
    load_config,
    load_data,
)
from spinfit.parameters import build_params_from_theta, pack_theta, unpack_theta
from spinfit.solvers import create_solver
from spinfit.losses import mse_loss
from spinfit.optimize import run_optuna_search
from spinfit.equations import rhs_ode


def run_optuna(cfg: Dict, config_path: str, trials: int | None, timeout: float | None):
    cfg = cfg.copy()
    cfg.setdefault("fit", {}).setdefault("optuna", {})
    if trials is not None:
        cfg["fit"]["optuna"]["n_trials"] = trials
    if timeout is not None:
        cfg["fit"]["optuna"]["timeout"] = timeout
    sensor_names, time_grid, data = load_data(cfg, config_path)
    solver_time_grid = build_solver_time_grid(cfg, sensor_names, time_grid)
    start = time.time()
    theta, best_loss, params_dict, study = run_optuna_search(
        cfg, sensor_names, time_grid, data, solver_time_grid=solver_time_grid
    )
    elapsed = time.time() - start
    completed = [t for t in study.trials if t.state.name == "COMPLETE" and np.isfinite(t.value)]
    return {
        "best_loss": best_loss,
        "duration_sec": elapsed,
        "completed_trials": len(completed),
        "backend": cfg["fit"].get("backend", "numpy"),
        "integrator": cfg["fit"].get("integrator", "euler"),
        "optimiser": "optuna",
        "continuity": cfg["fit"].get("continuity", {}).get("enabled", False),
        "continuity_weight": cfg["fit"].get("continuity", {}).get("weight", 0.0),
    }


def run_gradient(cfg: Dict, config_path: str):
    sensor_names, time_grid, data = load_data(cfg, config_path)
    solver_time_grid = build_solver_time_grid(cfg, sensor_names, time_grid)
    init_params = {
        'global': {
            'log10_mu0': cfg['fit']['parameters']['global']['log10_mu0']['init'],
            'm': cfg['fit']['parameters']['global']['m']['init'],
            'log10_E0': cfg['fit']['parameters']['global']['log10_E0']['init'],
            'alpha_E': cfg['fit']['parameters']['global']['alpha_E']['init'],
        },
        'per_sensor': {
            'k_flow': {s: cfg['fit']['parameters']['per_sensor']['k_flow']['init'] for s in sensor_names},
            'delta_E': {s: cfg['fit']['parameters']['per_sensor']['delta_E']['init'] for s in sensor_names},
        },
    }
    theta = pack_theta(init_params, sensor_names)
    backend = cfg['fit'].get('backend', 'numpy').lower()
    integrator = cfg['fit'].get('integrator', 'euler')
    lr = cfg['fit']['gradient']['learning_rate']
    max_iters = cfg['fit']['gradient']['max_iters']
    grad_eps = cfg['fit']['gradient']['grad_eps']
    tol_grad_norm = cfg['fit']['gradient']['tol_grad_norm']
    print_every = cfg['fit']['gradient']['print_every']

    start = time.time()
    best_theta = theta.copy()
    gp, sensors = build_params_from_theta(theta, cfg, sensor_names)
    solver = create_solver(gp, sensors, backend=backend, integrator=integrator, rhs_fn=rhs_ode)
    best_loss = mse_loss(solver, cfg, sensor_names, time_grid, data, solver_time_grid)
    completed = 0

    for it in range(1, max_iters + 1):
        gp, sensors = build_params_from_theta(theta, cfg, sensor_names)
        solver = create_solver(gp, sensors, backend=backend, integrator=integrator, rhs_fn=rhs_ode)
        loss_val = mse_loss(solver, cfg, sensor_names, time_grid, data, solver_time_grid)
        grad = np.zeros_like(theta)
        for i in range(len(theta)):
            dtheta = np.zeros_like(theta)
            dtheta[i] = grad_eps
            gp_p, sensors_p = build_params_from_theta(theta + dtheta, cfg, sensor_names)
            solver_p = create_solver(gp_p, sensors_p, backend=backend, integrator=integrator, rhs_fn=rhs_ode)
            f_plus = mse_loss(solver_p, cfg, sensor_names, time_grid, data, solver_time_grid)
            gp_m, sensors_m = build_params_from_theta(theta - dtheta, cfg, sensor_names)
            solver_m = create_solver(gp_m, sensors_m, backend=backend, integrator=integrator, rhs_fn=rhs_ode)
            f_minus = mse_loss(solver_m, cfg, sensor_names, time_grid, data, solver_time_grid)
            grad[i] = (f_plus - f_minus) / (2.0 * grad_eps)
        grad_norm = np.linalg.norm(grad)
        theta = theta - lr * grad
        completed = it
        if loss_val < best_loss:
            best_loss = loss_val
            best_theta = theta.copy()
        if print_every and it % print_every == 0:
            print(f"[GD] iter {it} loss={loss_val:.3e} grad_norm={grad_norm:.3e} best={best_loss:.3e}")
        if grad_norm < tol_grad_norm:
            break
    elapsed = time.time() - start
    return {
        "best_loss": best_loss,
        "duration_sec": elapsed,
        "completed_trials": completed,
        "backend": backend,
        "integrator": integrator,
        "optimiser": "gradient",
        "continuity": cfg["fit"].get("continuity", {}).get("enabled", False),
        "continuity_weight": cfg["fit"].get("continuity", {}).get("weight", 0.0),
    }


def main():
    parser = argparse.ArgumentParser(description="Run batch benchmarks and summarize results.")
    parser.add_argument('--configs', nargs='+', required=True, help='List of YAML config paths')
    parser.add_argument('--trials', type=int, help='Override Optuna n_trials')
    parser.add_argument('--timeout', type=float, help='Override Optuna timeout (sec)')
    parser.add_argument('--output', type=str, default='bench_report.md', help='Output markdown path')
    args = parser.parse_args()

    rows = []
    for cfg_path in args.configs:
        print(f"Running {cfg_path} ...")
        cfg = load_config(cfg_path)
        configure_jax_platform(cfg)
        opt = cfg['fit'].get('optimiser', 'optuna').lower()
        try:
            if opt == 'optuna':
                res = run_optuna(cfg, cfg_path, args.trials, args.timeout)
            elif opt == 'gradient':
                res = run_gradient(cfg, cfg_path)
            else:
                print(f"Skipping {cfg_path}: unsupported optimiser {opt}")
                continue
            res['config'] = cfg_path
            rows.append(res)
            print(f"  best_loss={res['best_loss']:.3e}, duration={res['duration_sec']:.1f}s, completed={res['completed_trials']}")
        except Exception as e:
            print(f"  ERROR in {cfg_path}: {e}")

    # Write summary markdown
    out_path = Path(args.output)
    with out_path.open("w") as f:
        f.write("# Benchmark Summary\n\n")
        f.write("| config | optimiser | backend | integrator | continuity(w) | completed | best_loss | duration[s] |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for r in rows:
            cont = f"{r['continuity']} ({r['continuity_weight']})"
            f.write(
                f"| {r['config']} | {r['optimiser']} | {r['backend']} | {r['integrator']} | {cont} | {r['completed_trials']} | {r['best_loss']:.3e} | {r['duration_sec']:.1f} |\n"
            )
    print(f"Summary written to {out_path}")


if __name__ == "__main__":
    main()
