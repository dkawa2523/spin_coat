"""Unified entrypoint to run optimisation (Optuna/gradient) or forward simulation based on YAML config."""

import argparse
import numpy as np

from io_utils import (
    build_solver_time_grid,
    configure_jax_platform,
    load_config,
    load_data,
    resolve_output_dir,
)
from plot_utils import plot_predictions, save_optuna_diagnostics
from spinfit.equations import rhs_ode
from spinfit.losses import mse_loss
from spinfit.optimize import run_optuna_search
from spinfit.parameters import build_params_from_theta, pack_theta, unpack_theta
from spinfit.solvers import create_solver
from spinfit.simulation import simulate_sensors
from spinfit.neural_ode import HybridSolver, load_params_npz
from model import render_equation_as_text  # reuse existing renderer


def run_forward(cfg, config_path):
    forward_cfg = cfg.get('forward', {})
    if not forward_cfg.get('enabled', False):
        print("Forward mode disabled (forward.enabled=false).")
        return
    sensor_names, time_grid, data = load_data(cfg, config_path)
    solver_time_grid = build_solver_time_grid(cfg, sensor_names, time_grid)
    output_dir = resolve_output_dir({'fit': {'output_dir': forward_cfg.get('output_dir', 'outputs_forward')}}, config_path)
    fixed_params = forward_cfg['parameters']
    theta = pack_theta(fixed_params, sensor_names)
    override = forward_cfg.get('override_model', {})
    model_cfg = cfg['model'].copy()
    for key in ('rho', 'omega', 'h_ref'):
        if override.get(key) is not None:
            model_cfg[key] = override[key]
    cfg_forward = cfg.copy()
    cfg_forward['model'] = model_cfg
    gp, sensors = build_params_from_theta(theta, cfg_forward, sensor_names)
    backend = cfg['fit'].get('backend', 'numpy')
    integrator = forward_cfg.get('integrator', cfg['fit'].get('integrator', 'euler'))

    # Optional: use trained neural ODE parameters for inference
    nn_infer = cfg.get('neural_ode', {}).get('inference', {})
    if nn_infer.get('enabled', False):
        params_path = nn_infer.get('params_path')
        if not params_path or not os.path.exists(params_path):
            raise FileNotFoundError(f"Neural ODE params_path not found: {params_path}")
        params_net = load_params_npz(params_path)
        corr_scale = cfg['neural_ode'].get('corr_scale', 1e-7)
        solver = HybridSolver(
            params_net=params_net,
            gp=gp,
            sensors=sensors,
            sensor_names=sensor_names,
            integrator=integrator,
            corr_scale=corr_scale,
        )
    else:
        solver = create_solver(gp, sensors, backend=backend, integrator=integrator, rhs_fn=rhs_ode)
    results = simulate_sensors(solver, sensor_names, time_grid, data, cfg, solver_time_grid)
    print("Forward simulation completed.\n")
    for name in sensor_names:
        print(f"Sensor: {name}")
        eqn = render_equation_as_text(gp, sensors[name])
        print(eqn)
        print(f"  h(t) start: {float(results[name][0]):.4e}, h(t) end: {float(results[name][-1]):.4e}")
        print()
    print(f"Output directory: {output_dir}")


def run_optuna_mode(cfg, config_path, args):
    sensor_names, time_grid, data = load_data(cfg, config_path)
    solver_time_grid = build_solver_time_grid(cfg, sensor_names, time_grid)
    output_dir = resolve_output_dir(cfg, config_path)
    theta, best_loss, params_dict, study = run_optuna_search(
        cfg, sensor_names, time_grid, data, solver_time_grid=solver_time_grid
    )
    gp, sensors = build_params_from_theta(theta, cfg, sensor_names)
    print("\nBest parameters found (unpacked):")
    for k, v in params_dict['global'].items():
        print(f"  {k}: {v:.6f}")
    for name in sensor_names:
        print(f"  k_flow_{name}: {params_dict['per_sensor']['k_flow'][name]:.6f}")
    for name in sensor_names:
        print(f"  delta_E_{name}: {params_dict['per_sensor']['delta_E'][name]:.6f}")
    print(f"\nBest loss (MSE): {best_loss:.6e}")
    print("\nModel equations for each sensor:\n")
    for name in sensor_names:
        print(f"Sensor: {name}")
        eqn = render_equation_as_text(gp, sensors[name])
        print(eqn)
        print()
    diag_paths = save_optuna_diagnostics(study, output_dir=output_dir)
    print("Saved Optuna diagnostics:")
    for kind, paths in diag_paths.items():
        if kind in ("plots", "tables"):
            for path in paths:
                print(f"  {path}")
    if diag_paths.get("top_trials"):
        print("\nTop 10 trials (by loss):")
        for entry in diag_paths["top_trials"]:
            params_str = ", ".join(
                f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in entry["params"].items()
            )
            print(f"  #{entry['rank']} trial={entry['trial_number']} value={entry['value']:.3e} :: {params_str}")
    if not args.no_plot:
        saved = plot_predictions(cfg, sensor_names, time_grid, solver_time_grid, data, theta, output_dir=output_dir)
        print("Saved plots:")
        for path in saved:
            print(f"  {path}")


def run_grad_mode(cfg, config_path):
    if cfg['fit'].get('optimiser', 'optuna').lower() != 'gradient':
        print("Config fit.optimiser != 'gradient'; skipping gradient descent. Set to 'gradient' to enable.")
        return
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
    if backend == 'jax':
        try:
            import jax  # type: ignore
        except ImportError:
            print("JAX backend requested but jax is not installed. Falling back to numpy.")
            backend = 'numpy'
    integrator = cfg['fit'].get('integrator', 'euler')
    lr = cfg['fit']['gradient']['learning_rate']
    max_iters = cfg['fit']['gradient']['max_iters']
    grad_eps = cfg['fit']['gradient']['grad_eps']
    tol_grad_norm = cfg['fit']['gradient']['tol_grad_norm']
    print_every = cfg['fit']['gradient']['print_every']

    best_theta = theta.copy()
    gp, sensors = build_params_from_theta(theta, cfg, sensor_names)
    solver = create_solver(gp, sensors, backend=backend, integrator=integrator, rhs_fn=rhs_ode)
    best_loss = mse_loss(solver, cfg, sensor_names, time_grid, data, solver_time_grid)

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
        if loss_val < best_loss:
            best_loss = loss_val
            best_theta = theta.copy()
        if print_every and it % print_every == 0:
            print(f"Iter {it:03d}: loss={loss_val:.6e}, grad_norm={grad_norm:.3e}, best_loss={best_loss:.6e}")
        if grad_norm < tol_grad_norm:
            print(f"Gradient norm {grad_norm:.3e} below tolerance {tol_grad_norm}; stopping at iter {it}.")
            break

    gp, sensors = build_params_from_theta(best_theta, cfg, sensor_names)
    unpacked = unpack_theta(best_theta, sensor_names)
    print("\nBest parameters found (unpacked):")
    for k, v in unpacked['global'].items():
        print(f"  {k}: {v:.6f}")
    for name in sensor_names:
        print(f"  k_flow_{name}: {unpacked['per_sensor']['k_flow'][name]:.6f}")
    for name in sensor_names:
        print(f"  delta_E_{name}: {unpacked['per_sensor']['delta_E'][name]:.6f}")
    print(f"\nBest loss (MSE): {best_loss:.6e}")
    print("\nModel equations for each sensor:\n")
    for name in sensor_names:
        print(f"Sensor: {name}")
        eqn = render_equation_as_text(gp, sensors[name])
        print(eqn)
        print()


def main():
    parser = argparse.ArgumentParser(description="Unified runner for optimisation or forward simulation.")
    parser.add_argument('--config', type=str, default='config_example.yaml', help='Path to config YAML')
    parser.add_argument('--trials', type=int, help='Override n_trials (Optuna)')
    parser.add_argument('--timeout', type=float, help='Override timeout (Optuna)')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting (Optuna only)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_jax_platform(cfg)

    # Optuna overrides
    cfg.setdefault('fit', {}).setdefault('optuna', {})
    if args.trials is not None:
        cfg['fit']['optuna']['n_trials'] = args.trials
    if args.timeout is not None:
        cfg['fit']['optuna']['timeout'] = args.timeout

    # Mode selection: explicit run_mode overrides; forward flag respected
    run_mode = cfg.get('run_mode', 'optimize').lower()
    if run_mode == 'forward' or cfg.get('forward', {}).get('enabled', False):
        run_forward(cfg, args.config)
        return
    optimiser = cfg['fit'].get('optimiser', 'optuna').lower()
    if optimiser == 'optuna':
        run_optuna_mode(cfg, args.config, args)
    elif optimiser == 'gradient':
        run_grad_mode(cfg, args.config)
    else:
        print(f"Unknown optimiser mode '{optimiser}'. Use 'optuna' or 'gradient', or enable forward.enabled.")


if __name__ == "__main__":
    main()
