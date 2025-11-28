"""Command-line interface to run simple gradient descent on spin coating data."""

import argparse
import numpy as np

from io_utils import build_solver_time_grid, configure_jax_platform, load_config, load_data
from spinfit.parameters import pack_theta, unpack_theta, build_params_from_theta
from spinfit.solvers import create_solver
from spinfit.losses import mse_loss
from model import render_equation_as_text  # reuse equation text


def main() -> None:
    parser = argparse.ArgumentParser(description="Spin coating parameter fitting with gradient descent")
    parser.add_argument('--config', type=str, default='config_example.yaml', help='Path to the YAML config file')
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_jax_platform(cfg)
    backend = cfg['fit'].get('backend', 'numpy').lower()
    if backend == 'jax':
        try:
            import jax  # type: ignore
        except ImportError:
            print("JAX backend requested but jax is not installed. Falling back to numpy.")
            cfg['fit']['backend'] = 'numpy'
            backend = 'numpy'
    if cfg['fit'].get('optimiser', 'optuna').lower() != 'gradient':
        print("Config fit.optimiser != 'gradient'; skipping gradient descent. Set to 'gradient' to enable.")
        return
    sensor_names, time_grid, data = load_data(cfg, args.config)
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
    integrator = cfg['fit'].get('integrator', 'euler')

    lr = cfg['fit']['gradient']['learning_rate']
    max_iters = cfg['fit']['gradient']['max_iters']
    grad_eps = cfg['fit']['gradient']['grad_eps']
    tol_grad_norm = cfg['fit']['gradient']['tol_grad_norm']
    print_every = cfg['fit']['gradient']['print_every']

    best_theta = theta.copy()
    gp, sensors = build_params_from_theta(theta, cfg, sensor_names)
    solver = create_solver(gp, sensors, backend=backend, integrator=integrator)
    best_loss = mse_loss(solver, cfg, sensor_names, time_grid, data, solver_time_grid)

    for it in range(1, max_iters + 1):
        gp, sensors = build_params_from_theta(theta, cfg, sensor_names)
        solver = create_solver(gp, sensors, backend=backend, integrator=integrator)
        loss_val = mse_loss(solver, cfg, sensor_names, time_grid, data, solver_time_grid)
        grad = np.zeros_like(theta)
        base_loss = loss_val
        for i in range(len(theta)):
            dtheta = np.zeros_like(theta)
            dtheta[i] = grad_eps
            gp_p, sensors_p = build_params_from_theta(theta + dtheta, cfg, sensor_names)
            solver_p = create_solver(gp_p, sensors_p, backend=backend, integrator=integrator)
            f_plus = mse_loss(solver_p, cfg, sensor_names, time_grid, data, solver_time_grid)
            gp_m, sensors_m = build_params_from_theta(theta - dtheta, cfg, sensor_names)
            solver_m = create_solver(gp_m, sensors_m, backend=backend, integrator=integrator)
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


if __name__ == '__main__':
    main()
