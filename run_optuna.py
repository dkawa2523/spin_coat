"""Command-line interface to run Optuna optimisation on spin coating data."""

import argparse

from io_utils import (
    build_solver_time_grid,
    configure_jax_platform,
    load_config,
    load_data,
    resolve_output_dir,
)
from plot_utils import plot_predictions, save_optuna_diagnostics
from spinfit.parameters import build_params_from_theta
from spinfit.optimize import run_optuna_search
from spinfit.equations import rhs_ode
from spinfit.solvers import create_solver
from model import render_equation_as_text  # reuse text rendering


def main() -> None:
    parser = argparse.ArgumentParser(description="Spin coating parameter fitting with Optuna")
    parser.add_argument('--config', type=str, default='config_example.yaml', help='Path to the YAML config file')
    parser.add_argument('--trials', type=int, help='Override n_trials in the config for quick experiments')
    parser.add_argument('--timeout', type=float, help='Override timeout seconds in the config')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting of best-fit curves')
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
    cfg.setdefault('fit', {}).setdefault('optuna', {})
    if args.trials is not None:
        cfg['fit']['optuna']['n_trials'] = args.trials
    if args.timeout is not None:
        cfg['fit']['optuna']['timeout'] = args.timeout
    sensor_names, time_grid, data = load_data(cfg, args.config)
    solver_time_grid = build_solver_time_grid(cfg, sensor_names, time_grid)
    output_dir = resolve_output_dir(cfg, args.config)
    if cfg['fit'].get('optimiser', 'optuna').lower() != 'optuna':
        print("Config fit.optimiser != 'optuna'; skipping Optuna run. Set to 'optuna' to enable.")
        return
    theta, best_loss, params_dict, study = run_optuna_search(
        cfg, sensor_names, time_grid, data, solver_time_grid=solver_time_grid
    )
    gp, sensors = build_params_from_theta(theta, cfg, sensor_names)
    solver = create_solver(gp, sensors, backend=backend, integrator=cfg['fit'].get('integrator', 'euler'), rhs_fn=rhs_ode)
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
            params_str = ", ".join(f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in entry["params"].items())
            print(f"  #{entry['rank']} trial={entry['trial_number']} value={entry['value']:.3e} :: {params_str}")
    if not args.no_plot:
        saved = plot_predictions(cfg, sensor_names, time_grid, solver_time_grid, data, theta, output_dir=output_dir)
        print("Saved plots:")
        for path in saved:
            print(f"  {path}")


if __name__ == '__main__':
    main()
