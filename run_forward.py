"""Forward simulation mode: use fixed parameters and process conditions to predict h(t) at each sensor.

This is separate from optimisation; no fitting is performed.
"""

import argparse

from io_utils import build_solver_time_grid, configure_jax_platform, load_config, load_data, resolve_output_dir
from spinfit.parameters import build_params_from_theta, pack_theta
from spinfit.solvers import create_solver
from spinfit.equations import rhs_ode
from model import render_equation_as_text  # reuse equation text


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward simulate spin coating thickness with fixed parameters")
    parser.add_argument('--config', type=str, default='config_example.yaml', help='Path to the YAML config file')
    args = parser.parse_args()

    cfg = load_config(args.config)
    forward_cfg = cfg.get('forward', {})
    if not forward_cfg.get('enabled', False):
        print("Forward mode is disabled in config (forward.enabled = false). Enable it to run forward simulation.")
        return
    configure_jax_platform(cfg)

    # Load data for time grids; thickness values are not used except possibly initial_h=from_data
    sensor_names, time_grid, data = load_data(cfg, args.config)
    solver_time_grid = build_solver_time_grid(cfg, sensor_names, time_grid)
    output_dir = resolve_output_dir({'fit': {'output_dir': forward_cfg.get('output_dir', 'outputs_forward')}}, args.config)

    # Build theta from forward.fixed parameters
    fixed_params = forward_cfg['parameters']
    theta = pack_theta(fixed_params, sensor_names)

    # Apply optional model overrides
    override = forward_cfg.get('override_model', {})
    model_cfg = cfg['model'].copy()
    for key in ('rho', 'omega', 'h_ref'):
        if override.get(key) is not None:
            model_cfg[key] = override[key]
    cfg_forward = cfg.copy()
    cfg_forward['model'] = model_cfg

    # Choose backend/integrator
    backend = cfg['fit'].get('backend', 'numpy').lower()
    integrator = forward_cfg.get('integrator', cfg['fit'].get('integrator', 'euler'))
    gp, sensors = build_params_from_theta(theta, cfg_forward, sensor_names)
    solver = create_solver(gp, sensors, backend=backend, integrator=integrator, rhs_fn=rhs_ode)

    # Run simulation
    results = {}
    for name in sensor_names:
        times = solver_time_grid[name]
        if cfg['forward']['initial_h']['mode'] == 'from_data':
            h0 = float(data[name][0])
        else:
            h0 = float(cfg['forward']['initial_h']['fixed_value'])
        pred = solver.solve(name, times, h0)
        results[name] = pred

    # Print summary
    print("Forward simulation completed.\n")
    for name in sensor_names:
        print(f"Sensor: {name}")
        eqn = render_equation_as_text(gp, sensors[name])
        print(eqn)
        print(f"  h(t) start: {float(results[name][0]):.4e}, h(t) end: {float(results[name][-1]):.4e}")
        print()
    print(f"Saved outputs directory (for plots if you add them): {output_dir}")


if __name__ == '__main__':
    main()
