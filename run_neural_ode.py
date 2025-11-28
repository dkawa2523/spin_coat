"""Train a hybrid Neural ODE (physical model + neural correction) using JAX."""

import argparse
import numpy as np
import jax.numpy as jnp
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from io_utils import build_solver_time_grid, configure_jax_platform, load_config, load_data
from spinfit.parameters import build_params_from_theta, pack_theta
from spinfit.neural_ode import HybridNeuralODETrainer, predict_with_hybrid, save_params_npz
from spinfit.equations import rhs_ode
from spinfit.solvers import create_solver
from spinfit.losses import mse_loss


def main():
    parser = argparse.ArgumentParser(description="Hybrid Neural ODE training (JAX)")
    parser.add_argument('--config', type=str, default='config_example.yaml', help='Path to base config YAML (data/model/common settings)')
    parser.add_argument('--train-config', type=str, help='Optional path to training-specific YAML (overrides base for training run)')
    parser.add_argument('--steps', type=int, help='Override training steps')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--enable', action='store_true', help='Force enable training regardless of config flag')
    args = parser.parse_args()

    def deep_update(base: dict, override: dict) -> dict:
        out = dict(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = deep_update(out[k], v)
            else:
                out[k] = v
        return out

    base_cfg = load_config(args.config)
    if args.train_config:
        train_cfg = load_config(args.train_config)
        cfg = deep_update(base_cfg, train_cfg)
    else:
        cfg = base_cfg
    configure_jax_platform(cfg)

    if not (cfg.get('neural_ode', {}).get('enabled', False) or args.enable):
        print("neural_ode.enabled is false; enable it in YAML or pass --enable to run training.")
        return

    sensor_names, time_grid_np, data_np = load_data(cfg, args.config)
    time_grid = {k: jnp.asarray(v) for k, v in time_grid_np.items()}
    data = {k: jnp.asarray(v) for k, v in data_np.items()}
    solver_time_grid = time_grid  # using data grid for training

    # Build base physical params from initial guesses
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
    gp, sensors = build_params_from_theta(theta, cfg, sensor_names)

    nn_cfg = cfg.get('neural_ode', {})
    trainer = HybridNeuralODETrainer(
        gp=gp,
        sensors=sensors,
        sensor_names=sensor_names,
        time_grid=time_grid,
        data=data,
        integrator=cfg['fit'].get('integrator', 'rk4'),
        lr=args.lr if args.lr is not None else nn_cfg.get('lr', 1e-3),
        steps=args.steps if args.steps is not None else nn_cfg.get('steps', 200),
        seed=nn_cfg.get('seed', 0),
        layer_sizes=nn_cfg.get('layer_sizes', [3, 32, 32, 1]),
        corr_scale_final=nn_cfg.get('corr_scale_final', 1e-7),
        warmup_steps=nn_cfg.get('warmup_steps', 200),
        clip_grad=nn_cfg.get('clip_grad', 1.0),
        reg_weight_start=nn_cfg.get('reg_weight_start', 0.5),
        reg_weight_end=nn_cfg.get('reg_weight_end', 0.05),
        target_r2=nn_cfg.get('target_r2', 0.8),
    )
    params_net, history = trainer.train()

    # Evaluate combined model loss (physical + neural) for reporting
    solver_phys = create_solver(gp, sensors, backend=cfg['fit'].get('backend', 'numpy'), integrator=cfg['fit'].get('integrator', 'rk4'), rhs_fn=rhs_ode)
    phys_loss = mse_loss(solver_phys, cfg, sensor_names, time_grid_np, data_np, solver_time_grid=None)
    print(f"Physical-only loss (MSE): {phys_loss:.3e}")

    # Predict with hybrid model using trained net
    init_h = {}
    if cfg['fit']['initial_h']['mode'] == 'from_data':
        for name in sensor_names:
            init_h[name] = float(data_np[name][0])
    else:
        for name in sensor_names:
            init_h[name] = float(cfg['fit']['initial_h']['fixed_value'])

    preds_hybrid = predict_with_hybrid(
        params_net,
        gp,
        sensors,
        sensor_names,
        time_grid,
        initial_h=init_h,
        corr_scale=nn_cfg.get('corr_scale', 1e-7),
        integrator=cfg['fit'].get('integrator', 'rk4'),
    )
    hybrid_loss = 0.0
    total = 0
    for name in sensor_names:
        pred = np.asarray(preds_hybrid[name])
        meas = data_np[name]
        hybrid_loss += float(np.sum((pred - meas) ** 2))
        total += len(meas)
    hybrid_loss /= total
    print(f"Hybrid Neural ODE loss (MSE): {hybrid_loss:.3e}")

    # Save params and plots
    out_dir = cfg['neural_ode'].get('output_dir', 'outputs_neural')
    os.makedirs(out_dir, exist_ok=True)
    params_out = cfg['neural_ode'].get('params_out', os.path.join(out_dir, "params.npz"))
    save_params_npz(params_net, params_out)
    print(f"Saved trained parameters to {params_out}")

    plt.figure()
    plt.plot(np.arange(1, len(history) + 1), history, label="train loss")
    plt.xlabel("Step")
    plt.ylabel("MSE")
    plt.title("Hybrid Neural ODE Learning Curve")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "neural_ode_learning_curve.png"), dpi=150)
    plt.close()

    for name in sensor_names:
        pred = np.asarray(preds_hybrid[name])
        meas = data_np[name]
        times = np.asarray(time_grid_np[name])
        mse = float(np.mean((pred - meas) ** 2))
        ss_res = np.sum((pred - meas) ** 2)
        ss_tot = np.sum((meas - np.mean(meas)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        # overlay
        plt.figure()
        plt.plot(times, meas, "o", markersize=4, alpha=0.7, label="data")
        plt.plot(times, pred, "-", linewidth=1.6, label="hybrid")
        plt.xlabel("Time [s]")
        plt.ylabel("Thickness [m]")
        plt.title(f"{name} (MSE={mse:.2e}, R2={r2:.3f})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_hybrid_fit.png"), dpi=150)
        plt.close()

        # scatter pred vs meas
        plt.figure()
        plt.scatter(meas, pred, s=10, alpha=0.7)
        lims = [min(meas.min(), pred.min()), max(meas.max(), pred.max())]
        plt.plot(lims, lims, "k--", linewidth=1)
        plt.xlabel("Measured h")
        plt.ylabel("Predicted h")
        plt.title(f"{name} Pred vs Meas (R2={r2:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_scatter_pred_meas.png"), dpi=150)
        plt.close()

        # residuals
        plt.figure()
        plt.plot(times, pred - meas, "o-", markersize=3, linewidth=1, alpha=0.8)
        plt.axhline(0, color="k", linestyle="--", linewidth=1)
        plt.xlabel("Time [s]")
        plt.ylabel("Residual (pred - meas)")
        plt.title(f"{name} Residuals")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_residuals.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
