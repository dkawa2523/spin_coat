from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import optax
import numpy as np

from .equations import rhs_ode
from .integrators import get_integrator_group
from .types import GlobalParams, SensorParams


def init_mlp_params(key, layer_sizes: List[int]):
    params = []
    keys = jax.random.split(key, len(layer_sizes) - 1)
    for k, (din, dout) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        w_key, b_key = jax.random.split(k)
        w = jax.random.normal(w_key, (din, dout)) * math.sqrt(2.0 / din)
        b = jnp.zeros((dout,))
        params.append((w, b))
    return params


def mlp_forward(params, x):
    h = x
    for i, (w, b) in enumerate(params):
        h = jnp.dot(h, w) + b
        if i != len(params) - 1:
            h = jnp.tanh(h)
    return h


def make_hybrid_rhs(params_net, gp: GlobalParams, sp: SensorParams, sensor_id: float, corr_scale: float):
    """Return rhs(h,t) = physical + neural correction."""
    def rhs(h, t):
        phys = rhs_ode(h, t, sp, gp, jnp)
        # features: normalized thickness, time, sensor id
        h_norm = h / gp.h_ref
        x = jnp.array([h_norm, t, sensor_id])
        corr_raw = mlp_forward(params_net, x).squeeze()
        corr = corr_scale * jnp.tanh(corr_raw)
        return phys + corr
    return rhs


@dataclass
class HybridNeuralODETrainer:
    gp: GlobalParams
    sensors: Dict[str, SensorParams]
    sensor_names: List[str]
    time_grid: Dict[str, jnp.ndarray]
    data: Dict[str, jnp.ndarray]
    integrator: str = "rk4"
    lr: float = 1e-3
    steps: int = 200
    seed: int = 0
    layer_sizes: List[int] = None
    corr_scale_final: float = 1e-7
    warmup_steps: int = 200
    clip_grad: float = 1.0
    reg_weight_start: float = 0.5
    reg_weight_end: float = 0.05
    target_r2: float = 0.8

    def train(self):
        key = jax.random.PRNGKey(self.seed)
        ls = self.layer_sizes or [3, 32, 32, 1]
        params_net = init_mlp_params(key, ls)
        opt = optax.chain(optax.clip_by_global_norm(self.clip_grad), optax.adam(self.lr))
        opt_state = opt.init(params_net)
        integrators = get_integrator_group(self.integrator)
        sensor_id_map = {name: float(i) for i, name in enumerate(self.sensor_names)}
        # Precompute physical baseline for regularisation
        phys_pred = {}
        for name in self.sensor_names:
            times = self.time_grid[name]
            meas = self.data[name]
            h0 = meas[0]
            rhs = lambda h, t: rhs_ode(h, t, self.sensors[name], self.gp, jnp)
            phys_pred[name] = integrators.jax(rhs, h0, times)

        def loss_fn(p, corr_scale_eff, reg_weight_eff):
            loss = 0.0
            total = 0
            for name in self.sensor_names:
                times = self.time_grid[name]
                meas = self.data[name]
                h0 = meas[0]
                rhs = make_hybrid_rhs(p, self.gp, self.sensors[name], sensor_id_map[name], corr_scale_eff)
                pred = integrators.jax(rhs, h0, times)
                loss += jnp.sum((pred - meas) ** 2)
                if reg_weight_eff > 0.0:
                    loss += reg_weight_eff * jnp.sum((pred - phys_pred[name]) ** 2)
                total += pred.shape[0]
            return loss / total

        def step(params, opt_state, corr_scale_eff, reg_weight_eff):
            loss, grads = jax.value_and_grad(lambda p: loss_fn(p, corr_scale_eff, reg_weight_eff))(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        history = []
        history = []
        def compute_r2(pred, meas):
            ss_res = np.sum((pred - meas) ** 2)
            ss_tot = np.sum((meas - np.mean(meas)) ** 2)
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        for i in range(1, self.steps + 1):
            warm = min(1.0, i / max(1, self.warmup_steps))
            corr_scale_eff = self.corr_scale_final * warm
            reg_weight_eff = self.reg_weight_start + (self.reg_weight_end - self.reg_weight_start) * warm
            params_net, opt_state, loss = step(params_net, opt_state, corr_scale_eff, reg_weight_eff)
            history.append(float(loss))
            if i % max(1, self.steps // 10) == 0 or i == self.steps:
                # compute mean R2 for monitoring
                r2_vals = []
                for name in self.sensor_names:
                    times = self.time_grid[name]
                    meas = np.asarray(self.data[name])
                    h0 = meas[0]
                    rhs = make_hybrid_rhs(params_net, self.gp, self.sensors[name], sensor_id_map[name], corr_scale_eff)
                    pred = np.asarray(integrators.jax(rhs, h0, times))
                    r2_vals.append(compute_r2(pred, meas))
                mean_r2 = float(np.nanmean(r2_vals))
                print(f"[Hybrid N-ODE] step {i}/{self.steps} loss={loss:.3e} meanR2={mean_r2:.3f} corr_scale={corr_scale_eff:.2e} reg={reg_weight_eff:.2e}")
                if mean_r2 >= self.target_r2:
                    print(f"Early stop: mean R2 {mean_r2:.3f} reached target {self.target_r2}")
                    break

        return params_net, history


def predict_with_hybrid(
    params_net,
    gp: GlobalParams,
    sensors: Dict[str, SensorParams],
    sensor_names: List[str],
    time_grid: Dict[str, jnp.ndarray],
    initial_h: Dict[str, float],
    corr_scale: float,
    integrator: str = "rk4",
) -> Dict[str, jnp.ndarray]:
    integrators = get_integrator_group(integrator)
    sensor_id_map = {name: float(i) for i, name in enumerate(sensor_names)}
    out = {}
    for name in sensor_names:
        times = time_grid[name]
        h0 = float(initial_h[name])
        rhs = make_hybrid_rhs(params_net, gp, sensors[name], sensor_id_map[name], corr_scale)
        out[name] = integrators.jax(rhs, h0, times)
    return out


def save_params_npz(params_net, path: str):
    to_save = {}
    for idx, (w, b) in enumerate(params_net):
        to_save[f"w_{idx}"] = np.asarray(w)
        to_save[f"b_{idx}"] = np.asarray(b)
    np.savez(path, **to_save)


def load_params_npz(path: str):
    data = np.load(path)
    params = []
    idx = 0
    while f"w_{idx}" in data and f"b_{idx}" in data:
        params.append((jnp.asarray(data[f"w_{idx}"]), jnp.asarray(data[f"b_{idx}"])))
        idx += 1
    if not params:
        raise ValueError(f"No parameters found in {path}")
    return params


class HybridSolver:
    """Hybrid solver for inference: physical RHS + trained neural correction."""

    def __init__(
        self,
        params_net,
        gp: GlobalParams,
        sensors: Dict[str, SensorParams],
        sensor_names: List[str],
        integrator: str,
        corr_scale: float,
    ):
        self.params_net = params_net
        self.gp = gp
        self.sensors = sensors
        self.sensor_names = sensor_names
        self.integrator = integrator
        self.integrators = get_integrator_group(integrator)
        self.sensor_id_map = {name: float(i) for i, name in enumerate(sensor_names)}
        self.corr_scale = corr_scale

    def solve(self, sensor_name: str, times, h0: float):
        rhs = make_hybrid_rhs(
            self.params_net,
            self.gp,
            self.sensors[sensor_name],
            self.sensor_id_map[sensor_name],
            self.corr_scale,
        )
        return self.integrators.jax(rhs, h0, jnp.asarray(times))
