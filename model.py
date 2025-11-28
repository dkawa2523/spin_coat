"""Core model definitions for the spin coating film thickness project.

This module defines the ODE describing the time evolution of film thickness
according to a simplified extension of the Meyerhofer–EBP model.  It also
implements utilities for parameter packing/unpacking, two numerical solvers
for NumPy and JAX backends, a simple mean squared error loss, and a
gradient descent optimiser.  The aim is to keep all model‑dependent code
inside this file so that adding new physics only requires changes here.
"""

from __future__ import annotations

import numpy as np
import yaml
from typing import Dict, List, Tuple, Iterable, Callable, Optional

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
except ImportError:
    # JAX is optional; if unavailable, jax-related functionality will raise
    jax = None
    jnp = None
    lax = None


class GlobalParams:
    """Container for parameters shared across all sensors."""
    def __init__(
        self,
        rho: float,
        omega: float,
        h_ref: float,
        mu0: float,
        m: float,
        E0: float,
        alpha_E: float,
    ) -> None:
        self.rho = rho
        self.omega = omega
        self.h_ref = h_ref
        self.mu0 = mu0
        self.m = m
        self.E0 = E0
        self.alpha_E = alpha_E


class SensorParams:
    """Container for parameters specific to a single sensor location."""
    def __init__(self, k_flow: float, delta_E: float) -> None:
        self.k_flow = k_flow
        self.delta_E = delta_E


def mu_model(h: np.ndarray, gp: GlobalParams, xp) -> np.ndarray:
    """Return viscosity μ(h) as a function of thickness.

    The model is μ(h) = μ0 * (h / h_ref)^(-m).  When h is extremely small,
    values are clipped to avoid division by zero.

    Parameters
    ----------
    h : array-like
        The film thickness (m).  Can be scalar or vector.
    gp : GlobalParams
        Global model parameters.
    xp : module
        Either numpy or jax.numpy.  This allows the same function to work
        for both backends.

    Returns
    -------
    ndarray
        The viscosity evaluated at each h.
    """
    h_safe = xp.clip(h, 1e-20, None)
    return gp.mu0 * (h_safe / gp.h_ref) ** (-gp.m)


def evap_model(h: np.ndarray, sp: SensorParams, gp: GlobalParams, xp) -> np.ndarray:
    """Return evaporation rate E(h) for a given sensor.

    The default model implements E_i(h) = (1 + delta_E_i) * E0 * (h / h_ref)^(alpha_E).
    When alpha_E = 0, evaporation is constant.  This is a very simple model
    intended to be replaced or extended as needed.
    """
    h_safe = xp.clip(h, 1e-20, None)
    base = (1.0 + sp.delta_E) * gp.E0
    if abs(gp.alpha_E) > 1e-12:
        base = base * (h_safe / gp.h_ref) ** gp.alpha_E
    return base


def rhs_ode(h: np.ndarray, t: float, sp: SensorParams, gp: GlobalParams, xp) -> np.ndarray:
    """Right hand side of the ODE dh/dt for a single sensor.

    Parameters
    ----------
    h : array-like
        Current film thickness (m).
    t : float
        Current time (s).  Not used in this model but included for API completeness.
    sp : SensorParams
        Sensor-specific parameters.
    gp : GlobalParams
        Global parameters.
    xp : module
        Either numpy or jax.numpy.

    Returns
    -------
    ndarray
        The time derivative dh/dt at thickness h.
    """
    # Effective flow coefficient for this sensor
    C_i = sp.k_flow * (2.0 * gp.rho * (gp.omega ** 2) / 3.0)
    mu_h = mu_model(h, gp, xp)
    evap = evap_model(h, sp, gp, xp)
    return -C_i * (h ** 3) / mu_h - evap


def integrate_ode_explicit(
    rhs: Callable[[np.ndarray, float], np.ndarray],
    h0: float,
    times: np.ndarray,
    xp,
) -> np.ndarray:
    """Integrate an ODE using a simple forward Euler method.

    This function accepts a right-hand-side function `rhs(h, t)` and an
    array of times.  It returns an array of the same shape as times
    containing the solution.  The time step is inferred from successive
    entries in `times`.

    Parameters
    ----------
    rhs : callable
        Function returning dh/dt.
    h0 : float
        Initial thickness.
    times : ndarray
        Strictly increasing array of times at which to integrate.
    xp : module
        numpy or jax.numpy for arithmetic.

    Returns
    -------
    ndarray
        Thickness values at each time point.
    """
    n = len(times)
    h = xp.zeros_like(times)
    h = h.astype(xp.float64)
    h = h.at[0].set(h0) if hasattr(h, 'at') else (h.__setitem__(0, h0) or h)
    # Use explicit loops for clarity.  JAX can JIT compile the loop.
    for i in range(1, n):
        dt = times[i] - times[i - 1]
        dh = rhs(h[i - 1], times[i - 1]) * dt
        h = h.at[i].set(h[i - 1] + dh) if hasattr(h, 'at') else (h.__setitem__(i, h[i - 1] + dh) or h)
    return h


def integrate_ode_explicit_jax(
    rhs: Callable[[jnp.ndarray, float], jnp.ndarray],
    h0: float,
    times: jnp.ndarray,
) -> jnp.ndarray:
    """JAX-specific forward Euler using lax.scan for speed and JIT compatibility."""
    dt = times[1:] - times[:-1]
    t_prev = times[:-1]

    def step(carry, inputs):
        h_prev = carry
        dt_i, t_i = inputs
        dh = rhs(h_prev, t_i) * dt_i
        h_new = h_prev + dh
        return h_new, h_new

    h_last, hs = lax.scan(step, jnp.asarray(h0), (dt, t_prev))
    return jnp.concatenate([jnp.asarray([h0]), hs])


def integrate_ode_rk4_numpy(
    rhs: Callable[[np.ndarray, float], np.ndarray],
    h0: float,
    times: np.ndarray,
) -> np.ndarray:
    """Classic RK4 for NumPy backend."""
    n = len(times)
    h = np.zeros_like(times, dtype=float)
    h[0] = h0
    for i in range(1, n):
        dt = times[i] - times[i - 1]
        t = times[i - 1]
        k1 = rhs(h[i - 1], t)
        k2 = rhs(h[i - 1] + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = rhs(h[i - 1] + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = rhs(h[i - 1] + dt * k3, t + dt)
        h[i] = h[i - 1] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return h


def integrate_ode_rk4_jax(
    rhs: Callable[[jnp.ndarray, float], jnp.ndarray],
    h0: float,
    times: jnp.ndarray,
) -> jnp.ndarray:
    """Classic RK4 for JAX backend using lax.scan."""
    dt = times[1:] - times[:-1]
    t_prev = times[:-1]

    def step(carry, inputs):
        h_prev = carry
        dt_i, t_i = inputs
        k1 = rhs(h_prev, t_i)
        k2 = rhs(h_prev + 0.5 * dt_i * k1, t_i + 0.5 * dt_i)
        k3 = rhs(h_prev + 0.5 * dt_i * k2, t_i + 0.5 * dt_i)
        k4 = rhs(h_prev + dt_i * k3, t_i + dt_i)
        h_new = h_prev + (dt_i / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return h_new, h_new

    _, hs = lax.scan(step, jnp.asarray(h0), (dt, t_prev))
    return jnp.concatenate([jnp.asarray([h0]), hs])


def integrate_ode_rk23_adaptive_numpy(
    rhs: Callable[[np.ndarray, float], np.ndarray],
    h0: float,
    times: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-9,
) -> np.ndarray:
    """Adaptive RK23 (Bogacki–Shampine) for scalar ODE."""
    y = float(h0)
    t_curr = float(times[0])
    out = np.zeros_like(times, dtype=float)
    out[0] = y
    dt = (times[1] - times[0]) if len(times) > 1 else 1e-3
    for idx in range(1, len(times)):
        t_target = float(times[idx])
        while t_curr < t_target:
            dt = min(dt, t_target - t_curr)
            k1 = rhs(y, t_curr)
            k2 = rhs(y + 0.5 * dt * k1, t_curr + 0.5 * dt)
            k3 = rhs(y + 0.75 * dt * k2, t_curr + 0.75 * dt)
            y3 = y + dt * (2.0 / 9.0 * k1 + 1.0 / 3.0 * k2 + 4.0 / 9.0 * k3)
            k4 = rhs(y3, t_curr + dt)
            y2 = y + dt * (7.0 / 24.0 * k1 + 0.25 * k2 + 1.0 / 3.0 * k3 + 1.0 / 8.0 * k4)
            err = abs(y3 - y2)
            tol = atol + rtol * max(abs(y), abs(y3))
            if err <= tol or dt <= 1e-12:
                t_curr += dt
                y = y3
                fac = 0.9 * (tol / max(err, 1e-16)) ** (1.0 / 3.0)
                dt = dt * min(2.0, max(0.2, fac))
            else:
                fac = 0.9 * (tol / err) ** (1.0 / 3.0)
                dt = dt * max(0.2, min(1.0, fac))
        out[idx] = y
    return out


def integrate_ode_rk23_adaptive_jax(
    rhs: Callable[[jnp.ndarray, float], jnp.ndarray],
    h0: float,
    times: jnp.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-9,
) -> jnp.ndarray:
    """Adaptive RK23 in pure Python/JAX ops (not JIT)."""
    y = float(h0)
    t_curr = float(times[0])
    out = [y]
    dt = float((times[1] - times[0]) if times.shape[0] > 1 else 1e-3)
    for idx in range(1, times.shape[0]):
        t_target = float(times[idx])
        while t_curr < t_target:
            dt = min(dt, t_target - t_curr)
            k1 = float(rhs(y, t_curr))
            k2 = float(rhs(y + 0.5 * dt * k1, t_curr + 0.5 * dt))
            k3 = float(rhs(y + 0.75 * dt * k2, t_curr + 0.75 * dt))
            y3 = y + dt * (2.0 / 9.0 * k1 + 1.0 / 3.0 * k2 + 4.0 / 9.0 * k3)
            k4 = float(rhs(y3, t_curr + dt))
            y2 = y + dt * (7.0 / 24.0 * k1 + 0.25 * k2 + 1.0 / 3.0 * k3 + 1.0 / 8.0 * k4)
            err = abs(y3 - y2)
            tol = atol + rtol * max(abs(y), abs(y3))
            if err <= tol or dt <= 1e-12:
                t_curr += dt
                y = y3
                fac = 0.9 * (tol / max(err, 1e-16)) ** (1.0 / 3.0)
                dt = dt * min(2.0, max(0.2, fac))
            else:
                fac = 0.9 * (tol / err) ** (1.0 / 3.0)
                dt = dt * max(0.2, min(1.0, fac))
        out.append(y)
    return jnp.asarray(out)


def integrate_ode_semi_implicit_numpy(
    rhs: Callable[[np.ndarray, float], np.ndarray],
    h0: float,
    times: np.ndarray,
    max_iter: int = 8,
    tol: float = 1e-8,
) -> np.ndarray:
    """Simple fixed-point semi-implicit (backward-Euler-like) update."""
    n = len(times)
    h = np.zeros_like(times, dtype=float)
    h[0] = h0
    for i in range(1, n):
        dt = times[i] - times[i - 1]
        h_prev = h[i - 1]
        h_new = h_prev
        for _ in range(max_iter):
            next_val = h_prev + dt * rhs(h_new, times[i])
            if abs(next_val - h_new) < tol:
                h_new = next_val
                break
            h_new = next_val
        h[i] = h_new
    return h


def integrate_ode_semi_implicit_jax(
    rhs: Callable[[jnp.ndarray, float], jnp.ndarray],
    h0: float,
    times: jnp.ndarray,
    max_iter: int = 8,
) -> jnp.ndarray:
    """Fixed-iteration semi-implicit step for JAX."""
    dt = times[1:] - times[:-1]
    t_next = times[1:]

    def step(carry, inputs):
        h_prev = carry
        dt_i, t_i = inputs

        def body_fn(i, val):
            return h_prev + dt_i * rhs(val, t_i)

        h_new = lax.fori_loop(0, max_iter, body_fn, h_prev)
        return h_new, h_new

    _, hs = lax.scan(step, jnp.asarray(h0), (dt, t_next))
    return jnp.concatenate([jnp.asarray([h0]), hs])


def integrate_ode_rk4_numpy(
    rhs: Callable[[np.ndarray, float], np.ndarray],
    h0: float,
    times: np.ndarray,
) -> np.ndarray:
    """Classic RK4 for NumPy backend."""
    n = len(times)
    h = np.zeros_like(times, dtype=float)
    h[0] = h0
    for i in range(1, n):
        dt = times[i] - times[i - 1]
        t = times[i - 1]
        k1 = rhs(h[i - 1], t)
        k2 = rhs(h[i - 1] + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = rhs(h[i - 1] + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = rhs(h[i - 1] + dt * k3, t + dt)
        h[i] = h[i - 1] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return h


def integrate_ode_rk4_jax(
    rhs: Callable[[jnp.ndarray, float], jnp.ndarray],
    h0: float,
    times: jnp.ndarray,
) -> jnp.ndarray:
    """Classic RK4 for JAX backend using lax.scan."""
    dt = times[1:] - times[:-1]
    t_prev = times[:-1]

    def step(carry, inputs):
        h_prev = carry
        dt_i, t_i = inputs
        k1 = rhs(h_prev, t_i)
        k2 = rhs(h_prev + 0.5 * dt_i * k1, t_i + 0.5 * dt_i)
        k3 = rhs(h_prev + 0.5 * dt_i * k2, t_i + 0.5 * dt_i)
        k4 = rhs(h_prev + dt_i * k3, t_i + dt_i)
        h_new = h_prev + (dt_i / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return h_new, h_new

    _, hs = lax.scan(step, jnp.asarray(h0), (dt, t_prev))
    return jnp.concatenate([jnp.asarray([h0]), hs])


class NumpySpinODESolver:
    """Solver for the spin coating ODE using NumPy."""

    def __init__(self, gp: GlobalParams, sensors: Dict[str, SensorParams], integrator: str = "euler"):
        self.gp = gp
        self.sensors = sensors
        self.integrator = integrator.lower()

    def solve(self, sensor_name: str, times: np.ndarray, h0: float) -> np.ndarray:
        sp = self.sensors[sensor_name]
        rhs = lambda h, t: rhs_ode(h, t, sp, self.gp, np)
        if self.integrator == "rk4":
            return integrate_ode_rk4_numpy(rhs, h0, times)
        if self.integrator == "rk23":
            return integrate_ode_rk23_adaptive_numpy(rhs, h0, times)
        if self.integrator == "semi_implicit":
            return integrate_ode_semi_implicit_numpy(rhs, h0, times)
        return integrate_ode_explicit(rhs, h0, times, np)


class JaxSpinODESolver:
    """Solver for the spin coating ODE using JAX.

    Note that this implementation is written to fall back to NumPy if JAX
    is not installed.  When JAX is present, the solve method returns a
    JAX array and can be JIT compiled for speed.
    """

    def __init__(self, gp: GlobalParams, sensors: Dict[str, SensorParams], integrator: str = "euler"):
        if jax is None:
            raise ImportError(
                "JAX is not installed; please install jax and jaxlib to use the JAX solver."
            )
        self.gp = gp
        self.sensors = sensors
        self.integrator = integrator.lower()

    def solve(self, sensor_name: str, times: np.ndarray, h0: float) -> jnp.ndarray:
        sp = self.sensors[sensor_name]
        # Capture gp, sp in closures for jax
        def rhs(h, t):
            return rhs_ode(h, t, sp, self.gp, jnp)

        times_j = jnp.asarray(times)
        h0_j = jnp.asarray(h0)
        if self.integrator == "rk4":
            return integrate_ode_rk4_jax(rhs, h0_j, times_j)
        if self.integrator == "rk23":
            return integrate_ode_rk23_adaptive_jax(rhs, h0_j, times_j)
        if self.integrator == "semi_implicit":
            return integrate_ode_semi_implicit_jax(rhs, h0_j, times_j)
        return integrate_ode_explicit_jax(rhs, h0_j, times_j)


def pack_theta(params: Dict[str, Dict[str, float]], sensor_names: List[str]) -> np.ndarray:
    """Pack parameter dictionaries into a flat vector.

    The ordering of the vector is:
      [log10_mu0, m, log10_E0, alpha_E,
       k_flow_sensor1, ..., k_flow_sensorN,
       delta_E_sensor1, ..., delta_E_sensorN]

    Parameters
    ----------
    params : dict
        Contains 'global' and 'per_sensor' keys with parameter values.
    sensor_names : list of str
        Names of sensors in the order to pack parameters.

    Returns
    -------
    ndarray
        Flattened parameter vector.
    """
    g = params['global']
    p = params['per_sensor']
    vec = [
        g['log10_mu0'],
        g['m'],
        g['log10_E0'],
        g.get('alpha_E', 0.0),
    ]
    # k_flow for each sensor
    for name in sensor_names:
        vec.append(p['k_flow'][name])
    # delta_E for each sensor
    for name in sensor_names:
        vec.append(p['delta_E'][name])
    return np.array(vec, dtype=float)


def unpack_theta(theta: np.ndarray, sensor_names: List[str]) -> Dict[str, Dict[str, float]]:
    """Unpack a flat vector into parameter dictionaries.

    Returns a nested dictionary with 'global' and 'per_sensor' keys.  See
    `pack_theta` for the ordering.
    """
    g = {
        'log10_mu0': float(theta[0]),
        'm': float(theta[1]),
        'log10_E0': float(theta[2]),
        'alpha_E': float(theta[3]),
    }
    n = len(sensor_names)
    k_flows = theta[4 : 4 + n]
    delta_Es = theta[4 + n : 4 + 2 * n]
    p = {
        'k_flow': {name: float(k_flows[i]) for i, name in enumerate(sensor_names)},
        'delta_E': {name: float(delta_Es[i]) for i, name in enumerate(sensor_names)},
    }
    return {'global': g, 'per_sensor': p}


def build_params_from_theta(theta: np.ndarray, cfg: Dict, sensor_names: List[str]) -> Tuple[GlobalParams, Dict[str, SensorParams]]:
    """Construct GlobalParams and per-sensor SensorParams objects from theta and config.

    The config provides constant values like rho and omega; theta provides
    variable parameters such as mu0, m, E0, etc.  log10_mu0 and log10_E0
    are exponentiated here to produce μ0 and E0.
    """
    unpacked = unpack_theta(theta, sensor_names)
    g_var = unpacked['global']
    p_var = unpacked['per_sensor']
    # Retrieve constants from config
    rho = cfg['model']['rho']
    omega = cfg['model']['omega']
    h_ref = cfg['model']['h_ref']
    # Convert log10 values to linear scale
    mu0 = 10.0 ** g_var['log10_mu0']
    E0 = 10.0 ** g_var['log10_E0']
    m = g_var['m']
    alpha_E = g_var.get('alpha_E', cfg['model']['evap_model'].get('alpha_E', 0.0))
    gp = GlobalParams(rho, omega, h_ref, mu0, m, E0, alpha_E)
    sensors = {}
    for name in sensor_names:
        k_flow = p_var['k_flow'][name]
        delta_E = p_var['delta_E'][name]
        sensors[name] = SensorParams(k_flow, delta_E)
    return gp, sensors


def mse_loss(
    theta: np.ndarray,
    cfg: Dict,
    sensor_names: List[str],
    time_grid: Dict[str, np.ndarray],
    data: Dict[str, np.ndarray],
    backend: str = 'numpy',
    solver_time_grid: Optional[Dict[str, np.ndarray]] = None,
) -> float:
    """Compute the mean squared error across all sensors for a given theta.

    Parameters
    ----------
    theta : ndarray
        Flattened parameter vector.
    cfg : dict
        Configuration dictionary (loaded from YAML).
    sensor_names : list of str
        Names of sensors.
    time_grid : dict
        Maps sensor names to arrays of time points at which to compute h(t).
    data : dict
        Maps sensor names to arrays of measured thickness values.
    backend : str
        'numpy' or 'jax'.  JAX backend requires jax to be installed.
    solver_time_grid : dict or None
        Optional solver grid; when provided, predictions are interpolated back
        to the measurement grid before computing the residuals.

    Returns
    -------
    float
        Mean squared error across all sensors and time points.
    """
    gp, sensors = build_params_from_theta(theta, cfg, sensor_names)
    loss = 0.0
    total = 0
    backend = backend.lower()
    integrator = cfg['fit'].get('integrator', 'euler')
    if backend == 'jax':
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but JAX is not installed.")
        solver = JaxSpinODESolver(gp, sensors, integrator=integrator)
    elif backend == 'numpy':
        solver = NumpySpinODESolver(gp, sensors, integrator=integrator)
    else:
        raise ValueError(f"Unsupported backend '{backend}'. Use 'numpy' or 'jax'.")
    for name in sensor_names:
        meas_times = time_grid[name]
        meas = data[name]
        times = solver_time_grid[name] if solver_time_grid else meas_times
        if cfg['fit']['initial_h']['mode'] == 'from_data':
            h0 = float(meas[0])
        else:
            h0 = float(cfg['fit']['initial_h']['fixed_value'])
        pred = solver.solve(name, times, h0)
        # Ensure pred is NumPy array for loss calculation
        pred_np = np.asarray(pred, dtype=float)
        if solver_time_grid is not None:
            pred_np = np.interp(meas_times, times, pred_np)
        if not np.all(np.isfinite(pred_np)):
            return 1e30
        # MSE for this sensor
        residual = pred_np - meas
        loss += float(np.sum(residual ** 2))
        total += len(meas)
    # Continuity regularisation (optional)
    cont_cfg = cfg.get('fit', {}).get('continuity', {})
    if cont_cfg.get('enabled', False) and cont_cfg.get('weight', 0.0) > 0.0:
        order = int(cont_cfg.get('order', 1))
        dt_scale = float(cont_cfg.get('dt_scale', 1.0))
        for name in sensor_names:
            times = solver_time_grid[name] if solver_time_grid else time_grid[name]
            meas = data[name]
            pred = solver.solve(name, times, float(meas[0] if cfg['fit']['initial_h']['mode'] == 'from_data' else cfg['fit']['initial_h']['fixed_value']))
            pred_np = np.asarray(pred, dtype=float)
            if solver_time_grid is not None:
                pred_np = np.interp(time_grid[name], times, pred_np)
                times = time_grid[name]
            dt = np.diff(times) * dt_scale
            if order == 1:
                diff = np.diff(pred_np) / dt
            else:
                diff = np.diff(pred_np, n=2) / (dt[:-1] * dt[1:])
            loss += float(cont_cfg['weight'] * np.sum(diff ** 2))
    result = loss / total
    if not np.isfinite(result):
        return 1e30
    return result


class GradientDescentOptimizer:
    """Simple finite-difference gradient descent optimiser for theta.

    Attributes are passed through YAML configuration for easy tuning.  This
    optimiser is mainly for demonstration; for serious use, consider more
    sophisticated optimisers (e.g. Adam in JAX or PyTorch).
    """

    def __init__(
        self,
        cfg: Dict,
        sensor_names: List[str],
        time_grid: Dict[str, np.ndarray],
        solver_time_grid: Dict[str, np.ndarray],
        data: Dict[str, np.ndarray],
    ) -> None:
        self.cfg = cfg
        self.sensor_names = sensor_names
        self.time_grid = time_grid
        self.solver_time_grid = solver_time_grid
        self.data = data
        # Extract gradient settings
        self.lr = cfg['fit']['gradient']['learning_rate']
        self.max_iters = cfg['fit']['gradient']['max_iters']
        self.grad_eps = cfg['fit']['gradient']['grad_eps']
        self.tol_grad_norm = cfg['fit']['gradient']['tol_grad_norm']
        self.print_every = cfg['fit']['gradient']['print_every']

    def compute_gradient(self, theta: np.ndarray, f0: float = None) -> np.ndarray:
        """Compute gradient of the loss function via finite differences."""
        grad = np.zeros_like(theta)
        base_loss = f0 if f0 is not None else mse_loss(
            theta,
            self.cfg,
            self.sensor_names,
            self.time_grid,
            self.data,
            self.cfg['fit']['backend'],
            self.solver_time_grid,
        )
        # Finite difference for each parameter
        for i in range(len(theta)):
            dtheta = np.zeros_like(theta)
            dtheta[i] = self.grad_eps
            f_plus = mse_loss(
                theta + dtheta,
                self.cfg,
                self.sensor_names,
                self.time_grid,
                self.data,
                self.cfg['fit']['backend'],
                self.solver_time_grid,
            )
            f_minus = mse_loss(
                theta - dtheta,
                self.cfg,
                self.sensor_names,
                self.time_grid,
                self.data,
                self.cfg['fit']['backend'],
                self.solver_time_grid,
            )
            grad[i] = (f_plus - f_minus) / (2.0 * self.grad_eps)
        return grad

    def optimise(self, theta_init: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run gradient descent to minimise the loss function.

        Returns the best theta found and its corresponding loss.
        """
        theta = theta_init.copy()
        best_theta = theta.copy()
        best_loss = mse_loss(
            theta,
            self.cfg,
            self.sensor_names,
            self.time_grid,
            self.data,
            self.cfg['fit']['backend'],
            self.solver_time_grid,
        )
        for it in range(1, self.max_iters + 1):
            loss_val = mse_loss(
                theta,
                self.cfg,
                self.sensor_names,
                self.time_grid,
                self.data,
                self.cfg['fit']['backend'],
                self.solver_time_grid,
            )
            grad = self.compute_gradient(theta, f0=loss_val)
            grad_norm = np.linalg.norm(grad)
            # Update
            theta = theta - self.lr * grad
            # Track best
            if loss_val < best_loss:
                best_theta = theta.copy()
                best_loss = loss_val
            if self.print_every and it % self.print_every == 0:
                print(
                    f"Iter {it:03d}: loss={loss_val:.6e}, grad_norm={grad_norm:.3e}, best_loss={best_loss:.6e}"
                )
            if grad_norm < self.tol_grad_norm:
                print(f"Gradient norm {grad_norm:.3e} below tolerance {self.tol_grad_norm}; stopping at iter {it}.")
                break
        return best_theta, best_loss


def run_optuna(
    cfg: Dict,
    sensor_names: List[str],
    time_grid: Dict[str, np.ndarray],
    data: Dict[str, np.ndarray],
    solver_time_grid: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, float, Dict[str, Dict[str, float]], "optuna.study.Study"]:
    """Run an Optuna search for the best parameters.

    The search space is defined in the config.  Returns the best theta and
    corresponding loss as well as a dictionary of unpacked parameters for convenience.
    """
    import optuna

    def suggest_params(trial: optuna.Trial) -> np.ndarray:
        # Suggest global parameters
        g_cfg = cfg['fit']['parameters']['global']
        log10_mu0 = trial.suggest_float('log10_mu0', g_cfg['log10_mu0']['optuna_low'], g_cfg['log10_mu0']['optuna_high'])
        m = trial.suggest_float('m', g_cfg['m']['optuna_low'], g_cfg['m']['optuna_high'])
        log10_E0 = trial.suggest_float('log10_E0', g_cfg['log10_E0']['optuna_low'], g_cfg['log10_E0']['optuna_high'])
        alpha_E = trial.suggest_float('alpha_E', g_cfg['alpha_E']['optuna_low'], g_cfg['alpha_E']['optuna_high'])
        # Suggest per sensor parameters
        p_cfg = cfg['fit']['parameters']['per_sensor']
        k_flows = []
        delta_Es = []
        for name in sensor_names:
            k_flows.append(
                trial.suggest_float(
                    f'k_flow_{name}', p_cfg['k_flow']['optuna_low'], p_cfg['k_flow']['optuna_high']
                )
            )
            delta_Es.append(
                trial.suggest_float(
                    f'delta_E_{name}', p_cfg['delta_E']['optuna_low'], p_cfg['delta_E']['optuna_high']
                )
            )
        theta_list = [log10_mu0, m, log10_E0, alpha_E] + k_flows + delta_Es
        return np.array(theta_list, dtype=float)

    def objective(trial: optuna.Trial) -> float:
        theta = suggest_params(trial)
        return mse_loss(
            theta,
            cfg,
            sensor_names,
            time_grid,
            data,
            backend=cfg['fit']['backend'],
            solver_time_grid=solver_time_grid,
        )

    optuna_cfg = cfg['fit']['optuna']
    sampler_name = (optuna_cfg.get('sampler') or 'tpe').lower()
    if sampler_name == 'random':
        sampler = optuna.samplers.RandomSampler()
    else:
        sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(direction=optuna_cfg['direction'], sampler=sampler)
    study.optimize(objective, n_trials=optuna_cfg.get('n_trials'), timeout=optuna_cfg.get('timeout'))
    best_theta = suggest_params(study.best_trial)
    best_loss = study.best_value
    unpacked_params = unpack_theta(best_theta, sensor_names)
    return best_theta, best_loss, unpacked_params, study


def render_equation_as_text(gp: GlobalParams, sensor: SensorParams) -> str:
    """Return a human‑readable string representation of the ODE with current parameters.

    This function is intended to help users verify or communicate the model.  It
    prints the ODE dh/dt explicitly with numerical coefficients substituted.
    """
    C_i = sensor.k_flow * (2.0 * gp.rho * gp.omega ** 2 / 3.0)
    base = (1.0 + sensor.delta_E) * gp.E0
    return (
        f"dh/dt = -( {C_i:.3e} ) * h^3 / μ(h) - ( {base:.3e} ) * (h/{gp.h_ref:.3e})^{gp.alpha_E:.3f}\n"
        f"μ(h) = {gp.mu0:.3e} * (h/{gp.h_ref:.3e})^-{gp.m:.3f}"
    )
