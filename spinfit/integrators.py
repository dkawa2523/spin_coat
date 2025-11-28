from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict

from .types import IntegratorName

try:
    import jax.numpy as jnp
    from jax import lax
except ImportError:  # pragma: no cover - optional dependency
    jnp = None
    lax = None


IntegratorFn = Callable[..., object]


@dataclass
class IntegratorGroup:
    numpy: IntegratorFn
    jax: IntegratorFn | None = None


def euler_numpy(rhs, h0, times):
    n = len(times)
    h = np.zeros_like(times, dtype=float)
    h[0] = h0
    for i in range(1, n):
        dt = times[i] - times[i - 1]
        h[i] = h[i - 1] + rhs(h[i - 1], times[i - 1]) * dt
    return h


def euler_jax(rhs, h0, times):
    dt = times[1:] - times[:-1]
    t_prev = times[:-1]

    def step(carry, inputs):
        h_prev = carry
        dt_i, t_i = inputs
        h_new = h_prev + rhs(h_prev, t_i) * dt_i
        return h_new, h_new

    _, hs = lax.scan(step, h0, (dt, t_prev))
    return jnp.concatenate([jnp.asarray([h0]), hs])


def rk4_numpy(rhs, h0, times):
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


def rk4_jax(rhs, h0, times):
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

    _, hs = lax.scan(step, h0, (dt, t_prev))
    return jnp.concatenate([jnp.asarray([h0]), hs])


def rk23_numpy(rhs, h0, times, rtol=1e-5, atol=1e-9):
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


def rk23_jax(rhs, h0, times, rtol=1e-5, atol=1e-9):
    # Pure Python loop for adaptivity (not JIT-friendly but works with jax numpy ops)
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


def semi_implicit_numpy(rhs, h0, times, max_iter=8, tol=1e-8):
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


def semi_implicit_jax(rhs, h0, times, max_iter=8):
    dt = times[1:] - times[:-1]
    t_next = times[1:]

    def step(carry, inputs):
        h_prev = carry
        dt_i, t_i = inputs

        def body_fn(i, val):
            return h_prev + dt_i * rhs(val, t_i)

        h_new = lax.fori_loop(0, max_iter, body_fn, h_prev)
        return h_new, h_new

    _, hs = lax.scan(step, h0, (dt, t_next))
    return jnp.concatenate([jnp.asarray([h0]), hs])


INTEGRATORS: Dict[IntegratorName, IntegratorGroup] = {
    "euler": IntegratorGroup(numpy=euler_numpy, jax=euler_jax),
    "rk4": IntegratorGroup(numpy=rk4_numpy, jax=rk4_jax),
    "rk23": IntegratorGroup(numpy=rk23_numpy, jax=rk23_jax),
    "semi_implicit": IntegratorGroup(numpy=semi_implicit_numpy, jax=semi_implicit_jax),
}


def get_integrator_group(name: IntegratorName) -> IntegratorGroup:
    key = name.lower()
    if key not in INTEGRATORS:
        raise ValueError(f"Unknown integrator '{name}'. Available: {list(INTEGRATORS.keys())}")
    return INTEGRATORS[key]
