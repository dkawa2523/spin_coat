from __future__ import annotations

import numpy as np
from typing import Callable, Dict

from .equations import rhs_ode
from .integrators import get_integrator_group
from .types import GlobalParams, SensorParams, Backend, IntegratorName

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = None
    jnp = None


class SpinSolver:
    """Backend-agnostic spin ODE solver with pluggable integrators and equations."""

    def __init__(
        self,
        gp: GlobalParams,
        sensors: Dict[str, SensorParams],
        backend: Backend = "numpy",
        integrator: IntegratorName = "euler",
        rhs_fn: Callable = rhs_ode,
    ):
        self.gp = gp
        self.sensors = sensors
        self.backend = backend.lower()
        self.integrator_name = integrator
        self.rhs_fn = rhs_fn
        self.integrators = get_integrator_group(integrator)

        if self.backend == "jax":
            if jax is None or jnp is None:
                raise RuntimeError("JAX backend requested but JAX is not installed.")
            self.xp = jnp
        else:
            self.xp = np

    def solve(self, sensor_name: str, times, h0: float):
        sp = self.sensors[sensor_name]
        xp = self.xp
        rhs = lambda h, t: self.rhs_fn(h, t, sp, self.gp, xp)
        if self.backend == "jax":
            integrator = self.integrators.jax
            return integrator(rhs, xp.asarray(h0), xp.asarray(times))  # type: ignore
        integrator = self.integrators.numpy
        return integrator(rhs, h0, np.asarray(times, dtype=float))


def create_solver(
    gp: GlobalParams,
    sensors: Dict[str, SensorParams],
    backend: Backend,
    integrator: IntegratorName,
    rhs_fn: Callable = rhs_ode,
) -> SpinSolver:
    return SpinSolver(gp, sensors, backend=backend, integrator=integrator, rhs_fn=rhs_fn)
