from __future__ import annotations

import numpy as np
from typing import Any

from .types import GlobalParams, SensorParams

try:
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - optional dependency
    jnp = None


def mu_model(h: Any, gp: GlobalParams, xp) -> Any:
    """Viscosity μ(h) with safe clipping."""
    h_safe = xp.clip(h, 1e-20, None)
    return gp.mu0 * (h_safe / gp.h_ref) ** (-gp.m)


def evap_model(h: Any, sp: SensorParams, gp: GlobalParams, xp) -> Any:
    """Evaporation rate E(h) with optional thickness exponent."""
    h_safe = xp.clip(h, 1e-20, None)
    base = (1.0 + sp.delta_E) * gp.E0
    if abs(gp.alpha_E) > 1e-12:
        base = base * (h_safe / gp.h_ref) ** gp.alpha_E
    return base


def rhs_ode(h: Any, t: float, sp: SensorParams, gp: GlobalParams, xp) -> Any:
    """Right-hand-side of dh/dt."""
    C_i = sp.k_flow * (2.0 * gp.rho * (gp.omega ** 2) / 3.0)
    mu_h = mu_model(h, gp, xp)
    evap = evap_model(h, sp, gp, xp)
    return -C_i * (h ** 3) / mu_h - evap
