from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class GlobalParams:
    rho: float
    omega: float
    h_ref: float
    mu0: float
    m: float
    E0: float
    alpha_E: float


@dataclass
class SensorParams:
    k_flow: float
    delta_E: float


Backend = str  # 'numpy' or 'jax'
IntegratorName = str  # euler | rk4 | rk23 | semi_implicit
