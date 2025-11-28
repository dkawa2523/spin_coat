from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

from .types import GlobalParams, SensorParams


def pack_theta(params: Dict[str, Dict[str, float]], sensor_names: List[str]) -> np.ndarray:
    g = params['global']
    p = params['per_sensor']
    vec = [
        g['log10_mu0'],
        g['m'],
        g['log10_E0'],
        g.get('alpha_E', 0.0),
    ]
    for name in sensor_names:
        vec.append(p['k_flow'][name])
    for name in sensor_names:
        vec.append(p['delta_E'][name])
    return np.array(vec, dtype=float)


def unpack_theta(theta: np.ndarray, sensor_names: List[str]) -> Dict[str, Dict[str, float]]:
    g = {
        'log10_mu0': float(theta[0]),
        'm': float(theta[1]),
        'log10_E0': float(theta[2]),
        'alpha_E': float(theta[3]),
    }
    n = len(sensor_names)
    k_flows = theta[4: 4 + n]
    delta_Es = theta[4 + n: 4 + 2 * n]
    p = {
        'k_flow': {name: float(k_flows[i]) for i, name in enumerate(sensor_names)},
        'delta_E': {name: float(delta_Es[i]) for i, name in enumerate(sensor_names)},
    }
    return {'global': g, 'per_sensor': p}


def build_params_from_theta(theta: np.ndarray, cfg: Dict, sensor_names: List[str]) -> Tuple[GlobalParams, Dict[str, SensorParams]]:
    unpacked = unpack_theta(theta, sensor_names)
    g_var = unpacked['global']
    p_var = unpacked['per_sensor']
    rho = cfg['model']['rho']
    omega = cfg['model']['omega']
    h_ref = cfg['model']['h_ref']
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
