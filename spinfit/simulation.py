from __future__ import annotations

import numpy as np
from typing import Dict, List

from .solvers import SpinSolver


def simulate_sensors(
    solver: SpinSolver,
    sensor_names: List[str],
    time_grid: Dict[str, np.ndarray],
    data: Dict[str, np.ndarray],
    cfg: Dict,
    solver_time_grid: Dict[str, np.ndarray] | None = None,
) -> Dict[str, np.ndarray]:
    results: Dict[str, np.ndarray] = {}
    for name in sensor_names:
        times = solver_time_grid[name] if solver_time_grid else time_grid[name]
        if cfg['forward']['initial_h']['mode'] == 'from_data':
            h0 = float(data[name][0])
        else:
            h0 = float(cfg['forward']['initial_h']['fixed_value'])
        results[name] = solver.solve(name, times, h0)
    return results
