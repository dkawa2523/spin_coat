from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

from .solvers import SpinSolver


def mse_loss(
    solver: SpinSolver,
    cfg: Dict,
    sensor_names: List[str],
    time_grid: Dict[str, np.ndarray],
    data: Dict[str, np.ndarray],
    solver_time_grid: Optional[Dict[str, np.ndarray]] = None,
) -> float:
    loss = 0.0
    total = 0
    for name in sensor_names:
        meas_times = time_grid[name]
        meas = data[name]
        times = solver_time_grid[name] if solver_time_grid else meas_times
        if cfg['fit']['initial_h']['mode'] == 'from_data':
            h0 = float(meas[0])
        else:
            h0 = float(cfg['fit']['initial_h']['fixed_value'])
        pred = solver.solve(name, times, h0)
        pred_np = np.asarray(pred, dtype=float)
        if solver_time_grid is not None:
            pred_np = np.interp(meas_times, times, pred_np)
        if not np.all(np.isfinite(pred_np)):
            return 1e30
        residual = pred_np - meas
        loss += float(np.sum(residual ** 2))
        total += len(meas)

    cont_cfg = cfg.get('fit', {}).get('continuity', {})
    if cont_cfg.get('enabled', False) and cont_cfg.get('weight', 0.0) > 0.0:
        order = int(cont_cfg.get('order', 1))
        dt_scale = float(cont_cfg.get('dt_scale', 1.0))
        for name in sensor_names:
            times = solver_time_grid[name] if solver_time_grid else time_grid[name]
            meas = data[name]
            h0 = float(meas[0] if cfg['fit']['initial_h']['mode'] == 'from_data' else cfg['fit']['initial_h']['fixed_value'])
            pred = solver.solve(name, times, h0)
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

    result = loss / max(total, 1)
    if not np.isfinite(result):
        return 1e30
    return result
