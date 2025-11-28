from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional

from optuna import Trial

from .parameters import build_params_from_theta, pack_theta, unpack_theta
from .solvers import create_solver
from .losses import mse_loss


def suggest_params(trial: Trial, cfg: Dict, sensor_names: List[str]) -> np.ndarray:
    g_cfg = cfg['fit']['parameters']['global']
    log10_mu0 = trial.suggest_float('log10_mu0', g_cfg['log10_mu0']['optuna_low'], g_cfg['log10_mu0']['optuna_high'])
    m = trial.suggest_float('m', g_cfg['m']['optuna_low'], g_cfg['m']['optuna_high'])
    log10_E0 = trial.suggest_float('log10_E0', g_cfg['log10_E0']['optuna_low'], g_cfg['log10_E0']['optuna_high'])
    alpha_E = trial.suggest_float('alpha_E', g_cfg['alpha_E']['optuna_low'], g_cfg['alpha_E']['optuna_high'])
    p_cfg = cfg['fit']['parameters']['per_sensor']
    k_flows = []
    delta_Es = []
    for name in sensor_names:
        k_flows.append(
            trial.suggest_float(f'k_flow_{name}', p_cfg['k_flow']['optuna_low'], p_cfg['k_flow']['optuna_high'])
        )
        delta_Es.append(
            trial.suggest_float(f'delta_E_{name}', p_cfg['delta_E']['optuna_low'], p_cfg['delta_E']['optuna_high'])
        )
    theta_list = [log10_mu0, m, log10_E0, alpha_E] + k_flows + delta_Es
    return np.array(theta_list, dtype=float)


def run_optuna_search(
    cfg: Dict,
    sensor_names: List[str],
    time_grid: Dict[str, np.ndarray],
    data: Dict[str, np.ndarray],
    solver_time_grid: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, float, Dict[str, Dict[str, float]], "optuna.study.Study"]:
    import optuna

    backend = cfg['fit']['backend']
    integrator = cfg['fit'].get('integrator', 'euler')

    def objective(trial: Trial) -> float:
        theta = suggest_params(trial, cfg, sensor_names)
        gp, sensors = build_params_from_theta(theta, cfg, sensor_names)
        solver = create_solver(gp, sensors, backend=backend, integrator=integrator)
        return mse_loss(solver, cfg, sensor_names, time_grid, data, solver_time_grid=solver_time_grid)

    optuna_cfg = cfg.get('fit', {}).get('optuna', {})
    sampler_name = (optuna_cfg.get('sampler') or 'tpe').lower()
    if sampler_name == 'random':
        sampler = optuna.samplers.RandomSampler()
    elif sampler_name == 'cmaes':
        sampler = optuna.samplers.CmaEsSampler()
    else:
        sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(direction=optuna_cfg.get('direction', 'minimize'), sampler=sampler)
    study.optimize(objective, n_trials=optuna_cfg.get('n_trials'), timeout=optuna_cfg.get('timeout'))
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and np.isfinite(t.value)]
    if not completed:
        raise RuntimeError("Optuna finished with no completed trials. Check objective stability or reduce search ranges.")
    best_trial = min(completed, key=lambda t: t.value)
    best_theta = suggest_params(best_trial, cfg, sensor_names)
    best_loss = best_trial.value
    unpacked_params = unpack_theta(best_theta, sensor_names)
    return best_theta, best_loss, unpacked_params, study
