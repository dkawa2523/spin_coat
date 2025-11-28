"""Shared helpers for CLI scripts (config and data loading).

These utilities keep the command-line entrypoints small and ensure the same
behaviour between gradient descent and Optuna runs.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import yaml


def load_config(path: str) -> Dict:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve_base_dir(cfg: Dict, config_path: str) -> str:
    """Resolve the data base directory relative to the config file."""
    base_dir = cfg["data"]["base_dir"]
    if not os.path.isabs(base_dir):
        config_dir = os.path.dirname(os.path.abspath(config_path))
        base_dir = os.path.join(config_dir, base_dir)
    return base_dir


def resolve_output_dir(cfg: Dict, config_path: str) -> str:
    """Resolve output directory relative to the config file."""
    out_dir = cfg.get("fit", {}).get("output_dir", "outputs")
    if not os.path.isabs(out_dir):
        config_dir = os.path.dirname(os.path.abspath(config_path))
        out_dir = os.path.join(config_dir, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def configure_jax_platform(cfg: Dict) -> None:
    """Set JAX platform (cpu/gpu) via env var before importing JAX modules."""
    platform = cfg.get("fit", {}).get("jax_platform")
    if platform:
        import os as _os
        # If the requested platform is unavailable, JAX will error at runtime.
        # For safety, fall back to CPU if GPU is requested but missing.
        if platform == "gpu":
            try:
                import jax  # type: ignore
                if not any(d.platform == "gpu" for d in jax.devices()):
                    _os.environ["JAX_PLATFORM_NAME"] = "cpu"
                else:
                    _os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
            except Exception:
                _os.environ["JAX_PLATFORM_NAME"] = "cpu"
        else:
            _os.environ.setdefault("JAX_PLATFORM_NAME", platform)
    enable_x64 = cfg.get("fit", {}).get("jax_enable_x64")
    if enable_x64 is not None:
        import os as _os

        _os.environ.setdefault("JAX_ENABLE_X64", "1" if enable_x64 else "0")


def load_data(cfg: Dict, config_path: str) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load sensor CSVs defined in the config.

    Returns sensor names, their measurement time grids, and measured thickness values.
    """
    base_dir = _resolve_base_dir(cfg, config_path)
    sensor_names: List[str] = [s["name"] for s in cfg["data"]["sensors"]]
    time_grid: Dict[str, np.ndarray] = {}
    data: Dict[str, np.ndarray] = {}
    for sensor in cfg["data"]["sensors"]:
        path = os.path.join(base_dir, sensor["file"])
        arr = np.loadtxt(path, delimiter=",")
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"Expected 2-column CSV for sensor '{sensor['name']}', got shape {arr.shape}.")
        time_grid[sensor["name"]] = arr[:, 0]
        data[sensor["name"]] = arr[:, 1]
    return sensor_names, time_grid, data


def build_solver_time_grid(cfg: Dict, sensor_names: List[str], data_time_grid: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Build the time grid used by the ODE solver for each sensor."""
    default_time = {"use_data_time_grid": True, "dt": 0.05, "t_min": 0.0, "t_max": 5.0}
    time_cfg = cfg.get("fit", {}).get("time", default_time)
    use_data = time_cfg.get("use_data_time_grid", default_time["use_data_time_grid"])
    if use_data:
        return {name: data_time_grid[name] for name in sensor_names}
    dt = float(time_cfg.get("dt", default_time["dt"]))
    t_min = float(time_cfg.get("t_min", default_time["t_min"]))
    t_max = float(time_cfg.get("t_max", default_time["t_max"]))
    grid = np.arange(t_min, t_max + 0.5 * dt, dt)
    for name in sensor_names:
        if data_time_grid[name][-1] > grid[-1]:
            raise ValueError(
                f"Data for sensor '{name}' extends beyond solver grid t_max={t_max}. "
                "Increase fit.time.t_max or enable use_data_time_grid."
            )
    return {name: grid for name in sensor_names}
