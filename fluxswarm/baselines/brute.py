"""Scripted brute-force baselines."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np

from fluxswarm.config import Config
from fluxswarm.envs.swarm_env import BatchedSwarmEnv
from fluxswarm.runs.recorder import RunRecorder


def brute_upstream_actions(batch: int, n: int) -> np.ndarray:
    return np.tile([-1.0, 0.0], (batch, n, 1)).astype(np.float64)


def brute_wall_actions(
    env: BatchedSwarmEnv,
    mode: Literal["static", "dynamic"] = "static",
) -> np.ndarray:
    inv = 1.0 / math.sqrt(2.0)
    actions = np.zeros((env.batch, env.n, 2), dtype=np.float64)
    mid = env.cfg.sim.length_y / 2.0
    if mode == "static":
        y = env.solver.swarm.pos0.detach().cpu().numpy()[..., 1]
    else:
        y = env.solver.swarm.pos.detach().cpu().numpy()[..., 1]
    upper = y > mid
    actions[upper] = [-inv, inv]
    actions[~upper] = [-inv, -inv]
    return actions


def run_brute(
    cfg: Config,
    policy: Literal["upstream", "wall"] = "upstream",
    max_steps: int | None = None,
) -> Path:
    cfg_d = Config.from_dict(cfg.to_dict())
    cfg_d.train.batch_envs = min(cfg_d.train.batch_envs, 8)
    cfg_d.run.tag = cfg_d.run.tag or f"brute_{policy}"
    # Prefer one-way for fast baselines
    if cfg_d.sim.coupling == "two-way" and max_steps is None:
        pass  # honor config
    env = BatchedSwarmEnv(cfg_d, batch=cfg_d.train.batch_envs)
    recorder = RunRecorder(cfg_d, algorithm=f"brute_{policy}")
    recorder.save_config(cfg_d)
    env.reset(seed=cfg_d.train.seed)

    steps = max_steps or min(cfg_d.train.total_timesteps, 5000)
    global_steps = 0
    while global_steps < steps:
        if policy == "upstream":
            actions = brute_upstream_actions(env.batch, env.n)
        else:
            actions = brute_wall_actions(env, cfg_d.run.wall_policy_mode)
        _, scalar, _, _, info = env.step(actions)
        recorder.log_step(global_steps, env, scalar, info, save_fields=False)
        global_steps += env.batch
    recorder.close()
    return recorder.run_dir
