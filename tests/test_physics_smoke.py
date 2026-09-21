"""Physics smoke tests (GPU Brinkman/DCT). Skip if CUDA unavailable when desired."""
from __future__ import annotations

import pytest
import torch

from fluxswarm.config import Config
from fluxswarm.envs.swarm_env import BatchedSwarmEnv


@pytest.fixture
def tiny_cfg():
    cfg = Config()
    cfg.sim.resolution_scale = 5.0  # 500 x 10
    cfg.sim.dt_substeps = 2
    cfg.sim.inflow_velocity = 100.0
    cfg.swarm.num_x = 2
    cfg.swarm.num_y = 2
    cfg.train.batch_envs = 2
    cfg.task.episode_duration = 0.05
    cfg.obs.preset = "rich"
    cfg.obs.history = 2
    cfg.obs.ring_points = 4
    cfg.train.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def test_env_reset_step_shapes(tiny_cfg):
    env = BatchedSwarmEnv(tiny_cfg, batch=2)
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] == 2
    assert obs.shape[1] == 4
    assert obs.shape[2] == env.obs_dim
    actions = __import__("numpy").zeros((2, 4, 2), dtype="float64")
    actions[..., 0] = -1.0
    next_obs, rewards, terms, truncs, info = env.step(actions)
    assert next_obs.shape == obs.shape
    assert rewards.shape == (2,)
    assert "reward_matrix" in info
    assert info["reward_matrix"].shape == (2, 4, 3)
    assert env.solver.last_drag is not None


def test_two_way_step_does_not_crash(tiny_cfg):
    tiny_cfg.train.batch_envs = 1
    env = BatchedSwarmEnv(tiny_cfg, batch=1)
    env.reset(seed=1)
    actions = __import__("numpy").tile([-0.7, 0.0], (1, 4, 1))
    env.step(actions)
