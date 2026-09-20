"""Unit tests that do not require a full PhiFlow GPU solve."""
from __future__ import annotations

import math

import numpy as np
import torch

from fluxswarm.agents.networks import Actor, tanh_log_prob, tanh_sample
from fluxswarm.agents.pcgrad import pcgrad_merge_actor
from fluxswarm.config import Config, parse_config
from fluxswarm.envs.observations import ObservationBuilder, obs_dim, obs_dim_stacked
from fluxswarm.envs.rewards import _energy, _smoothness
from fluxswarm.physics.inflow import beat_waveform, poiseuille_profile
from fluxswarm.physics.swarm import member_mass_2d, project_actions_to_unit_disk, resolve_collisions
from fluxswarm.physics.domain import Domain


def test_config_roundtrip(tmp_path):
    cfg = Config()
    cfg.sim.coupling = "one-way"
    cfg.obs.localization = "imu"
    path = tmp_path / "cfg.yaml"
    cfg.save(path)
    loaded = Config.load(path)
    assert loaded.sim.coupling == "one-way"
    assert loaded.obs.localization == "imu"


def test_cli_overrides():
    cfg = parse_config(["--batch-envs", "4", "--no-pcgrad", "--coupling", "one-way"])
    assert cfg.train.batch_envs == 4
    assert cfg.train.use_pcgrad is False
    assert cfg.sim.coupling == "one-way"


def test_mass_is_2d_disc():
    m = member_mass_2d(15.12, 0.25)
    assert abs(m - 15.12 * math.pi * 0.25 ** 2) < 1e-9


def test_beat_waveform_period():
    a0 = beat_waveform(0.0, 400.0, 1.0)
    a1 = beat_waveform(1.0, 400.0, 1.0)
    assert abs(a0 - a1) < 1e-6


def test_poiseuille_peaks_center():
    y = torch.linspace(0, 2, 21)
    p = poiseuille_profile(y, 2.0)
    assert float(p[10]) > float(p[0])
    assert float(p[0]) < 1e-6


def test_unit_disk_projection():
    a = torch.tensor([[[3.0, 4.0]]])
    p = project_actions_to_unit_disk(a)
    assert abs(float(torch.linalg.norm(p)) - 1.0) < 1e-6


def test_obs_dim_no_position_in_rich_none():
    cfg = Config().obs
    cfg.preset = "rich"
    cfg.localization = "none"
    d = obs_dim(cfg, 16)
    # Should not include absolute x,y
    assert d == 2 + cfg.ring_points + 2 * cfg.ring_points + cfg.neighbor_k * 4 + 2
    assert obs_dim_stacked(cfg, 16) == d * cfg.history


def test_tanh_logprob_consistency():
    actor = Actor(8, 2, (32, 32))
    obs = torch.randn(4, 3, 8)
    dist, _ = actor.dist(obs)
    action, logp = tanh_sample(dist)
    logp2 = tanh_log_prob(dist, action)
    assert torch.allclose(logp, logp2, atol=1e-4)


def test_pcgrad_actor_only():
    actor = Actor(4, 2, (16, 16))
    obs = torch.randn(8, 4)
    dist, _ = actor.dist(obs)
    action, logp = tanh_sample(dist)
    # Fake advantages
    loss1 = -(logp.sum() * 1.0)
    loss2 = -(logp.sum() * -0.5)
    pcgrad_merge_actor(actor, [loss1, loss2])
    grads = [p.grad for p in actor.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_energy_and_smooth_bounds():
    class _Swarm:
        action = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
        prev_action = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])

    class _S:
        swarm = _Swarm()

    e = _energy(_S())
    sm = _smoothness(_S())
    assert float(e[0, 0]) == 0.0
    assert float(e[0, 1]) == 1.0
    assert 0.0 <= float(sm[0, 0]) <= 1.0


def test_domain_clamp():
    d = Domain(100, 2, 1000, 20, 0.1, 0.1)
    pos = torch.tensor([[[ -1.0, 0.5], [50.0, 5.0]]])
    radii = torch.tensor([[0.25, 0.25]])
    clamped, hit = d.clamp_positions(pos, radii)
    assert hit.any()
    assert float(clamped[0, 0, 0]) >= 0.25
    assert float(clamped[0, 1, 1]) <= 2.0 - 0.25
