"""Multi-objective rewards: progress, energy, smoothness."""
from __future__ import annotations

import math

import torch

from fluxswarm.config import TaskConfig
from fluxswarm.physics.solver import BatchedFluidSolver


def compute_rewards(
    solver: BatchedFluidSolver,
    task: TaskConfig,
    prev_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (reward_matrix, objectives_matrix, scalar_reward).

    reward_matrix / objectives_matrix: (B, N, 3) — progress, energy, smooth.
    reward_matrix is weighted; objectives_matrix is unweighted.
    scalar_reward: (B,) mean over agents of sum of weighted objectives.
    """
    B, N, _ = solver.swarm.pos.shape
    device = solver.swarm.pos.device
    progress = _progress(solver, task, prev_pos)
    energy = _energy(solver)
    smooth = _smoothness(solver)

    objectives = torch.stack([progress, energy, smooth], dim=-1)  # (B,N,3)
    weights = torch.tensor(
        [task.w_progress, task.w_energy, task.w_smooth],
        device=device,
        dtype=objectives.dtype,
    )
    reward_matrix = objectives * weights
    scalar = reward_matrix.sum(dim=-1).mean(dim=-1)
    return reward_matrix, objectives, scalar


def _progress(solver: BatchedFluidSolver, task: TaskConfig, prev_pos: torch.Tensor) -> torch.Tensor:
    pos = solver.swarm.pos
    mode = task.progress_reward
    L = max(task.failure_x - task.success_x, 1e-6)

    if mode == "potential":
        dx = prev_pos[..., 0] - pos[..., 0]  # positive when moving upstream
        prog = 100.0 * dx / L
        # Terminal bonuses applied by env on done; keep step shaping clean
        return prog

    if mode == "fluid-relative":
        fu = solver.last_fluid_center_u
        if fu is None:
            fu = torch.zeros(pos.shape[0], pos.shape[1], device=pos.device, dtype=pos.dtype)
        v_ref = max(abs(solver.cfg.sim.inflow_velocity), 1e-6)
        return torch.clamp((fu - solver.swarm.vel[..., 0]) / v_ref, -1.0, 1.0)

    # legacy
    prog = torch.zeros(pos.shape[0], pos.shape[1], device=pos.device, dtype=pos.dtype)
    xs = pos[..., 0]
    mean_x = xs.mean(dim=-1, keepdim=True)
    success = mean_x <= task.success_x
    prog = torch.where(success.expand_as(prog), torch.full_like(prog, 10.0), prog)

    x_t = xs
    x_prev = prev_pos[..., 0]
    moving_left = x_t < x_prev
    delta = (x_prev - x_t) / L
    tanh_term = torch.tanh(3.0 * delta) / math.tanh(3.0) * 100.0 - 0.1
    linear_term = delta * 100.0 - 0.1
    step_r = torch.where(moving_left, tanh_term, linear_term)
    failed = x_t >= task.failure_x
    step_r = torch.where(failed, torch.full_like(step_r, -10.0), step_r)
    # Don't overwrite success bonus unless failure
    prog = torch.where(success.expand_as(prog) & ~failed, prog, step_r)
    return prog


def _energy(solver: BatchedFluidSolver) -> torch.Tensor:
    """1 - ||a|| (actions already in unit disk)."""
    a = solver.swarm.action
    mag = torch.linalg.norm(a, dim=-1).clamp(0.0, 1.0)
    return 1.0 - mag


def _smoothness(solver: BatchedFluidSolver) -> torch.Tensor:
    a = solver.swarm.action
    p = solver.swarm.prev_action
    na = torch.linalg.norm(a, dim=-1).clamp_min(1e-8)
    np_ = torch.linalg.norm(p, dim=-1).clamp_min(1e-8)
    cos = (a * p).sum(dim=-1) / (na * np_)
    cos = cos.clamp(-1.0, 1.0)
    # If prev action is zero (first step), return 0.5 neutral
    first = np_ < 1e-7
    return torch.where(first, torch.full_like(cos, 0.5), (cos + 1.0) / 2.0)
