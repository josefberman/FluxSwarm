"""Batched swarm state, forces, and collisions (GPU tensors)."""
from __future__ import annotations

import math

import torch

from fluxswarm.config import SimConfig, SwarmConfig
from fluxswarm.physics.domain import Domain


def member_mass_2d(density: float, radius: float) -> float:
    """2D disc mass = density * pi * r^2 (mg when density is mg/mm^3 areal)."""
    return density * math.pi * radius * radius


def initial_layout(
    swarm: SwarmConfig,
    sim: SimConfig,
    batch: int,
    device: torch.device,
) -> torch.Tensor:
    """Return positions (B, N, 2) for a rectangular grid, optionally centered."""
    n = swarm.num_members
    if swarm.use_centered_layout or swarm.member_interval_x is None:
        gap = (sim.length_y - 2 * swarm.num_y * swarm.member_radius) / (swarm.num_y + 1)
        interval_x = gap
        interval_y = gap
        total_w = 2 * (swarm.num_x - 1) * swarm.member_radius + (swarm.num_x - 1) * gap
        left = sim.length_x / 2.0 - total_w / 2.0
        bottom = gap + swarm.member_radius
    else:
        interval_x = float(swarm.member_interval_x)
        interval_y = float(swarm.member_interval_y or interval_x)
        left = float(swarm.left_location or 0.0)
        bottom = float(swarm.bottom_location or swarm.member_radius)

    xs, ys = [], []
    for iy in range(swarm.num_y):
        for ix in range(swarm.num_x):
            xs.append(left + ix * (2 * swarm.member_radius + interval_x))
            ys.append(bottom + iy * (2 * swarm.member_radius + interval_y))
    pos = torch.tensor(list(zip(xs, ys)), dtype=torch.float64, device=device)  # (N, 2)
    return pos.unsqueeze(0).expand(batch, -1, -1).contiguous()


class SwarmState:
    """Batched member kinematics on device. Shapes: (B, N, ...)."""

    def __init__(
        self,
        batch: int,
        swarm: SwarmConfig,
        sim: SimConfig,
        device: torch.device,
    ):
        self.batch = batch
        self.n = swarm.num_members
        self.device = device
        self.radius = swarm.member_radius
        self.max_force = swarm.member_max_force
        mass = member_mass_2d(swarm.member_density, swarm.member_radius)
        self.masses = torch.full((batch, self.n), mass, dtype=torch.float64, device=device)
        self.radii = torch.full((batch, self.n), swarm.member_radius, dtype=torch.float64, device=device)
        self.max_forces = torch.full((batch, self.n), swarm.member_max_force, dtype=torch.float64, device=device)
        self.pos = initial_layout(swarm, sim, batch, device)
        self.vel = torch.zeros(batch, self.n, 2, dtype=torch.float64, device=device)
        self.action = torch.zeros(batch, self.n, 2, dtype=torch.float64, device=device)
        self.prev_action = torch.zeros(batch, self.n, 2, dtype=torch.float64, device=device)
        self.pos0 = self.pos.clone()  # episode-start positions for localization
        self.imu_bias = torch.zeros(batch, self.n, 2, dtype=torch.float64, device=device)
        self.imu_vel = torch.zeros(batch, self.n, 2, dtype=torch.float64, device=device)
        self.imu_pos = torch.zeros(batch, self.n, 2, dtype=torch.float64, device=device)

    def reset_envs(self, env_ids: torch.Tensor, swarm: SwarmConfig, sim: SimConfig) -> None:
        if env_ids.numel() == 0:
            return
        layout = initial_layout(swarm, sim, 1, self.device).squeeze(0)  # (N, 2)
        for i in env_ids.tolist():
            self.pos[i] = layout
            self.vel[i] = 0
            self.action[i] = 0
            self.prev_action[i] = 0
            self.pos0[i] = layout
            self.imu_bias[i] = 0
            self.imu_vel[i] = 0
            self.imu_pos[i] = 0


def apply_force_integrate(
    pos: torch.Tensor,
    vel: torch.Tensor,
    force: torch.Tensor,
    masses: torch.Tensor,
    radii: torch.Tensor,
    domain: Domain,
    dt: float,
    max_force_mag: float | None = None,
    v_max: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Semi-implicit Euler with wall prediction, per-axis.

    Optional ``max_force_mag`` / ``v_max`` are safety ceilings (never bind under
    correct physics; prevent runaway if a force bug appears).
    """
    if max_force_mag is not None and max_force_mag > 0:
        f_mag = torch.linalg.norm(force, dim=-1, keepdim=True).clamp_min(1e-12)
        force = torch.where(f_mag > max_force_mag, force * (max_force_mag / f_mag), force)

    accel = force / masses.unsqueeze(-1)
    # X then Y separately (matches legacy ordering)
    for axis in (0, 1):
        L = domain.length_x if axis == 0 else domain.length_y
        a = accel[..., axis]
        v_old = vel[..., axis]
        v_new = v_old + a * dt
        pred = pos[..., axis] + 0.5 * (v_new + v_old) * dt
        r = radii
        inside = (pred >= r) & (pred <= L - r)
        v_new = torch.where(inside, v_new, torch.zeros_like(v_new))
        pos_new = pos[..., axis] + 0.5 * (v_new + v_old) * dt
        pos_new = torch.clamp(pos_new, r, L - r)
        hit = (pos_new <= r + 1e-12) | (pos_new >= L - r - 1e-12)
        v_new = torch.where(hit, torch.zeros_like(v_new), v_new)
        vel = vel.clone()
        pos = pos.clone()
        vel[..., axis] = v_new
        pos[..., axis] = pos_new

    if v_max is not None and v_max > 0:
        v_mag = torch.linalg.norm(vel, dim=-1, keepdim=True).clamp_min(1e-12)
        vel = torch.where(v_mag > v_max, vel * (v_max / v_mag), vel)
    return pos, vel


def stokes_drag_2d(
    vel: torch.Tensor,
    fluid_u: torch.Tensor,
    fluid_v: torch.Tensor,
    mu: float = 3.0,
) -> torch.Tensor:
    """Viscous-only 2D Stokes drag per unit depth (pressure drag from the ring).

    ``F = -4 π μ (v_agent - v_fluid)`` with unit depth = 1 mm.
    """
    v_agent_minus_fluid = torch.stack(
        [vel[..., 0] - fluid_u, vel[..., 1] - fluid_v],
        dim=-1,
    )
    return -4.0 * math.pi * mu * v_agent_minus_fluid


def pressure_force(
    pressure_ring: torch.Tensor,
    radii: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    """2D ring quadrature of -p n hat. pressure_ring: (B, N, K), angles: (K,)."""
    cos_t = torch.cos(angles)
    sin_t = torch.sin(angles)
    k = angles.numel()
    fx = -(pressure_ring * cos_t).sum(dim=-1) * radii * (2 * math.pi / k)
    fy = -(pressure_ring * sin_t).sum(dim=-1) * radii * (2 * math.pi / k)
    return torch.stack([fx, fy], dim=-1)


def thrust_force(actions: torch.Tensor, max_forces: torch.Tensor) -> torch.Tensor:
    """actions (B,N,2) in unit disk -> force."""
    return actions * max_forces.unsqueeze(-1)


def resolve_collisions(
    pos: torch.Tensor,
    vel: torch.Tensor,
    radii: torch.Tensor,
    masses: torch.Tensor,
    domain: Domain,
    iterations: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized sphere-sphere + wall clamp. Soft O(N^2) broad phase on GPU."""
    B, N, _ = pos.shape
    for _ in range(iterations):
        # pairwise
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # (B, N, N, 2)
        dist = torch.linalg.norm(diff, dim=-1).clamp_min(1e-12)
        min_dist = radii.unsqueeze(2) + radii.unsqueeze(1)
        overlap = min_dist - dist
        mask = (overlap > 0) & ~torch.eye(N, device=pos.device, dtype=torch.bool).unsqueeze(0)
        # push proportional to inverse mass
        inv_m = 1.0 / masses.clamp_min(1e-12)
        w_i = inv_m.unsqueeze(2)
        w_j = inv_m.unsqueeze(1)
        w_sum = (w_i + w_j).clamp_min(1e-12)
        push = (overlap / w_sum).unsqueeze(-1) * (diff / dist.unsqueeze(-1))
        push = torch.where(mask.unsqueeze(-1), push, torch.zeros_like(push))
        # each particle receives half of its pairs (i pushed by j)
        delta = (push * w_i.unsqueeze(-1)).sum(dim=2) - (push.transpose(1, 2) * w_j.unsqueeze(-1)).sum(dim=2)
        # Actually: for pair (i,j), move i by +push*w_i/w_sum already in push construction
        # Simpler approach: move i away from j by 0.5 * overlap along normal, mass-weighted
        nrm = diff / dist.unsqueeze(-1)
        corr_i = (overlap * (w_i / w_sum)).unsqueeze(-1) * nrm
        corr_i = torch.where(mask.unsqueeze(-1), corr_i, torch.zeros_like(corr_i))
        pos = pos + corr_i.sum(dim=2)

        # zero approach velocity along contact normal
        v_rel = vel.unsqueeze(2) - vel.unsqueeze(1)
        vn = (v_rel * nrm).sum(dim=-1)
        approaching = (vn < 0) & mask
        impulse = (-vn / w_sum).unsqueeze(-1) * nrm
        impulse = torch.where(approaching.unsqueeze(-1), impulse, torch.zeros_like(impulse))
        vel = vel + (impulse * w_i.unsqueeze(-1)).sum(dim=2)

        pos, hit = domain.clamp_positions(pos, radii)
        vel = torch.where(hit.unsqueeze(-1), torch.zeros_like(vel), vel)
    return pos, vel


def project_actions_to_unit_disk(actions: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.norm(actions, dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.where(norms > 1.0, actions / norms, actions)
