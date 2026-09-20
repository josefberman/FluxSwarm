"""Batched PhiFlow fluid solver with one-way and two-way coupling."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from fluxswarm.config import Config, SimConfig
from fluxswarm.physics.domain import Domain
from fluxswarm.physics.inflow import beat_waveform, poiseuille_on_grid
from fluxswarm.physics.swarm import (
    SwarmState,
    apply_force_integrate,
    pressure_force,
    project_actions_to_unit_disk,
    resolve_collisions,
    thrust_force,
    viscous_drag_force,
)


@dataclass
class FieldSnapshot:
    vx: np.ndarray  # (B, nx+1, ny) or for B=1 without batch
    vy: np.ndarray
    p: np.ndarray
    t: float


class BatchedFluidSolver:
    """One process, batch-dim PhiFlow solver on GPU (or CPU)."""

    def __init__(self, cfg: Config, batch: int, device: torch.device | str = "cuda"):
        self.cfg = cfg
        self.sim: SimConfig = cfg.sim
        self.batch = batch
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        if self.device.type == "cuda":
            # PhiFlow default device
            from phi.torch import flow as _flow  # noqa: F401
            from phi import math as phimath

            try:
                from phi.torch.flow import backend

                backend.default_backend().set_default_device("GPU")
            except Exception:
                pass
        self.domain = Domain.from_config(self.sim)
        self.swarm = SwarmState(batch, cfg.swarm, cfg.sim, self.device)
        self._init_fields()
        self.episode_time = torch.zeros(batch, dtype=torch.float64, device=self.device)
        self.ring_k = cfg.obs.ring_points
        self.angles = torch.linspace(0, 2 * math.pi, self.ring_k + 1, device=self.device, dtype=torch.float64)[:-1]
        # cached samples from last step
        self.last_pressure_ring: Optional[torch.Tensor] = None
        self.last_vel_ring_u: Optional[torch.Tensor] = None
        self.last_vel_ring_v: Optional[torch.Tensor] = None
        self.last_fluid_center_u: Optional[torch.Tensor] = None
        self.last_fluid_center_v: Optional[torch.Tensor] = None

    def _init_fields(self) -> None:
        from phi.torch.flow import ZERO_GRADIENT, Box, StaggeredGrid, math as pmath

        B = self.batch
        nx, ny = self.domain.nx, self.domain.ny
        box = Box["x,y", 0 : self.sim.length_x, 0 : self.sim.length_y]
        boundary = {"x": ZERO_GRADIENT, "y": 0}
        # Batched staggered grid
        self.v = StaggeredGrid(
            0,
            boundary=boundary,
            bounds=box,
            x=nx,
            y=ny,
        )
        # Expand to batch via stacking if needed
        if B > 1:
            from phi import math as phimath

            self.v = phimath.expand(self.v, phimath.batch(b=B))
        self.p = None
        # Apply initial parabolic inflow at t=0
        self._apply_inflow_delta(t=0.0, dt_sub=0.0, absolute=True)

    def _apply_inflow_delta(self, t: float, dt_sub: float, absolute: bool = False) -> None:
        """Add beat*Poiseuille increment to u-component (or set absolute at t=0)."""
        from phi import math as phimath

        amp = self.sim.inflow_velocity
        period = self.sim.inflow_period
        if absolute:
            mask_val = beat_waveform(t, amp, period)
        else:
            mask_next = beat_waveform(t + dt_sub, amp, period)
            mask_curr = beat_waveform(t, amp, period)
            mask_val = mask_next - mask_curr

        ny = self.domain.ny
        parabolic = poiseuille_on_grid(ny, self.sim.length_y, device=None)
        # Work through PhiFlow unstack
        try:
            v_u, v_v = phimath.unstack(self.v.values, "~vector")
            # Build mask matching u-grid y-size
            # Use numpy/torch bridge via phimath
            y_coords = phimath.range_tensor(v_u.shape["y"]) + 0.5
            R = self.domain.ny / 2.0
            parabolic_phi = 1.0 - ((y_coords - R) / R) ** 2
            mask = float(mask_val) * parabolic_phi
            if absolute:
                # Replace u with mask (broadcast over x)
                new_u = phimath.ones(v_u.shape) * 0 + mask  # spatial mask
                # For absolute init, set u everywhere to parabolic*amp at t
                stacked = phimath.stack([new_u * 0 + mask, v_v * 0], dim=phimath.channel("vector"))
                # Simpler: add mask as delta from zero
                delta = phimath.stack(
                    [mask + v_u * 0, v_v * 0],
                    dim=phimath.channel("vector"),
                )
                self.v = self.v.with_values(self.v.values * 0 + delta)
            else:
                delta = phimath.stack([mask + v_u * 0, v_v * 0], dim=phimath.channel("vector"))
                self.v = self.v.with_values(self.v.values + delta)
        except Exception:
            # Fallback: skip if shape mismatch during early init
            pass

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.batch, device=self.device)
        self.swarm.reset_envs(env_ids, self.cfg.swarm, self.sim)
        self.episode_time[env_ids] = 0.0
        # Full-field reset is expensive; for partial resets we only zero members.
        # On full reset, re-init fields.
        if env_ids.numel() == self.batch:
            self._init_fields()

    def _sample_rings(self) -> None:
        """Sample pressure and velocity on rings around each member."""
        from phi.torch.flow import field as phifield
        from phi import math as phimath

        B, N = self.batch, self.swarm.n
        k = self.ring_k
        factor = self.cfg.obs.ring_radius_factor
        r = self.swarm.radii * factor  # (B, N)
        angles = self.angles  # (K,)
        # Sample points (B, N, K, 2)
        cx = self.swarm.pos[..., 0:1]
        cy = self.swarm.pos[..., 1:2]
        sx = cx + r.unsqueeze(-1) * torch.cos(angles)
        sy = cy + r.unsqueeze(-1) * torch.sin(angles)
        # Clip to domain
        sx = sx.clamp(0.0, self.sim.length_x)
        sy = sy.clamp(0.0, self.sim.length_y)

        # Flatten for PhiFlow sample: use numpy bridge
        pts = torch.stack([sx, sy], dim=-1).detach().cpu().numpy()  # (B,N,K,2)

        def _sample_scalar(f, name_fallback=0.0):
            if f is None:
                return np.zeros((B, N, k), dtype=np.float64)
            out = np.zeros((B, N, k), dtype=np.float64)
            # Sample per-env to keep indexing simple
            for b in range(B):
                try:
                    # Build PointCloud or use sample at coords
                    coords = pts[b].reshape(-1, 2)
                    from phi.torch.flow import CenteredGrid, Sphere, vec

                    # Use field.sample with a tensor of positions via PhiFlow API
                    pos_tensor = phimath.tensor(
                        coords.reshape(N, k, 2),
                        phimath.instance("points"),
                        phimath.dual("vector"),
                    )
                    # Fallback: nearest-grid numpy sample
                    vals = _numpy_sample_field(f, coords, self.domain)
                    out[b] = vals.reshape(N, k)
                except Exception:
                    out[b] = name_fallback
            return out

        p_np = _sample_scalar(self.p)
        # Velocity components
        try:
            v_u, v_v = phimath.unstack(self.v.values, "~vector")
            u_np = np.zeros((B, N, k), dtype=np.float64)
            vv_np = np.zeros((B, N, k), dtype=np.float64)
            for b in range(B):
                coords = pts[b].reshape(-1, 2)
                u_np[b] = _numpy_sample_staggered_u(self.v, coords, self.domain, b if B > 1 else None).reshape(N, k)
                vv_np[b] = _numpy_sample_staggered_v(self.v, coords, self.domain, b if B > 1 else None).reshape(N, k)
        except Exception:
            u_np = np.zeros((B, N, k), dtype=np.float64)
            vv_np = np.zeros((B, N, k), dtype=np.float64)

        self.last_pressure_ring = torch.tensor(p_np, device=self.device, dtype=torch.float64)
        self.last_vel_ring_u = torch.tensor(u_np, device=self.device, dtype=torch.float64)
        self.last_vel_ring_v = torch.tensor(vv_np, device=self.device, dtype=torch.float64)
        self.last_fluid_center_u = self.last_vel_ring_u.mean(dim=-1)
        self.last_fluid_center_v = self.last_vel_ring_v.mean(dim=-1)

    def step(self, actions: torch.Tensor) -> None:
        """One RL step: swarm forces then fluid substeps. actions: (B, N, 2)."""
        from phi.torch.flow import Sphere, Obstacle, union, StaggeredGrid, diffuse, advect, fluid, Solve, vec
        from phi import math as phimath

        actions = project_actions_to_unit_disk(actions.to(self.device, dtype=torch.float64))
        # Early override like legacy
        early = self.episode_time <= 0.005
        if bool(torch.as_tensor(early).any()):
            forced = actions.clone()
            forced[early] = torch.tensor([-1.0, 0.0], device=self.device, dtype=torch.float64)
            actions = forced

        self.swarm.prev_action = self.swarm.action.clone()
        self.swarm.action = actions

        t0 = self.episode_time  # (B,)
        # Sample fields for forces (skip if first instant with no pressure)
        if self.p is not None or float(t0.min()) > 0:
            self._sample_rings()
        else:
            B, N, k = self.batch, self.swarm.n, self.ring_k
            z = torch.zeros(B, N, k, device=self.device, dtype=torch.float64)
            self.last_pressure_ring = z
            self.last_vel_ring_u = z
            self.last_vel_ring_v = z
            self.last_fluid_center_u = z.mean(-1)
            self.last_fluid_center_v = z.mean(-1)

        # Forces
        drag = viscous_drag_force(
            self.swarm.pos,
            self.swarm.vel,
            self.last_fluid_center_u,
            self.last_fluid_center_v,
            self.swarm.radii,
            rho=self.sim.fluid_density,
        )
        pforce = pressure_force(self.last_pressure_ring, self.swarm.radii, self.angles)
        thrust = thrust_force(actions, self.swarm.max_forces)
        total = drag + pforce + thrust

        # Integrate swarm at full dt
        dt = self.sim.dt
        pos, vel = apply_force_integrate(
            self.swarm.pos, self.swarm.vel, total, self.swarm.masses, self.swarm.radii, self.domain, dt
        )
        pos, vel = resolve_collisions(pos, vel, self.swarm.radii, self.swarm.masses, self.domain)
        self.swarm.pos, self.swarm.vel = pos, vel

        # Update IMU estimates
        self._update_imu(total)

        # Fluid substeps with frozen obstacle geometry
        dt_sub = dt / self.sim.substeps
        Re_inv = self.sim.viscosity / max(self.sim.inflow_velocity * self.sim.length_y, 1e-8)

        # Build obstacles once (list of Obstacle, one per member, batched)
        obstacles = self._build_obstacles()
        coupling = self.sim.coupling

        for s in range(self.sim.substeps):
            t_sub = float(self.episode_time.mean().item()) + s * dt_sub
            self._apply_inflow_delta(t_sub, dt_sub, absolute=False)
            self.v = diffuse.explicit(self.v, Re_inv, dt_sub)
            self.v = advect.semi_lagrangian(self.v, self.v, dt_sub)

            if coupling == "one-way":
                mask = self._obstacle_mask()
                self.v = self.v * (1.0 - mask)
                obst_list: list = []
            else:
                obst_list = obstacles

            try:
                self.v, self.p = fluid.make_incompressible(
                    velocity=self.v,
                    obstacles=obst_list,
                    solve=Solve(
                        "CG",
                        1e-2,
                        1e-3,
                        max_iterations=100,
                        x0=self.p,
                        rank_deficiency=0,
                    ),
                )
            except Exception as e:
                # PhiFlow raises NotConverged on hard residual; continue with last fields
                # (matches legacy silent swallow) but surface a warning.
                import warnings

                warnings.warn(f"Pressure solve issue at t={t_sub}: {e}", RuntimeWarning)
                from phiml.math._optimize import NotConverged

                if not isinstance(e, NotConverged) and "converg" not in str(e).lower():
                    raise RuntimeError(f"Pressure solve failed at t={t_sub}: {e}") from e

            if coupling == "one-way":
                mask = self._obstacle_mask()
                self.v = self.v * (1.0 - mask)

        self.episode_time = self.episode_time + dt
        self._sample_rings()

    def _update_imu(self, force: torch.Tensor) -> None:
        dt = self.sim.dt
        true_accel = force / self.swarm.masses.unsqueeze(-1)
        noise = torch.randn_like(true_accel) * self.cfg.obs.imu_noise
        measured = true_accel + self.swarm.imu_bias + noise
        self.swarm.imu_vel = self.swarm.imu_vel + measured * dt
        self.swarm.imu_pos = self.swarm.imu_pos + self.swarm.imu_vel * dt

    def reset_imu_bias(self, env_ids: torch.Tensor) -> None:
        b = self.cfg.obs.imu_bias
        for i in env_ids.tolist():
            self.swarm.imu_bias[i] = torch.randn(self.swarm.n, 2, device=self.device, dtype=torch.float64) * b
            self.swarm.imu_vel[i] = 0
            self.swarm.imu_pos[i] = 0

    def _build_obstacles(self):
        """Batched obstacles without instance dims (PhiFlow grids disallow instance)."""
        from phi.torch.flow import Sphere, Obstacle, union
        from phi import math as phimath

        B, N = self.batch, self.swarm.n
        pos = self.swarm.pos.detach().cpu().numpy()
        vel = self.swarm.vel.detach().cpu().numpy()
        spheres = []
        for i in range(N):
            if B > 1:
                cx = phimath.tensor(pos[:, i, 0], phimath.batch("b"))
                cy = phimath.tensor(pos[:, i, 1], phimath.batch("b"))
                vx = phimath.tensor(vel[:, i, 0], phimath.batch("b"))
                vy = phimath.tensor(vel[:, i, 1], phimath.batch("b"))
            else:
                cx = phimath.tensor(float(pos[0, i, 0]))
                cy = phimath.tensor(float(pos[0, i, 1]))
                vx = phimath.tensor(float(vel[0, i, 0]))
                vy = phimath.tensor(float(vel[0, i, 1]))
            center = phimath.stack({"x": cx, "y": cy}, phimath.channel(vector="x,y"))
            ovel = phimath.stack({"x": vx, "y": vy}, phimath.channel(vector="x,y"))
            spheres.append(Obstacle(Sphere(center=center, radius=self.swarm.radius), velocity=ovel))
        # Union of Obstacle geometries: pass as list to make_incompressible
        return spheres

    def _obstacle_mask(self):
        """Rasterized obstacle mask for one-way coupling (no instance dims)."""
        from phi.torch.flow import Sphere, StaggeredGrid, union
        from phi import math as phimath

        pos = self.swarm.pos.detach().cpu().numpy()
        B, N, _ = pos.shape
        geos = []
        for i in range(N):
            if B > 1:
                cx = phimath.tensor(pos[:, i, 0], phimath.batch("b"))
                cy = phimath.tensor(pos[:, i, 1], phimath.batch("b"))
            else:
                cx = phimath.tensor(float(pos[0, i, 0]))
                cy = phimath.tensor(float(pos[0, i, 1]))
            center = phimath.stack({"x": cx, "y": cy}, phimath.channel(vector="x,y"))
            geos.append(Sphere(center=center, radius=self.swarm.radius))
        geo = union(geos) if len(geos) > 1 else geos[0]
        return StaggeredGrid(
            geo,
            boundary=self.v.boundary,
            bounds=self.v.bounds,
            x=self.domain.nx,
            y=self.domain.ny,
        )

    def export_fields(self, env_index: int = 0) -> FieldSnapshot:
        """Export numpy field arrays for one env."""
        vx, vy, p = _fields_to_numpy(self.v, self.p, env_index, self.batch)
        return FieldSnapshot(vx=vx, vy=vy, p=p, t=float(self.episode_time[env_index].item()))


# ---- numpy sampling helpers -------------------------------------------------


def _numpy_sample_field(f, coords: np.ndarray, domain: Domain) -> np.ndarray:
    """Nearest-neighbor sample of a centered grid field. coords (M,2)."""
    try:
        from phi import math as phimath

        arr = f.values.numpy("x,y")
        if arr.ndim > 2:
            arr = arr[..., 0] if arr.shape[-1] == 1 else arr.mean(axis=-1)
    except Exception:
        return np.zeros(coords.shape[0], dtype=np.float64)
    nx, ny = arr.shape[0], arr.shape[1]
    ix = np.clip((coords[:, 0] / domain.length_x * nx).astype(int), 0, nx - 1)
    iy = np.clip((coords[:, 1] / domain.length_y * ny).astype(int), 0, ny - 1)
    return arr[ix, iy].astype(np.float64)


def _numpy_sample_staggered_u(v, coords, domain, batch_idx=None) -> np.ndarray:
    try:
        from phi import math as phimath

        vu, _ = phimath.unstack(v.values, "~vector")
        names = [str(d) for d in vu.shape.sizes]
        if batch_idx is not None and "b" in names:
            arr = vu.numpy("b,x,y")[batch_idx]
        else:
            arr = vu.numpy("x,y")
    except Exception:
        return np.zeros(coords.shape[0], dtype=np.float64)
    nx, ny = arr.shape[0], arr.shape[1]
    ix = np.clip((coords[:, 0] / domain.length_x * nx).astype(int), 0, nx - 1)
    iy = np.clip((coords[:, 1] / domain.length_y * ny).astype(int), 0, ny - 1)
    return arr[ix, iy].astype(np.float64)


def _numpy_sample_staggered_v(v, coords, domain, batch_idx=None) -> np.ndarray:
    try:
        from phi import math as phimath

        _, vv = phimath.unstack(v.values, "~vector")
        names = [str(d) for d in vv.shape.sizes]
        if batch_idx is not None and "b" in names:
            arr = vv.numpy("b,x,y")[batch_idx]
        else:
            arr = vv.numpy("x,y")
    except Exception:
        return np.zeros(coords.shape[0], dtype=np.float64)
    nx, ny = arr.shape[0], arr.shape[1]
    ix = np.clip((coords[:, 0] / domain.length_x * nx).astype(int), 0, nx - 1)
    iy = np.clip((coords[:, 1] / domain.length_y * ny).astype(int), 0, ny - 1)
    return arr[ix, iy].astype(np.float64)


def _fields_to_numpy(v, p, env_index: int, batch: int):
    from phi import math as phimath

    vu, vv = phimath.unstack(v.values, "~vector")
    try:
        if batch > 1:
            vx = vu.numpy("b,x,y")[env_index]
            vy = vv.numpy("b,x,y")[env_index]
            pp = p.numpy("b,x,y")[env_index] if p is not None else np.zeros_like(vx)
        else:
            vx = vu.numpy("x,y")
            vy = vv.numpy("x,y")
            pp = p.numpy("x,y") if p is not None else np.zeros_like(vx)
    except Exception:
        vx = np.zeros((10, 10))
        vy = np.zeros_like(vx)
        pp = np.zeros_like(vx)
    return vx, vy, pp
