import numpy as np
from scipy.spatial.distance import euclidean
from phi.geom import union
from phi.field import StaggeredGrid

from data_structures import Simulation, Swarm, Inflow, Fluid, Member

# GPU ACCELERATION: Use PyTorch backend for PhiFlow when CUDA is available
import torch
if torch.cuda.is_available():
    from phi.torch.flow import *
    print(f"[PhiFlow] GPU enabled: {torch.cuda.get_device_name(0)}")
else:
    from phi.flow import *
    print("[PhiFlow] Running on CPU (CUDA not available)")

from datetime import datetime
import phi.field as field
import phi.math
from auxiliary import trapezoidal_waveform, beat_waveform
from scipy.spatial import distance
from time import perf_counter

PADDING = 2

# Number of angular samples around each member used for pressure sampling
NUM_PRESSURE_ANGLES = 4
_ANGLE_TRIG_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}
PROFILE_SYNC_POINTS = False


def _get_angle_trig(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Get cached cosine/sine vectors for n angular samples."""
    cached = _ANGLE_TRIG_CACHE.get(n)
    if cached is not None:
        return cached
    angles = np.arange(0, 2 * np.pi, 2 * np.pi / n, dtype=np.float64)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    _ANGLE_TRIG_CACHE[n] = (cos_a, sin_a)
    return cos_a, sin_a


def build_sampling_coords_tensor(swarm: Swarm, sim: Simulation, n: int, offset=2):
    """Build sampling coordinates around all members once for reuse across fields."""
    cos_a, sin_a = _get_angle_trig(n)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    member_x = torch.tensor([m.location['x'] for m in swarm.members], dtype=torch.float64, device=device).unsqueeze(1)
    member_y = torch.tensor([m.location['y'] for m in swarm.members], dtype=torch.float64, device=device).unsqueeze(1)
    member_r = torch.tensor([m.radius for m in swarm.members], dtype=torch.float64, device=device).unsqueeze(1)
    cos_t = torch.tensor(cos_a, dtype=torch.float64, device=device).unsqueeze(0)
    sin_t = torch.tensor(sin_a, dtype=torch.float64, device=device).unsqueeze(0)

    x_world = member_x + member_r * cos_t
    y_world = member_y + member_r * sin_t

    # Adjust offsets exactly as before
    x_world += torch.sign(cos_t) * offset * (sim.length_x / sim.resolution[0])
    y_world += torch.sign(sin_t) * offset * (sim.length_y / sim.resolution[1])

    # Clip coordinates to domain
    x_world = torch.clamp(x_world, 0, sim.length_x)
    y_world = torch.clamp(y_world, 0, sim.length_y)

    coords = torch.stack([x_world, y_world], dim=-1)  # shape (num_members, n, 2)
    return tensor(coords, instance('members'), spatial('n'), channel(vector='x,y'))


def generate_parabolic_profile_mask(v: Field, sim: Simulation, inflow: Inflow, t: float):
    trap_wave = beat_waveform(t=t, v_peak=inflow.amplitude, v_dia=0, tau=inflow.frequency, upstroke=inflow.upstroke, plateau=inflow.plateau, downstroke=inflow.downstroke)
    
    R = sim.resolution[1] / 2.0
    delta = R / 2.0
    
    # Parse the exact unpadded vector components from the grid
    v_u, v_v = math.unstack(v.values, '~vector')
    
    # Ensure y_coords aligns with the exact cell centers of the staggered U component
    y_coords = math.range_tensor(v_u.shape['y']) + 0.5
    
    # Default is trap_wave for core region
    mask = math.ones(v_u.shape['y']) * trap_wave
    
    # Parabolic ramps at the bottom
    mask = math.where(y_coords < delta, trap_wave * (1 - (delta - y_coords)**2 / delta**2), mask)
    
    # Parabolic ramps at the top
    mask = math.where(y_coords > 2*R - delta, trap_wave * (1 - (y_coords - (2*R - delta))**2 / delta**2), mask)
    
    return mask, v_u, v_v

def step(v: Field, p: Field, inflow: Inflow, sim: Simulation, swarm: Swarm, fluid_obj: Fluid, t: float,
         force_actions: np.ndarray) -> tuple[Field | None, Field | None, Swarm]:
    """
    Advances the fluid simulation and swarm members' state by one time step.

    This function updates the fluid's velocity and pressure fields based on
    inflow conditions, diffusion, advection, and incompressibility. It also
    processes the interactions between fluid and swarm members, such as
    applying force actions, updating locations, and storing history. Divergence
    in the simulation logic is caught and handled appropriately. Finally, the
    function extracts profiles around the obstacles and applies relevant properties
    such as viscous drag and pressure forces to the swarm members.

    The execution consists of multiple stages including inflow wave calculation,
    field updates (diffusion and semi-lagrangian advection), obstacle representation,
    incompressible fluid projection, extracting profiles, and applying simulated
    physics to the swarm members.

    :param v: Field representing the velocity grid.
    :param p: Field representing the pressure grid.
    :param inflow: Inflow conditions including parameters like amplitude.
    :param sim: Simulation parameters such as resolution, time step, and domain size.
    :param swarm: The collection of dynamic objects and their current states.
    :param fluid_obj: Fluid properties including viscosity.
    :param t: Current simulation time in seconds.
    :param force_actions: External force actions applied to the swarm.
    :return: Updated velocity and pressure fields, and the swarm state.
    """
    from phiml.math import SolveTape
    from phiml.math._tensors import TensorStack
    from phiml.math._optimize import Diverged, NotConverged

    timings = {}
    t0 = perf_counter()
    dt_sub = sim.dt / sim.substeps
    # Swarm geometry is unchanged within a major step; build mask once.
    obstacles = swarm.as_obstacle_list()
    swarm_shapes = [obs.geometry for obs in obstacles]
    swarm_geo = union(swarm_shapes)
    swarm_mask = StaggeredGrid(swarm_geo, boundary=v.boundary, bounds=v.bounds,
                               x=sim.resolution[0], y=sim.resolution[1])
    for step_idx in range(sim.substeps):
        t_sub = t + step_idx * dt_sub
        mask_next, v_u, v_v = generate_parabolic_profile_mask(v, sim, inflow, t_sub + dt_sub)
        mask_current, _, _ = generate_parabolic_profile_mask(v, sim, inflow, t_sub)
        
        # In incompressible flow, the entire fluid column accelerates instantly.
        # Add the temporal difference to the entire field to preserve existing wakes.
        delta_mask = mask_next - mask_current
        v_tensor_u = v_u + math.expand(delta_mask, v_u.shape['x'])
        
        # Use TensorStack to securely pack the true component shapes natively
        stacked_v = TensorStack((v_tensor_u, v_v), dual(vector='x,y'))
        v = v.with_values(stacked_v)
        reynolds = inflow.amplitude * sim.length_y / fluid_obj.viscosity
        v = diffuse.explicit(v, 1 / reynolds, dt_sub)
        v = advect.semi_lagrangian(v, v, dt_sub)
        v = v * (1.0 - swarm_mask)
        
        try:
            t_solver = perf_counter()
            solver = Solve(
                method='CG',
                x0=p,
                max_iterations=50,
                rel_tol=1e-2,
                abs_tol=1e-3,
                suppress=[NotConverged]
            )
            with SolveTape() as solves:
                v, p = fluid.make_incompressible(
                    velocity=v,
                    obstacles=(),
                    solve=solver
                )
            timings['solver'] = timings.get('solver', 0.0) + (perf_counter() - t_solver)
            # info = solves[solver]
            # print(
            #     f"[make_incompressible] t_sub={t_sub:.4f} "
            #     f"iterations={info.iterations} residual={info.residual}"
            # )
        except Diverged as e:
            print(f'Time sub-step {t_sub} diverged: {e}')
            pass
    # Vectorized sampling for all members to reduce Python overhead
    t_sampling = perf_counter()
    coords_tensor = build_sampling_coords_tensor(
        swarm=swarm, sim=sim, n=NUM_PRESSURE_ANGLES, offset=2
    )
    pressure_profiles_all = sample_field_around_obstacles(
        f=p, swarm=swarm, sim=sim, n=NUM_PRESSURE_ANGLES, offset=2, coords_tensor=coords_tensor
    )  # shape: (num_members, n)
    velocity_profiles = sample_field_around_obstacles(
        f=v, swarm=swarm, sim=sim, n=NUM_PRESSURE_ANGLES, offset=2, coords_tensor=coords_tensor
    )
    timings['sampling'] = perf_counter() - t_sampling
    t_updates = perf_counter()
    pos, vel, radii, masses, max_forces = _extract_swarm_state_arrays(swarm)

    pos, vel = vectorized_advance_by_viscous_drag(
        pos=pos,
        vel=vel,
        radii=radii,
        masses=masses,
        sim=sim,
        fluid=fluid_obj,
        velocity_profiles=velocity_profiles,
    )
    pos, vel = vectorized_advance_by_pressure_gradient(
        pos=pos,
        vel=vel,
        radii=radii,
        masses=masses,
        sim=sim,
        pressure_profiles=pressure_profiles_all,
    )
    pos, vel = vectorized_advance_by_forces(
        pos=pos,
        vel=vel,
        radii=radii,
        masses=masses,
        max_forces=max_forces,
        sim=sim,
        internal_forces=torch.as_tensor(force_actions, dtype=torch.float64, device=pos.device),
    )

    pos, vel = resolve_collisions_tensor(pos, vel, radii, masses, sim, restitution=0.2)
    timings['updates'] = perf_counter() - t_updates

    t_writeback = perf_counter()
    _writeback_swarm_state_arrays(swarm, pos, vel)

    # Capture post-collision state for trajectory histories
    for i in range(len(swarm.members)):
        member = swarm.members[i]
        member.previous_locations.append(member.location.copy())
        member.previous_velocities.append(member.velocity.copy())
        member.previous_actions.append({'x': force_actions[i][0], 'y': force_actions[i][1]})
    timings['writeback_logging'] = perf_counter() - t_writeback
    timings['total'] = perf_counter() - t0
    if PROFILE_SYNC_POINTS:
        print(
            f"[step_timing] total={timings['total']:.4f}s "
            f"solver={timings.get('solver', 0.0):.4f}s sampling={timings.get('sampling', 0.0):.4f}s "
            f"updates={timings.get('updates', 0.0):.4f}s writeback={timings.get('writeback_logging', 0.0):.4f}s"
        )
    return v, p, swarm


def sample_field_around_obstacles(
    f: Field,
    swarm: Swarm,
    sim: Simulation,
    n: int,
    offset=2,
    coords_tensor=None
) -> torch.Tensor:
    """
    Vectorized sampling of field around all members. Returns array of shape (num_members, n).
    """
    if coords_tensor is None:
        coords_tensor = build_sampling_coords_tensor(swarm=swarm, sim=sim, n=n, offset=offset)

    samples = phi.field.sample(f, coords_tensor)
    try:
        if 'vector' in samples.shape:
            return samples.native(('members', 'n', 'vector'))
        return samples.native(('members', 'n'))
    except Exception:
        # Fallback for backends where native extraction with names is unavailable.
        if 'vector' in samples.shape:
            return torch.as_tensor(samples.numpy('members,n,vector').astype(np.float64))
        return torch.as_tensor(samples.numpy('members,n').astype(np.float64))


def _field_values_to_numpy(f: Field):
    """
    Utility to convert PhiFlow fields (Staggered or Dense) to numpy arrays with spatial dims (x,y).
    """
    if hasattr(f, 'values'):
        try:
            return f.values.numpy('x,y')
        except (AttributeError, TypeError):
            # Fall back for dense tensors where .values is not supported
            pass
    # Dense tensor or generic phi.math Tensor
    try:
        return f.numpy('x,y')
    except TypeError:
        return f.numpy()


def _extract_swarm_state_arrays(swarm: Swarm):
    """Extract member state into contiguous torch tensors."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos = torch.tensor([[m.location['x'], m.location['y']] for m in swarm.members], dtype=torch.float64, device=device)
    vel = torch.tensor([[m.velocity['x'], m.velocity['y']] for m in swarm.members], dtype=torch.float64, device=device)
    radii = torch.tensor([m.radius for m in swarm.members], dtype=torch.float64, device=device)
    masses = torch.tensor([m.mass for m in swarm.members], dtype=torch.float64, device=device)
    max_forces = torch.tensor([m.max_force for m in swarm.members], dtype=torch.float64, device=device)
    return pos, vel, radii, masses, max_forces


def _writeback_swarm_state_arrays(swarm: Swarm, pos: torch.Tensor, vel: torch.Tensor) -> None:
    """Write numpy array state back to member objects."""
    pos_cpu = pos.detach().cpu().numpy()
    vel_cpu = vel.detach().cpu().numpy()
    for i, m in enumerate(swarm.members):
        m.location['x'] = float(pos_cpu[i, 0])
        m.location['y'] = float(pos_cpu[i, 1])
        m.velocity['x'] = float(vel_cpu[i, 0])
        m.velocity['y'] = float(vel_cpu[i, 1])


def _apply_force_with_bounds(
    pos: torch.Tensor,
    vel: torch.Tensor,
    force: torch.Tensor,
    radii: torch.Tensor,
    masses: torch.Tensor,
    sim: Simulation,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate one force contribution with the same predictor/boundary logic as before."""
    pos = pos.clone()
    vel = vel.clone()
    inv_masses = torch.where(masses > 0, 1.0 / masses, torch.zeros_like(masses))

    dt = sim.dt
    dt2 = dt * dt
    x_lower = radii
    x_upper = sim.length_x - radii
    y_lower = radii
    y_upper = sim.length_y - radii

    # X component
    x_acc = force[:, 0] * inv_masses
    x_pred_minus = pos[:, 0] + vel[:, 0] * dt + 0.5 * x_acc * dt2 - radii
    x_pred_plus = pos[:, 0] + vel[:, 0] * dt + 0.5 * x_acc * dt2 + radii
    x_ok = (x_pred_minus > x_lower) & (x_pred_plus < x_upper)
    prev_vx = vel[:, 0].clone()
    vel[:, 0] = torch.where(x_ok, vel[:, 0] + x_acc * dt, torch.zeros_like(vel[:, 0]))
    pos[:, 0] += 0.5 * (vel[:, 0] + prev_vx) * dt
    clamped_x = (pos[:, 0] < x_lower) | (pos[:, 0] > x_upper)
    pos[:, 0] = torch.minimum(torch.maximum(pos[:, 0], x_lower), x_upper)
    vel[:, 0] = torch.where(clamped_x, torch.zeros_like(vel[:, 0]), vel[:, 0])

    # Y component
    y_acc = force[:, 1] * inv_masses
    y_pred_minus = pos[:, 1] + vel[:, 1] * dt + 0.5 * y_acc * dt2 - radii
    y_pred_plus = pos[:, 1] + vel[:, 1] * dt + 0.5 * y_acc * dt2 + radii
    y_ok = (y_pred_minus > y_lower) & (y_pred_plus < y_upper)
    prev_vy = vel[:, 1].clone()
    vel[:, 1] = torch.where(y_ok, vel[:, 1] + y_acc * dt, torch.zeros_like(vel[:, 1]))
    pos[:, 1] += 0.5 * (vel[:, 1] + prev_vy) * dt
    clamped_y = (pos[:, 1] < y_lower) | (pos[:, 1] > y_upper)
    pos[:, 1] = torch.minimum(torch.maximum(pos[:, 1], y_lower), y_upper)
    vel[:, 1] = torch.where(clamped_y, torch.zeros_like(vel[:, 1]), vel[:, 1])

    return pos, vel


def vectorized_advance_by_viscous_drag(
    pos: torch.Tensor,
    vel: torch.Tensor,
    radii: torch.Tensor,
    masses: torch.Tensor,
    sim: Simulation,
    fluid: Fluid,
    velocity_profiles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized viscous drag update for all members."""
    velocity_u = velocity_profiles[..., 0]
    velocity_v = velocity_profiles[..., 1]

    v_rel_u = torch.mean(velocity_u, dim=1) + vel[:, 0]
    v_rel_v = torch.mean(velocity_v, dim=1) + vel[:, 1]
    v_mag = torch.sqrt(v_rel_u**2 + v_rel_v**2) + 1e-9

    rho = 1.06
    Re = rho * v_mag * 2.0 * radii / fluid.viscosity
    cd = torch.where(
        Re < 0.1,
        24.0 / torch.clamp(Re, min=1e-12),
        24.0 / torch.clamp(Re, min=1e-12) * (1.0 + 0.15 * Re**0.687)
    )
    area = np.pi * radii**2
    f_mag = 0.5 * rho * v_mag**2 * area * cd
    force_u = -f_mag * v_rel_u / v_mag
    force_v = -f_mag * v_rel_v / v_mag
    force = torch.stack([force_u, force_v], dim=1)

    bad = ~torch.isfinite(force).all(dim=1)
    if torch.any(bad):
        print("[WARNING: simulation.py] NaN/Inf detected in viscous drag force. Clamping to 0.")
        force[bad] = 0.0

    return _apply_force_with_bounds(pos, vel, force, radii, masses, sim)


def vectorized_advance_by_pressure_gradient(
    pos: torch.Tensor,
    vel: torch.Tensor,
    radii: torch.Tensor,
    masses: torch.Tensor,
    sim: Simulation,
    pressure_profiles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized pressure-gradient update for all members."""
    pressure_profiles = pressure_profiles.to(dtype=torch.float64)
    num_angles = pressure_profiles.shape[1] if pressure_profiles.ndim == 2 and pressure_profiles.shape[1] > 0 else 1
    cos_angles, sin_angles = _get_angle_trig(num_angles)
    device = pressure_profiles.device
    cos_t = torch.tensor(cos_angles, dtype=torch.float64, device=device)
    sin_t = torch.tensor(sin_angles, dtype=torch.float64, device=device)

    lin_force_x = -torch.sum(pressure_profiles * cos_t.unsqueeze(0), dim=1) * radii**2
    lin_force_y = -torch.sum(pressure_profiles * sin_t.unsqueeze(0), dim=1) * radii**2
    force = torch.stack([lin_force_x, lin_force_y], dim=1)

    bad = ~torch.isfinite(force).all(dim=1)
    if torch.any(bad):
        print("[WARNING: simulation.py] NaN/Inf detected in pressure gradient force. Clamping to 0.")
        force[bad] = 0.0

    return _apply_force_with_bounds(pos, vel, force, radii, masses, sim)


def vectorized_advance_by_forces(
    pos: torch.Tensor,
    vel: torch.Tensor,
    radii: torch.Tensor,
    masses: torch.Tensor,
    max_forces: torch.Tensor,
    sim: Simulation,
    internal_forces: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized internal/contact-force update for all members."""
    n_members = pos.shape[0]
    if n_members == 0:
        return pos, vel

    forces = internal_forces.to(dtype=torch.float64, device=pos.device)
    if forces.shape[0] != n_members:
        raise ValueError("internal_forces length does not match swarm member count")

    # Self internal force term (equivalent to member == other_member branch).
    self_force = forces * max_forces[:, None]

    # Pairwise geometry from i to j: r_ij = pos_j - pos_i
    r_ij = pos[None, :, :] - pos[:, None, :]
    dist = torch.linalg.norm(r_ij, dim=2)
    safe_dist = torch.where(dist > 1e-12, dist, torch.ones_like(dist))
    n_ij = r_ij / safe_dist[:, :, None]

    # Contact mask equivalent to the original condition:
    # 0 < dist < (2 * radius_j + 2 * max(dx,dy))
    contact_thresh = 2.0 * radii[None, :] + 2.0 * max(sim.dx, sim.dy)
    contact_mask = (dist > 0.0) & (dist < contact_thresh)
    contact_mask.fill_diagonal_(False)

    # Original implementation adds scalar dot(...) to both components.
    # Preserve that behavior exactly.
    other_force = forces[None, :, :] * max_forces[None, :, None]
    contact_scalar = torch.sum(other_force * n_ij, dim=2)  # (i, j)
    contact_scalar = torch.where(contact_mask, contact_scalar, torch.zeros_like(contact_scalar))
    contact_sum = torch.sum(contact_scalar, dim=1)  # (i,)
    total_force = self_force + contact_sum[:, None]

    bad = ~torch.isfinite(total_force).all(dim=1)
    if torch.any(bad):
        print("[WARNING: simulation.py] NaN/Inf detected in internal/contact forces. Clamping to 0.")
        total_force[bad] = 0.0

    return _apply_force_with_bounds(pos, vel, total_force, radii, masses, sim)


def advance_by_pressure_gradient(member: Member, sim: Simulation, pressure_profile: np.array):
    """
    Advances the motion of a member within the simulation domain based on the pressure
    gradient forces applied. This function iteratively computes the resultant forces in
    both the x and y directions due to pressure applied at discrete angles around the
    member's circumference.

    The motion is then updated based on these calculated forces, respecting boundary
    conditions of the simulation domain. The update ensures that the member's position
    and velocity are consistent with the applied forces and the simulation's time step.

    :param member: The member object within the simulation whose motion is being updated.
    :param sim: The simulation context, containing environmental properties and grid
        characteristics such as domain size and timestep.
    :param pressure_profile: A numpy array representing pressure values distributed
        around the member at 8 equidistant angles.
    :return: None
    """
    # Compute resultant forces from pressure samples for arbitrary number of angles
    pressure_profile = np.asarray(pressure_profile, dtype=np.float64)
    num_angles = max(1, pressure_profile.shape[0])
    d_theta = 2 * np.pi / num_angles
    angles = np.arange(0, 2 * np.pi, d_theta, dtype=np.float64)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    lin_force_x = -np.sum(pressure_profile * cos_angles) * member.radius**2
    lin_force_y = -np.sum(pressure_profile * sin_angles) * member.radius**2
    
    # SAFEGUARD: Check for exploding pressure forces
    if not np.isfinite(lin_force_x) or not np.isfinite(lin_force_y):
        print("[WARNING: simulation.py] NaN/Inf detected in pressure gradient force. Clamping to 0.")
        lin_force_x, lin_force_y = 0.0, 0.0
    # Add force due to gradient in x
    x_pred_minus = member.location['x'] + member.velocity[
        'x'] * sim.dt + 0.5 * lin_force_x / member.mass * sim.dt * sim.dt - member.radius
    x_pred_plus = member.location['x'] + member.velocity[
        'x'] * sim.dt + 0.5 * lin_force_x / member.mass * sim.dt * sim.dt + member.radius
    x_lower = member.radius
    x_upper = sim.length_x - member.radius
    prev_velocity_x = member.velocity['x']
    if (x_pred_minus > x_lower) and (x_pred_plus < x_upper):
        member.velocity['x'] += float(lin_force_x / member.mass * sim.dt)
    else:
        member.velocity['x'] = 0
    member.location['x'] += float((member.velocity['x'] + prev_velocity_x) / 2 * sim.dt)
    # Clamp to bounds and damp velocity if clamped
    if member.location['x'] < x_lower:
        member.location['x'] = x_lower
        member.velocity['x'] = 0
    elif member.location['x'] > x_upper:
        member.location['x'] = x_upper
        member.velocity['x'] = 0
    # Add force due to gradient in y
    y_pred_minus = member.location['y'] + member.velocity[
        'y'] * sim.dt + 0.5 * lin_force_y / member.mass * sim.dt * sim.dt - member.radius
    y_pred_plus = member.location['y'] + member.velocity[
        'y'] * sim.dt + 0.5 * lin_force_y / member.mass * sim.dt * sim.dt + member.radius
    y_lower = member.radius
    y_upper = sim.length_y - member.radius
    prev_velocity_y = member.velocity['y']
    if (y_pred_minus > y_lower) and (y_pred_plus < y_upper):
        member.velocity['y'] += float(lin_force_y / member.mass * sim.dt)
    else:
        member.velocity['y'] = 0
    member.location['y'] += float((member.velocity['y'] + prev_velocity_y) / 2 * sim.dt)
    if member.location['y'] < y_lower:
        member.location['y'] = y_lower
        member.velocity['y'] = 0
    elif member.location['y'] > y_upper:
        member.location['y'] = y_upper
        member.velocity['y'] = 0

def advance_by_viscous_drag(member: Member, sim: Simulation, fluid: Fluid, velocity_profile: np.array):
    """
    Advances the motion of a member within the simulation domain based on the viscous drag forces applied.
    :param member: The member object within the simulation whose motion is being updated.
    :param sim: The simulation context, containing environmental properties and grid
        characteristics such as domain size and timestep.
    :param fluid: The fluid object providing the viscosity for Stokes drag calculations.
    :param velocity_profile: A numpy array representing the velocity profile in the x direction around the member.
    :return: None
    """
    velocity_u_profile, velocity_v_profile = velocity_profile
    v_rel_u = np.mean(velocity_u_profile) - (-member.velocity['x'])
    v_rel_v = np.mean(velocity_v_profile) - (-member.velocity['y'])
    v_mag = np.sqrt(v_rel_u**2 + v_rel_v**2) + 1e-9 # avoid division by zero
    rho = 1.06 # density of blood in mg/mm^3
    Re = rho * v_mag * 2 * member.radius / fluid.viscosity
    if Re < 0.1:
        cd = 24/Re
    else:
        cd = 24/Re * (1 + 0.15 * Re**0.687)
    area = np.pi * member.radius**2
    f_mag = 0.5 * rho * v_mag**2 * area * cd
    total_force_u = -f_mag * v_rel_u / v_mag
    total_force_v = -f_mag * v_rel_v / v_mag
    
    # SAFEGUARD: Catch exploding drag forces (e.g. from infinite relative velocity)
    if not np.isfinite(total_force_u) or not np.isfinite(total_force_v):
        print("[WARNING: simulation.py] NaN/Inf detected in viscous drag force. Clamping to 0.")
        total_force_u, total_force_v = 0.0, 0.0
        
    prev_velocity_x = member.velocity['x']
    prev_velocity_y = member.velocity['y']
    member.velocity['x'] += float(total_force_u / member.mass * sim.dt)
    member.velocity['y'] += float(total_force_v / member.mass * sim.dt)
    member.location['x'] += float((member.velocity['x'] + prev_velocity_x) / 2 * sim.dt)
    member.location['y'] += float((member.velocity['y'] + prev_velocity_y) / 2 * sim.dt)

def advance_by_forces(member: Member, sim: Simulation, fluid: Fluid,
                      internal_forces: np.array, swarm_members: list[Member]):
    """
    Advances the position and velocity of a member within a swarm considering internal forces,
    contact forces, and Stokes drag, while adhering to defined simulation boundaries.

    :param member: An individual swarm member whose position and velocity need to be updated.
    :param sim: The simulation object containing the time step, spatial boundaries, and other
        simulation parameters.
    :param fluid: The fluid object providing the viscosity for Stokes drag calculations.
    :param internal_forces: A numpy array representing the internal forces acting between
        the members of the swarm.
    :param swarm_members: List of all swarm members that interact with the specified member.
    :return: None
    """
    total_force = np.zeros(2)
    for i, other_member in enumerate(swarm_members):
        # Add internal force
        if member == other_member:
            total_force += internal_forces[i] * member.max_force
        # Add contact forces
        if member != other_member:
            r_ij = np.array(
                [other_member.location['x'] - member.location['x'], other_member.location['y'] - member.location['y']])
            dist = np.linalg.norm(r_ij)
            n = r_ij / dist
            if 0 < dist < (2 * other_member.radius + 2 * np.max([sim.dx, sim.dy])):
                total_force += np.dot(internal_forces[i] * other_member.max_force, n)
    # SAFEGUARD: Catch diverging internal/contact forces
    if not np.isfinite(total_force).all():
        print("[WARNING: simulation.py] NaN/Inf detected in internal/contact forces. Clamping to 0.")
        total_force = np.zeros(2)
        
    x_pred_minus = member.location['x'] + member.velocity['x'] * sim.dt + 0.5 * total_force[
        0] / member.mass * sim.dt * sim.dt - member.radius
    x_pred_plus = member.location['x'] + member.velocity['x'] * sim.dt + 0.5 * total_force[
        0] / member.mass * sim.dt * sim.dt + member.radius
    x_lower = member.radius
    x_upper = sim.length_x - member.radius
    prev_velocity_x = member.velocity['x']
    if (x_pred_minus > x_lower) and (x_pred_plus < x_upper):
        member.velocity['x'] += float(total_force[0] / member.mass * sim.dt)
    else:
        member.velocity['x'] = 0
    member.location['x'] += float((member.velocity['x'] + prev_velocity_x) / 2 * sim.dt)
    if member.location['x'] < x_lower:
        member.location['x'] = x_lower
        member.velocity['x'] = 0
    elif member.location['x'] > x_upper:
        member.location['x'] = x_upper
        member.velocity['x'] = 0
    y_pred_minus = member.location['y'] + member.velocity['y'] * sim.dt + 0.5 * total_force[
        1] / member.mass * sim.dt * sim.dt - member.radius
    y_pred_plus = member.location['y'] + member.velocity['y'] * sim.dt + 0.5 * total_force[
        1] / member.mass * sim.dt * sim.dt + member.radius
    y_lower = member.radius
    y_upper = sim.length_y - member.radius
    prev_velocity_y = member.velocity['y']
    if (y_pred_minus > y_lower) and (y_pred_plus < y_upper):
        member.velocity['y'] += float(total_force[1] / member.mass * sim.dt)
    else:
        member.velocity['y'] = 0
    member.location['y'] += float((member.velocity['y'] + prev_velocity_y) / 2 * sim.dt)
    if member.location['y'] < y_lower:
        member.location['y'] = y_lower
        member.velocity['y'] = 0
    elif member.location['y'] > y_upper:
        member.location['y'] = y_upper
        member.velocity['y'] = 0


def resolve_collisions_tensor(
    pos: torch.Tensor,
    vel: torch.Tensor,
    radii: torch.Tensor,
    masses: torch.Tensor,
    sim: Simulation,
    restitution: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve collisions for tensor state using spatial-hash broad phase."""
    num = pos.shape[0]
    if num < 2:
        return pos, vel

    pos = pos.clone()
    vel = vel.clone()
    inv_masses = torch.where(masses > 0, 1.0 / masses, torch.zeros_like(masses))

    # Spatial hash in Python space (candidate generation only).
    pos_cpu = pos.detach().cpu().numpy()
    radii_cpu = radii.detach().cpu().numpy()
    max_interaction = 2.0 * float(np.max(radii_cpu)) + 2.0 * max(sim.dx, sim.dy)
    cell_size = max(max_interaction, 1e-9)
    cell_coords = np.floor(pos_cpu / cell_size).astype(np.int64)

    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, c in enumerate(cell_coords):
        key = (int(c[0]), int(c[1]))
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(idx)

    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]
    candidate_pairs: set[tuple[int, int]] = set()
    for key, indices in buckets.items():
        for i in indices:
            for dx, dy in neighbor_offsets:
                nkey = (key[0] + dx, key[1] + dy)
                if nkey not in buckets:
                    continue
                for j in buckets[nkey]:
                    if j <= i:
                        continue
                    candidate_pairs.add((i, j))

    for i, j in sorted(candidate_pairs):
        diff_ij = pos[i] - pos[j]
        d = torch.linalg.norm(diff_ij)
        min_dist_ij = radii[i] + radii[j]
        if not (d < min_dist_ij):
            continue

        if d <= 1e-12:
            n = torch.tensor([1.0, 0.0], dtype=pos.dtype, device=pos.device)
            d = torch.tensor(1e-6, dtype=pos.dtype, device=pos.device)
        else:
            n = diff_ij / d

        overlap = min_dist_ij - d
        correction = 0.5 * overlap * n
        pos[i] += correction
        pos[j] -= correction

        v_rel = vel[i] - vel[j]
        v_sep = torch.dot(v_rel, n)
        if v_sep < 0:
            j_imp = -(1.0 + restitution) * v_sep / (inv_masses[i] + inv_masses[j] + 1e-12)
            impulse = j_imp * n
            vel[i] += impulse * inv_masses[i]
            vel[j] -= impulse * inv_masses[j]

    x_lower = radii
    x_upper = sim.length_x - radii
    y_lower = radii
    y_upper = sim.length_y - radii
    pos[:, 0] = torch.minimum(torch.maximum(pos[:, 0], x_lower), x_upper)
    pos[:, 1] = torch.minimum(torch.maximum(pos[:, 1], y_lower), y_upper)
    return pos, vel


def resolve_collisions(swarm: Swarm, sim: Simulation, restitution: float = 0.0) -> None:
    """Compatibility wrapper that resolves collisions and writes back into swarm objects."""
    pos, vel, radii, masses, _ = _extract_swarm_state_arrays(swarm)
    pos, vel = resolve_collisions_tensor(pos, vel, radii, masses, sim, restitution=restitution)
    _writeback_swarm_state_arrays(swarm, pos, vel)
