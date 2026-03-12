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

RECORDING_TIME = 0

PADDING = 2

# Number of angular samples around each member used for pressure sampling
NUM_PRESSURE_ANGLES = 4


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
    from phiml.math._tensors import TensorStack
    from phiml.math._optimize import Diverged, NotConverged

    dt_sub = sim.dt / sim.substeps
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
        obstacles = swarm.as_obstacle_list()
        swarm_shapes = [obs.geometry for obs in obstacles]
        swarm_geo = union(swarm_shapes)
        swarm_mask = StaggeredGrid(swarm_geo, boundary=v.boundary, bounds=v.bounds,
                                   x=sim.resolution[0], y=sim.resolution[1])
        v = v * (1.0 - swarm_mask)
        
        try:
            v, p = fluid.make_incompressible(
                velocity=v,
                obstacles=(),
                solve=Solve(method='CG', x0=p, max_iterations=100_000, rel_tol=5e-3, abs_tol=1e-5)
            )
        except (Diverged, NotConverged) as e:
            print(f'Time sub-step {t_sub} diverged or did not converge: {e}')
            return None, None, swarm
    if t >= RECORDING_TIME:
        # Vectorized sampling for all members to reduce Python overhead
        pressure_profiles_all = sample_field_around_obstacles(
            f=p, swarm=swarm, sim=sim, n=NUM_PRESSURE_ANGLES, offset=2
        )  # shape: (num_members, n)
        velocity_profiles = sample_field_around_obstacles(
            f=v, swarm=swarm, sim=sim, n=NUM_PRESSURE_ANGLES, offset=2
        )
        velocity_u_profiles_all = velocity_profiles[..., 0] # shape: (num_members, n)
        velocity_v_profiles_all = velocity_profiles[..., 1] # shape: (num_members, n)
        for i in range(len(swarm.members)):
            member = swarm.members[i]
            velocity_u_profile = velocity_u_profiles_all[i]
            velocity_v_profile = velocity_v_profiles_all[i]
            advance_by_viscous_drag(member=member, sim=sim, fluid=fluid_obj, velocity_profile=(velocity_u_profile, velocity_v_profile))
            pressure_profile = pressure_profiles_all[i]
            advance_by_pressure_gradient(member=member, sim=sim, pressure_profile=pressure_profile)
            advance_by_forces(member=member, sim=sim, fluid=fluid_obj, internal_forces=force_actions,
                              swarm_members=swarm.members)
            member.previous_locations.append(member.location.copy())
            member.previous_velocities.append(member.velocity.copy())
            member.previous_actions.append({'x': force_actions[i][0], 'y': force_actions[i][1]})
        # Enforce collision constraints between members
        resolve_collisions(swarm=swarm, sim=sim, restitution=0.2)
    return v, p, swarm


def sample_field_around_obstacle(f: Field, member: Member, sim: Simulation, n: int, offset=2) -> np.array:
    """
    Samples the field values around a given obstacle in a simulation environment. The function generates
    sampling points along the boundary of the obstacle, computes their respective offsets based on
    the given distance, and then extracts the field values at the adjusted sample points.

    :param f: The field containing the values to be sampled.
    :type f: Field
    :param member: The obstacle or object whose surrounding field values are to be sampled.
    :type member: Member
    :param sim: The simulation in which the field and the obstacle are defined.
    :type sim: Simulation
    :param n: The number of equally spaced angles used for sampling around the obstacle.
    :type n: int
    :param offset: The distance by which the sample points are offset along the direction of the sampled angles
        around the obstacle. Default value is 2.
    :type offset: int
    :return: An array containing the sampled field values around the obstacle, where each entry corresponds
        to a single sample point.
    :rtype: np.array
    """
    angles = np.arange(0, 2 * np.pi, 2 * np.pi / n)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    x_world = member.location['x'] + member.radius * cos_a
    y_world = member.location['y'] + member.radius * sin_a

    x_world += np.sign(cos_a) * offset * (sim.length_x / sim.resolution[0])
    y_world += np.sign(sin_a) * offset * (sim.length_y / sim.resolution[1])
    
    x_world = np.clip(x_world, 0, sim.length_x)
    y_world = np.clip(y_world, 0, sim.length_y)

    coords = np.stack([x_world, y_world], axis=-1)
    coords_tensor = tensor(coords, spatial('n'), channel(vector='x,y'))
    
    samples = phi.field.sample(f, coords_tensor)
    if 'vector' in samples.shape:
        return samples.numpy('n,vector').astype(np.float64)
    return samples.numpy('n').astype(np.float64)


def sample_field_around_obstacles(f: Field, swarm: Swarm, sim: Simulation, n: int, offset=2) -> np.array:
    """
    Vectorized sampling of field around all members. Returns array of shape (num_members, n).
    """
    num_members = len(swarm.members)
    angles = np.arange(0, 2 * np.pi, 2 * np.pi / n)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    member_x = np.array([m.location['x'] for m in swarm.members], dtype=np.float64)[:, None]
    member_y = np.array([m.location['y'] for m in swarm.members], dtype=np.float64)[:, None]
    member_r = np.array([m.radius for m in swarm.members], dtype=np.float64)[:, None]

    x_world = member_x + member_r * cos_a
    y_world = member_y + member_r * sin_a

    # Adjust offsets exactly as before
    x_world += np.sign(cos_a) * offset * (sim.length_x / sim.resolution[0])
    y_world += np.sign(sin_a) * offset * (sim.length_y / sim.resolution[1])
    
    # Clip coordinates to domain
    x_world = np.clip(x_world, 0, sim.length_x)
    y_world = np.clip(y_world, 0, sim.length_y)
    
    # We want to use natively phi.field.Sample or similar.
    # A PointCloud can resample the field directly at those coordinates!
    coords = np.stack([x_world, y_world], axis=-1)  # shape (num_members, n, 2)
    # create generic Phiflow tensor from these coordinates
    coords_tensor = tensor(coords, instance('members'), spatial('n'), channel(vector='x,y'))
    
    samples = phi.field.sample(f, coords_tensor)
    if 'vector' in samples.shape:
        return samples.numpy('members,n,vector').astype(np.float64)
    return samples.numpy('members,n').astype(np.float64)


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


def resolve_collisions(swarm: Swarm, sim: Simulation, restitution: float = 0.0) -> None:
    """
    Resolve pairwise collisions between circular members by separating overlaps and
    applying a simple impulse along the collision normal to reduce interpenetration.
    Vectorized to eliminate slow nested Python loops.

    :param swarm: Swarm containing members with location, velocity, radius, mass
    :param sim: Simulation for boundary extents
    :param restitution: Coefficient of restitution in [0,1], 0 = inelastic, 1 = elastic
    :return: None
    """
    num = len(swarm.members)
    if num < 2:
        return

    # Extract properties into numpy arrays
    pos = np.array([[m.location['x'], m.location['y']] for m in swarm.members], dtype=np.float64)
    vel = np.array([[m.velocity['x'], m.velocity['y']] for m in swarm.members], dtype=np.float64)
    radii = np.array([m.radius for m in swarm.members], dtype=np.float64)
    masses = np.array([m.mass for m in swarm.members], dtype=np.float64)
    inv_masses = np.where(masses > 0, 1.0 / masses, 0.0)

    # Calculate pairwise distances vector-style
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :] # (num, num, 2)
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf) # Ignore self-collision

    dist = np.sqrt(dist_sq)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]

    # Find collisions (upper triangle only to avoid double-processing)
    colliding = (dist < min_dist) & np.triu(np.ones((num, num), dtype=bool), k=1)
    pairs = np.argwhere(colliding)

    for i, j in pairs:
        d = dist[i, j]
        if d == 0.0:
            n = np.array([1.0, 0.0], dtype=np.float64)
            d = 1e-6
        else:
            n = diff[i, j] / d

        # Positional correction
        overlap = min_dist[i, j] - d
        correction = 0.5 * overlap * n
        pos[i] += correction
        pos[j] -= correction

        # Velocity response
        v_rel = vel[i] - vel[j]
        v_sep = np.dot(v_rel, n)
        
        if v_sep < 0:
            j_imp = -(1.0 + restitution) * v_sep / (inv_masses[i] + inv_masses[j] + 1e-12)
            impulse = j_imp * n
            vel[i] += impulse * inv_masses[i]
            vel[j] -= impulse * inv_masses[j]

    # Batch damp and clamp to domain
    x_lower = radii
    x_upper = sim.length_x - radii
    y_lower = radii
    y_upper = sim.length_y - radii

    pos[:, 0] = np.clip(pos[:, 0], x_lower, x_upper)
    pos[:, 1] = np.clip(pos[:, 1], y_lower, y_upper)

    # Write back to objects
    for i, m in enumerate(swarm.members):
        m.location['x'] = float(pos[i, 0])
        m.location['y'] = float(pos[i, 1])
        m.velocity['x'] = float(vel[i, 0])
        m.velocity['y'] = float(vel[i, 1])
