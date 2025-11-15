import numpy as np
from scipy.spatial.distance import euclidean

from data_structures import Simulation, Swarm, Inflow, Fluid, Member
from phi.flow import *
from datetime import datetime
# from plotting import plot_save_current_step
import phi.field as field
import phi.math
from auxiliary import trapezoidal_waveform, beat_waveform
from scipy.spatial import distance

RECORDING_TIME = 0

PADDING = 2

# Number of angular samples around each member used for pressure sampling
NUM_PRESSURE_ANGLES = 4


def step(v: Field, p: Field, inflow: Inflow, sim: Simulation, swarm: Swarm, fluid_obj: Fluid, t: float,
         force_actions: np.ndarray):
    """
    Performs a single simulation step, updating velocity, pressure fields, and swarm dynamics.

    This function performs operations to advance a simulation step by modifying
    the velocity (`v`) and pressure (`p`) fields according to specified inflow
    conditions, fluid properties, and forces. For time frames exceeding a
    certain threshold, it also updates the motion and dynamics of the swarm
    within the simulated fluid domain.

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
    force_actions = np.column_stack((np.full(len(swarm.members), -1.0), np.zeros(len(swarm.members), dtype=float)))
    # trap_wave = trapezoidal_waveform(t=t, a=inflow.amplitude, tau=inflow.frequency, h=inflow.h_shift, v=inflow.v_shift)
    # trap_wave = trapezoidal_waveform(t=t, a=inflow.amplitude, tau=inflow.frequency, h=inflow.h_shift, v=inflow.amplitude/2)
    trap_wave = beat_waveform(t=t, v_peak=inflow.amplitude, v_dia=0, tau=inflow.frequency, upstroke=inflow.upstroke, plateau=inflow.plateau, downstroke=inflow.downstroke)
    # trap_wave = inflow.amplitude / 2
    v_tensor_u = v.staggered_tensor()[0].numpy('x,y')
    v_tensor_u[:33, :] = trap_wave
    v_tensor_u = tensor(v_tensor_u[:, :-1], spatial('x,y'))
    v_tensor_v = v.staggered_tensor()[1].numpy('x,y')
    v_tensor_v = tensor(v_tensor_v[1:, 1:-1], spatial('x,y'))
    v = StaggeredGrid(math.stack([v_tensor_u, v_tensor_v], dual(vector='x,y')), boundary=v.boundary, bounds=v.bounds,
                      x=sim.resolution[0], y=sim.resolution[1])
    reynolds = inflow.amplitude * sim.length_y / fluid_obj.viscosity
    v = diffuse.explicit(v, 1 / reynolds, sim.dt)
    v = advect.semi_lagrangian(v, v, sim.dt)
    try:
        v, p = fluid.make_incompressible(
            velocity=v,
            obstacles=swarm.as_obstacle_list(),
            solve=Solve(method='scipy-CG', x0=p, max_iterations=0, rel_tol=5e-3, abs_tol=1e-5)
        )
    except Diverged:
        return None, None, swarm
    if t >= RECORDING_TIME:
        # Vectorized sampling for all members to reduce Python overhead
        pressure_profiles_all = sample_field_around_obstacles(
            f=p, swarm=swarm, sim=sim, n=NUM_PRESSURE_ANGLES, offset=2
        )  # shape: (num_members, n)
        for i in range(len(swarm.members)):
            member = swarm.members[i]
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

    ix_off = (x_world * sim.resolution[0] / sim.length_x).astype(int) + np.sign(cos_a).astype(int) * offset
    iy_off = (y_world * sim.resolution[1] / sim.length_y).astype(int) + np.sign(sin_a).astype(int) * offset

    ix_off = np.clip(ix_off, 0, sim.resolution[0] - 1)
    iy_off = np.clip(iy_off, 0, sim.resolution[1] - 1)

    # Fast gather using NumPy on the pressure values array
    values = f.values.numpy('x,y')
    field_samples = values[ix_off, iy_off].astype(np.float32)
    return field_samples


def sample_field_around_obstacles(f: Field, swarm: Swarm, sim: Simulation, n: int, offset=2) -> np.array:
    """
    Vectorized sampling of field around all members. Returns array of shape (num_members, n).
    """
    num_members = len(swarm.members)
    angles = np.arange(0, 2 * np.pi, 2 * np.pi / n)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    member_x = np.array([m.location['x'] for m in swarm.members], dtype=np.float32)[:, None]
    member_y = np.array([m.location['y'] for m in swarm.members], dtype=np.float32)[:, None]
    member_r = np.array([m.radius for m in swarm.members], dtype=np.float32)[:, None]

    x_world = member_x + member_r * cos_a
    y_world = member_y + member_r * sin_a

    ix_off = (x_world * sim.resolution[0] / sim.length_x).astype(int) + (np.sign(cos_a).astype(int) * offset)
    iy_off = (y_world * sim.resolution[1] / sim.length_y).astype(int) + (np.sign(sin_a).astype(int) * offset)

    ix_off = np.clip(ix_off, 0, sim.resolution[0] - 1)
    iy_off = np.clip(iy_off, 0, sim.resolution[1] - 1)

    values = f.values.numpy('x,y')
    samples = values[ix_off, iy_off].astype(np.float32)
    return samples


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
    pressure_profile = np.asarray(pressure_profile, dtype=np.float32)
    num_angles = max(1, pressure_profile.shape[0])
    d_theta = 2 * np.pi / num_angles
    angles = np.arange(0, 2 * np.pi, d_theta, dtype=np.float32)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    lin_force_x = -np.sum(pressure_profile * cos_angles) * d_theta * member.radius
    lin_force_y = -np.sum(pressure_profile * sin_angles) * d_theta * member.radius
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
    # Add Stokes drag
    # total_force -= 6 * np.pi * fluid.viscosity * member.radius * np.array([member.velocity['x'], member.velocity['y']])
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

    :param swarm: Swarm containing members with location, velocity, radius, mass
    :param sim: Simulation for boundary extents
    :param restitution: Coefficient of restitution in [0,1], 0 = inelastic, 1 = elastic
    :return: None
    """
    num = len(swarm.members)
    for i in range(num):
        mi = swarm.members[i]
        for j in range(i + 1, num):
            mj = swarm.members[j]

            dx = np.array([mj.location['x'] - mi.location['x'], mj.location['y'] - mi.location['y']], dtype=float)
            dist = float(np.linalg.norm(dx))
            min_dist = mi.radius + mj.radius

            if dist == 0.0:
                # Arbitrary small normal to separate coincident centers
                n = np.array([1.0, 0.0], dtype=float)
                dist = 1e-6
            else:
                n = dx / dist

            if dist < min_dist:
                # Positional correction (split overlap)
                overlap = min_dist - dist
                correction = 0.5 * overlap * n
                mi.location['x'] -= float(correction[0])
                mi.location['y'] -= float(correction[1])
                mj.location['x'] += float(correction[0])
                mj.location['y'] += float(correction[1])

                # Clamp to domain after correction
                for m in (mi, mj):
                    x_lower = m.radius
                    x_upper = sim.length_x - m.radius
                    y_lower = m.radius
                    y_upper = sim.length_y - m.radius
                    if m.location['x'] < x_lower:
                        m.location['x'] = x_lower
                    elif m.location['x'] > x_upper:
                        m.location['x'] = x_upper
                    if m.location['y'] < y_lower:
                        m.location['y'] = y_lower
                    elif m.location['y'] > y_upper:
                        m.location['y'] = y_upper

                # Velocity response (impulse along normal)
                vi = np.array([mi.velocity['x'], mi.velocity['y']], dtype=float)
                vj = np.array([mj.velocity['x'], mj.velocity['y']], dtype=float)
                rel_v = vj - vi
                rel_normal_speed = float(np.dot(rel_v, n))
                if rel_normal_speed < 0:
                    inv_mass_i = 0.0 if mi.mass == 0 else 1.0 / mi.mass
                    inv_mass_j = 0.0 if mj.mass == 0 else 1.0 / mj.mass
                    j_imp = -(1.0 + restitution) * rel_normal_speed / (inv_mass_i + inv_mass_j + 1e-12)
                    impulse = j_imp * n
                    vi -= impulse * inv_mass_i
                    vj += impulse * inv_mass_j
                    mi.velocity['x'], mi.velocity['y'] = float(vi[0]), float(vi[1])
                    mj.velocity['x'], mj.velocity['y'] = float(vj[0]), float(vj[1])
