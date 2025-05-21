import numpy as np
from scipy.spatial.distance import euclidean

from data_structures import Simulation, Swarm, Inflow, Fluid, Member
from phi.flow import *
from datetime import datetime
# from plotting import plot_save_current_step
import phi.field as field
import phi.math
from auxiliary import trapezoidal_waveform

RECORDING_TIME = 0


def step(v: Field, p: Field, inflow: Inflow, sim: Simulation, swarm: Swarm, fluid_obj: Fluid, t: float, force_actions: np.ndarray):
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
    trap_wave = trapezoidal_waveform(t=t, a=inflow.amplitude, tau=0.5, h=1.5, v=inflow.amplitude / 2)
    v_tensor_u = v.staggered_tensor()[0].numpy('x,y')
    v_tensor_u[:33, :] = trap_wave
    v_tensor_u = tensor(v_tensor_u[:, :-1], spatial('x,y'))
    v_tensor_v = v.staggered_tensor()[1].numpy('x,y')
    v_tensor_v = tensor(v_tensor_v[1:, 1:-1], spatial('x,y'))
    v = StaggeredGrid(math.stack([v_tensor_u, v_tensor_v], dual(vector='x,y')), boundary=v.boundary, bounds=v.bounds, x=sim.resolution[0], y=sim.resolution[1])
    reynolds = inflow.amplitude * sim.length_y / fluid_obj.viscosity
    v = diffuse.explicit(v, 1 / reynolds, sim.dt)
    v = advect.semi_lagrangian(v, v, sim.dt)
    try:
        v, p = fluid.make_incompressible(velocity=v, obstacles=swarm.as_obstacle_list(), solve=Solve(method='scipy-direct', x0=p, max_iterations=0, rel_tol=1e-3, abs_tol=1e-6))
    except Diverged:
        return None, None, swarm
    if t >= RECORDING_TIME:
        # Calculate movement and rotation of swarm members
        for i in range(len(swarm.members)):
            member = swarm.members[i]
            pressure_profile = sample_field_around_obstacle(f=p, member=member, sim=sim, n=8)  # ug/(mm*s^2)
            advance_by_pressure_gradient(member=member, sim=sim, pressure_profile=pressure_profile)
            advance_by_forces(member=member, sim=sim, fluid=fluid_obj, internal_forces=force_actions, swarm_members=swarm.members)
            member.previous_locations.append(member.location.copy())
            member.previous_velocities.append(member.velocity.copy())
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
    field_samples = np.zeros(n, dtype=object)
    for i, angle in enumerate(np.arange(0, 2 * np.pi, 2 * np.pi / n)):
        x_world = member.location['x'] + member.radius * np.cos(angle)
        y_world = member.location['y'] + member.radius * np.sin(angle)
        ix_off = int(x_world * sim.resolution[0] / sim.length_x) + int(np.sign(np.cos(angle)) * offset)
        iy_off = int(y_world * sim.resolution[1] / sim.length_y) + int(np.sign(np.sin(angle)) * offset)
        if iy_off >= sim.resolution[1]:
            iy_off = sim.resolution[1] - 1
        if iy_off < 0:
            iy_off = 0
        # print(f'[{ix_off},{iy_off}]')
        field_samples[i] = f.values.x[ix_off].y[iy_off]
    return field_samples


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
    lin_force_y = 0
    lin_force_x = 0
    for i, angle in enumerate(np.arange(start=0, stop=2 * np.pi, step=np.pi / 4)):
        lin_force_x += -pressure_profile[i] * np.cos(angle) * np.pi / 4 * member.radius
        lin_force_y += -pressure_profile[i] * np.sin(angle) * np.pi / 4 * member.radius
    # Add force due to gradient in x
    x_pred_minus = member.location['x'] - member.radius + lin_force_x / member.mass * sim.dt * sim.dt
    x_pred_plus = member.location['x'] + member.radius + lin_force_x / member.mass * sim.dt * sim.dt
    x_lower = 6 * sim.dx
    x_upper = sim.length_x - 6 * sim.dx
    if (x_pred_minus > x_lower).all or (x_pred_plus < x_upper).all:
        member.velocity['x'] += lin_force_x / member.mass * sim.dt
        member.location['x'] += member.velocity['x'] * sim.dt
    # Add force due to gradient in y
    y_pred_minus = member.location['y'] - member.radius + lin_force_x / member.mass * sim.dt * sim.dt
    y_pred_plus = member.location['y'] + member.radius + lin_force_y / member.mass * sim.dt * sim.dt
    y_lower = 6 * sim.dy
    y_upper = sim.length_y - 6 * sim.dy
    if (y_pred_minus > y_lower).all or (y_pred_plus < y_upper).all:
        member.velocity['y'] += lin_force_y / member.mass * sim.dt
        member.location['y'] += member.velocity['y'] * sim.dt


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
            if 0 < dist < 2 * other_member.radius:
                total_force += np.dot(internal_forces[i] * other_member.max_force, n)
    # Add Stokes drag
    total_force -= 6 * np.pi * fluid.viscosity * member.radius * np.array([member.velocity['x'], member.velocity['y']])
    x_pred_minus = member.location['x'] - member.radius + total_force[0] / member.mass * sim.dt * sim.dt
    x_pred_plus = member.location['x'] + member.radius + total_force[0] / member.mass * sim.dt * sim.dt
    x_lower = 6 * sim.dx
    x_upper = sim.length_x - 6 * sim.dx
    if (x_pred_minus > x_lower).all or (x_pred_plus < x_upper).all:
        member.velocity['x'] += total_force[0] / member.mass * sim.dt
        member.location['x'] += member.velocity['x'] * sim.dt
    y_pred_minus = member.location['y'] - member.radius + total_force[1] / member.mass * sim.dt * sim.dt
    y_pred_plus = member.location['y'] + member.radius + total_force[1] / member.mass * sim.dt * sim.dt
    y_lower = 6 * sim.dy
    y_upper = sim.length_y - 6 * sim.dx
    if (y_pred_minus > y_lower).all or (y_pred_plus < y_upper).all:
        member.velocity['y'] += total_force[1] / member.mass * sim.dt
        member.location['y'] += member.velocity['y'] * sim.dt
