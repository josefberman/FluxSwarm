import numpy as np
import phiml.math
from fontTools.misc.bezierTools import epsilon
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean

from data_structures import Simulation, Swarm, Inflow, Fluid, Member
from phi.flow import *
from datetime import datetime
from plotting import plot_save_current_step
import phi.field as field
import phi.math
from auxiliary import trapezoidal_waveform

RECORDING_TIME = 0


def step(v: Field, p: Field, inflow: Inflow, sim: Simulation, swarm: Swarm, fluid_obj: Fluid,
         t: float):
    trap_wave = trapezoidal_waveform(t=t, a=inflow.amplitude, tau=2, h=1.5, v=inflow.amplitude / 2)
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
        v, p = fluid.make_incompressible(velocity=v, obstacles=swarm.as_obstacle_list(),
                                     solve=Solve(method='scipy-direct', x0=p, max_iterations=1_000_000))
    except Diverged:
        return None, None, swarm
    if t >= RECORDING_TIME:
        # Calculate movement and rotation of swarm members
        for i in range(len(swarm.members)):
            member = swarm.members[i]
            pressure_profile = sample_field_around_obstacle(f=p, member=member, sim=sim)  # ug/(mm*s^2)
            # velocity_profile = sample_field_around_obstacle(f=v, member=member, sim=sim)  # mm/s
            # for member2 in swarm.members:
            #     if member2 is not member:
            #         update_for_impact(member, member2)
            advance_linear_motion(member=member, sim=sim, pressure_profile=pressure_profile)
            for j in range(i + 1, len(swarm.members)):
                member2 = swarm.members[j]
                handle_collisions(member, member2)
            member.previous_locations.append(member.location.copy())
            member.previous_velocities.append(member.velocity.copy())
    return v, p, swarm


def run_simulation(velocity_field: Field, pressure_field: Field | None,
                   inflow: Inflow, sim: Simulation, swarm: Swarm, fluid_obj: Fluid, folder_name: str) -> None:
    for time_step in range(1, sim.time_steps + 1):
        print(f'Sim time:{time_step * sim.dt:.2f}')
        calc_start = datetime.now()
        velocity_field, pressure_field, swarm = step(v=velocity_field, p=pressure_field, inflow=inflow,
                                                     sim=sim, swarm=swarm, fluid_obj=fluid_obj,
                                                     t=time_step * sim.dt)
        print('Calculation time:', datetime.now() - calc_start)
        if (time_step * sim.dt) >= RECORDING_TIME:
            plot_save_current_step(current_time=time_step * sim.dt, folder_name=folder_name, v_field=velocity_field,
                                   p_field=pressure_field, sim=sim, swarm=swarm)
            phi.field.write(velocity_field, f'../runs/run_{folder_name}/velocity/{time_step:04}')
            phi.field.write(pressure_field, f'../runs/run_{folder_name}/pressure/{time_step:04}')

    return None


def sample_field_around_obstacle(f: Field, member: Member, sim: Simulation) -> np.array:
    field_samples = np.zeros(8, dtype=object)
    x_add = np.array([1, 1, 0, -1, -1, -1, 0, 1]) * 2  # ensuring measurement outside of disc
    y_add = np.roll(x_add, 2)
    for i, angle in enumerate(np.arange(start=0, stop=2 * np.pi, step=np.pi / 4)):
        x = int((member.location['x'] + member.radius * np.cos(angle)) * sim.resolution[
            0] / sim.length_x) + int(x_add[i])
        y = int((member.location['y'] + member.radius * np.sin(angle)) * sim.resolution[
            1] / sim.length_y) + int(y_add[i])
        field_samples[i] = f.values.x[x].y[y]
    return field_samples


def update_for_impact(member1: Member, member2: Member):
    dist = euclidean([member1.location['x'], member1.location['y']], [member2.location['x'], member2.location['y']])
    if dist < (member1.radius + member2.radius):
        v_rel_x = member2.velocity['x'] - member1.velocity['x']
        v_rel_y = member2.velocity['y'] - member1.velocity['y']
        nx = (member2.location['x'] - member1.location['x']) / dist
        ny = (member2.location['y'] - member1.location['y']) / dist
        v_rel_n = v_rel_x * nx + v_rel_y * ny
        if v_rel_n <= 0:
            impulse = (2 * v_rel_n) / (1 / member1.mass + 1 / member2.mass)
            member1.velocity['x'] += (impulse / member1.mass) * nx
            member1.velocity['y'] += (impulse / member1.mass) * ny
            member2.velocity['x'] -= (impulse / member2.mass) * nx
            member2.velocity['y'] -= (impulse / member2.mass) * ny
            overlap = (member1.radius + member2.radius) - dist
            member1.location['x'] -= (overlap / 2) * nx
            member1.location['y'] -= (overlap / 2) * ny
            member2.location['x'] += (overlap / 2) * nx
            member2.location['y'] += (overlap / 2) * ny


def handle_collisions(member1: Member, member2: Member):
    """
    Checks for collisions among swarm members and updates their velocities based on elastic collision physics.

    :param member1:
    :param member2:
    """
    distance = euclidean([member1.location['x'], member1.location['y']],
                         [member2.location['x'], member2.location['y']])
    if distance < (member1.radius + member2.radius):  # Collision detected
        # Compute unit normal and tangent vectors
        normal = [(member2.location['x'] - member1.location['x']) / distance,
                  (member2.location['y'] - member1.location['y']) / distance]
        tangent = [-normal[1], normal[0]]

        # Project velocities onto normal and tangent vectors
        v1n = normal[0] * member1.velocity['x'] + normal[1] * member1.velocity['y']
        v1t = tangent[0] * member1.velocity['x'] + tangent[1] * member1.velocity['y']
        v2n = normal[0] * member2.velocity['x'] + normal[1] * member2.velocity['y']
        v2t = tangent[0] * member2.velocity['x'] + tangent[1] * member2.velocity['y']

        # Compute new normal velocities after elastic collision
        v1n_new = (v1n * (member1.mass - member2.mass) + 2 * member2.mass * v2n) / (member1.mass + member2.mass)
        v2n_new = (v2n * (member2.mass - member1.mass) + 2 * member1.mass * v1n) / (member1.mass + member2.mass)

        # Convert back to x, y components
        member1.velocity['x'] = v1n_new * normal[0] + v1t * tangent[0]
        member1.velocity['y'] = v1n_new * normal[1] + v1t * tangent[1]
        member2.velocity['x'] = v2n_new * normal[0] + v2t * tangent[0]
        member2.velocity['y'] = v2n_new * normal[1] + v2t * tangent[1]


def advance_linear_motion(member: Member, sim: Simulation, pressure_profile: np.array):
    lin_force_y = 0
    lin_force_x = 0
    for i, angle in enumerate(np.arange(start=0, stop=2 * np.pi, step=np.pi / 4)):
        lin_force_x += -pressure_profile[i] * np.cos(angle) * np.pi / 4 * member.radius
        lin_force_y += -pressure_profile[i] * np.sin(angle) * np.pi / 4 * member.radius
    lin_acceleration_x = lin_force_x / member.mass
    lin_acceleration_y = lin_force_y / member.mass
    added_location_x = member.velocity['x'] * sim.dt + 0.5 * lin_acceleration_x * sim.dt ** 2
    added_velocity_x = lin_acceleration_x * sim.dt
    added_location_y = member.velocity['y'] * sim.dt + 0.5 * lin_acceleration_y * sim.dt ** 2
    added_velocity_y = lin_acceleration_y * sim.dt
    if (4 + member.radius) < (member.location['x'] + added_location_x) < (
            sim.length_x - member.radius - 4 * sim.dx):
        member.location['x'] += added_location_x
        member.velocity['x'] += added_velocity_x
    if (4 + member.radius) < (member.location['y'] + added_location_y) < (
            sim.length_y - member.radius - 4 * sim.dy):
        member.location['y'] += added_location_y
        member.velocity['y'] += added_velocity_y
