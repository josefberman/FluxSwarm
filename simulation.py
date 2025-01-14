import numpy as np
from matplotlib import pyplot as plt

from data_structures import Simulation, Swarm, Inflow, Fluid, Member
from phi.flow import *
from datetime import datetime
from plotting import plot_save_current_step
import phi.field as field
import phi.math


def step(v: Field, p: Field, inflow: Inflow, sim: Simulation, swarm: Swarm, fluid_obj: Fluid,
         t: float):
    if np.floor(t) % 2 == 0:
        rect_wave = inflow.amplitude
    else:
        rect_wave = 0
    rect_wave = inflow.amplitude
    # v_values = math.where(v.points.vector[0] < sim.length_x / 100, math.tensor([rect_wave, 0.0], channel('vector')),
    #                       v.values)
    mask = HardGeometryMask(Box(x=sim.length_x / 100, y=sim.length_y))
    # v = CenteredGrid(values=v_values, boundary=v.boundary, bounds=v.bounds, x=sim.resolution[0], y=sim.resolution[1])
    v = v * (1 - mask) + (rect_wave,0) * mask
    reynolds = inflow.amplitude * sim.length_y / fluid_obj.viscosity
    v = diffuse.explicit(v, fluid_obj.viscosity, sim.dt)
    v = advect.semi_lagrangian(v, v, sim.dt)
    v, p = fluid.make_incompressible(velocity=v, obstacles=swarm.as_obstacle_list(),
                                     solve=Solve(method='scipy-direct', x0=p, max_iterations=1_000_000))
    if t >= 0:
        # Calculate movement and rotation of swarm members
        for member in swarm.members:
            pressure_profile = sample_field_around_obstacle(f=p, member=member, sim=sim)  # ug/(mm*s^2)
            velocity_profile = sample_field_around_obstacle(f=v, member=member, sim=sim)  # mm/s
            advance_linear_motion(member=member, sim=sim, pressure_profile=pressure_profile)
            advance_angular_motion(member=member, sim=sim, inflow=inflow, fluid_obj=fluid_obj,
                                   velocity_profile=velocity_profile)
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
        if (time_step * sim.dt) >= 0:
            plot_save_current_step(time_step=time_step, folder_name=folder_name, v_field=velocity_field,
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


def advance_angular_motion(member: Member, sim: Simulation, inflow: Inflow, fluid_obj: Fluid,
                           velocity_profile: np.array):
    torque = 0
    reynolds = inflow.amplitude * sim.length_y / fluid_obj.viscosity
    Cd = 0.964 + 168.24 / (1 + (reynolds / 0.0307) ** 0.7626)  # for Reynolds number below 18,550
    torque_angles = np.linspace(6 * np.pi / 4, -np.pi / 4, 8)
    for i, v_i in enumerate(velocity_profile):
        speed_i = np.sqrt(v_i['x'].numpy() ** 2 + v_i['y'].numpy() ** 2)
        dir_i = np.arctan(v_i['y'].numpy() / v_i['x'].numpy())
        speed_contribution_size = speed_i * np.cos(dir_i - torque_angles[i])
        speed_contribution_direction = np.sign(np.cos(dir_i - torque_angles[i]))
        torque += -speed_contribution_direction * 0.5 * member.radius ** 2 * np.pi / 4 * Cd * speed_contribution_size ** 2
        # print('speed:', speed_i, ', dir:', np.rad2deg(dir_i), ', torque:', torque)
    moment_of_inertia = 0.5 * member.mass * member.radius ** 2  # MoI of a thin disc
    ang_acceleration = torque / moment_of_inertia
    added_location_theta = member.velocity['omega'] * sim.dt + 0.5 * ang_acceleration * sim.dt ** 2
    added_velocity_omega = ang_acceleration * sim.dt
    member.location['theta'] += added_location_theta
    print('theta:', np.rad2deg(member.location['theta']), 'deg')
    member.velocity['omega'] += added_velocity_omega
