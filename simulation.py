import numpy as np

from data_structures import Simulation, Swarm, Inflow, Fluid, Member
from phi.flow import *
from datetime import datetime
from plotting import plot_save_current_step
import phi.field as field
import phi.math


def step(v: Field, p: Field, inflow_field: Field, inflow: Inflow, sim: Simulation, swarm: Swarm,
         t: float):
    if np.floor(t) % 2 == 0:
        rect_wave = inflow.amplitude
    else:
        rect_wave = 0
    # rect_wave = 4 / np.pi * np.sin(inflow.frequency * t)
    # for n in range(3, 100, 2):
    #     rect_wave += 4 / np.pi * 1 / n * np.sin(n * inflow.frequency * t)
    # rect_wave = np.sin(inflow.frequency * t)
    # rect_wave = 1

    v_tensor_u = v.staggered_tensor()[0].numpy('x,y')
    # v_tensor_u[:33, 10:-10] = (0.5 * inflow.amplitude * rect_wave + 0.5 * inflow.amplitude)
    v_tensor_u[:33, :] = rect_wave
    v_tensor_u = tensor(v_tensor_u[:, :-1], spatial('x,y'))
    v_tensor_v = v.staggered_tensor()[1].numpy('x,y')
    v_tensor_v = tensor(v_tensor_v[:-1, :-2], spatial('x,y'))
    # print(f'{t=:.2f}, V={rect_wave}')
    v = StaggeredGrid(math.stack([v_tensor_u, v_tensor_v], dual(vector='x,y')), boundary=v.boundary, bounds=v.bounds,
                      x=sim.resolution[0], y=sim.resolution[1])
    v = advect.semi_lagrangian(v, v, sim.dt)
    v, p = fluid.make_incompressible(velocity=v, obstacles=swarm.as_obstacle_list(),
                                     solve=Solve(method='scipy-direct', x0=p, max_iterations=1_000_000))
    # rel_tol = 1e-02, abs_tol = 1e-02,
    # Calculate movement and rotation of swarm members
    for member in swarm.members:
        pressure_profile = sample_pressure_around_obstacle(p=p, member=member,
                                                           sim=sim)  # pg/(um*s^2)
        lin_force_y = lin_force_x = torque = 0
        for i, angle in enumerate(np.arange(start=0, stop=2 * np.pi, step=np.pi / 4)):
            lin_force_y += pressure_profile[i] * member.radius * np.sin(angle - np.pi / 2) * np.pi / 4
            lin_force_x += pressure_profile[i] * member.radius * np.sin(angle - np.pi) * np.pi / 4
        for p_i, p_j in zip(pressure_profile, np.roll(pressure_profile, -1)):
            torque += member.radius ** 2 * (p_j - p_i) * np.pi / 4 * np.cos(np.pi / 8)
        lin_acceleration_y = lin_force_y / member.mass
        lin_acceleration_x = lin_force_x / member.mass
        moment_of_inertia = 0.5 * member.mass * member.radius ** 2  # MoI of a thin disc
        ang_acceleration = torque / moment_of_inertia
        # print(f'----{pressure_profile=}')
        # print(f'----{lin_acceleration_x=}')
        # print(f'----{lin_acceleration_y=}')
        print(f'{ang_acceleration=}')
        # print(f'----location before: {member.location}')
        # print(f'----velocity before: {member.velocity}')
        added_location_x = member.velocity['x'] * sim.dt + 0.5 * lin_acceleration_x * sim.dt ** 2
        added_velocity_x = lin_acceleration_x * sim.dt
        added_location_y = member.velocity['y'] * sim.dt + 0.5 * lin_acceleration_y * sim.dt ** 2
        added_velocity_y = lin_acceleration_y * sim.dt
        added_location_theta = member.velocity['omega'] * sim.dt + 0.5 * ang_acceleration * sim.dt ** 2
        added_velocity_omega = ang_acceleration * sim.dt
        if (4 + member.radius) < (member.location['x'] + added_location_x) < (
                sim.length_x - member.radius - 4 * sim.dx):
            member.location['x'] += added_location_x
            member.velocity['x'] += added_velocity_x
        if (4 + member.radius) < (member.location['y'] + added_location_y) < (
                sim.length_y - member.radius - 4 * sim.dy):
            member.location['y'] += added_location_y
            member.velocity['y'] += added_velocity_y
        member.location['theta'] += added_location_theta
        member.velocity['omega'] += added_velocity_omega
        member.previous_locations.append(member.location)
        member.previous_velocities.append(member.velocity)
        print(member.previous_locations[-1])
        print(f'----location after: {member.location}')
        print(f'----velocity after: {member.velocity}')

    return v, p, inflow_field, swarm


def run_simulation(velocity_field: Field, pressure_field: Field | None, inflow_field: Field,
                   inflow: Inflow, sim: Simulation, swarm: Swarm, folder_name: str) -> None:
    for time_step in range(1, sim.time_steps + 1):
        print('Sim time:', time_step * sim.dt)
        calc_start = datetime.now()
        velocity_field, pressure_field, inflow_field, swarm = step(v=velocity_field, p=pressure_field,
                                                                   inflow_field=inflow_field, inflow=inflow, sim=sim,
                                                                   swarm=swarm, t=time_step * sim.dt)
        print('Calculation time:', datetime.now() - calc_start)
        plot_save_current_step(time_step=time_step, folder_name=folder_name, v_field=velocity_field,
                               p_field=pressure_field,
                               inflow_field=inflow_field, sim=sim, swarm=swarm)
        phi.field.write(velocity_field, f'./run_{folder_name}/velocity/{time_step:04}')
        phi.field.write(pressure_field, f'./run_{folder_name}/pressure/{time_step:04}')
        phi.field.write(inflow_field, f'./run_{folder_name}/inflow/{time_step:04}')
    return None


def sample_pressure_around_obstacle(p: Field, member: Member, sim: Simulation) -> np.array:
    pressure_samples = np.zeros(8)
    x_add = np.array([0, 1, 1, 1, 0, -1, -1, -1]) * 3  # ensuring measurement outside of sphere
    y_add = np.roll(x_add, -2)
    x_add = y_add = np.zeros(8)
    for i, angle in enumerate(np.arange(start=0, stop=2 * np.pi, step=np.pi / 4)):
        x = int((member.location['x'] + member.radius * np.cos(np.pi / 2 - angle)) * sim.resolution[
            0] / sim.length_x) + int(x_add[i])
        y = int((member.location['y'] + member.radius * np.sin(np.pi / 2 - angle)) * sim.resolution[
            1] / sim.length_y) + int(y_add[i])
        pressure_samples[i] = p.values.x[x].y[y]
    return pressure_samples
