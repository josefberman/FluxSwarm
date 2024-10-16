import numpy as np

from data_structures import Simulation, Swarm, Inflow, Fluid, Member
from phi.flow import *
from datetime import datetime
from plotting import plot_save_current_step
import phi.field as field
import phi.math
from auxiliary import KG_TO_UG, P_TO_WATER


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
    v_tensor_u[:33, 10:-10] = rect_wave
    v_tensor_u = tensor(v_tensor_u[:, :-1], spatial('x,y'))
    v_tensor_v = v.staggered_tensor()[1].numpy('x,y')
    v_tensor_v = tensor(v_tensor_v[:-1, :-2], spatial('x,y'))
    print(f'{t=:.2f}, V={rect_wave}')
    v = StaggeredGrid(math.stack([v_tensor_u, v_tensor_v], dual(vector='x,y')), boundary=v.boundary, bounds=v.bounds,
                      x=sim.resolution[0], y=sim.resolution[1])
    v = advect.semi_lagrangian(v, v, sim.dt)
    v, p = fluid.make_incompressible(velocity=v, obstacles=swarm.as_obstacle_list(),
                                     solve=Solve(rel_tol=1e-03, abs_tol=1e-03, x0=p, max_iterations=1_000_000))
    # Calculate movement and rotation of swarm members
    for member in swarm.members:
        pressure_profile = sample_pressure_around_obstacle(p=p, member=member,
                                                           sim=sim) * P_TO_WATER * KG_TO_UG  # From kg/(μm*s^2) to μg/(μm*s^2)
        lin_force_y = lin_force_x = 0
        for i, angle in enumerate(np.arange(start=0, stop=2 * np.pi, step=np.pi / 4)):
            lin_force_y += pressure_profile[i] * member.radius * np.sin(angle - np.pi / 2) * np.pi / 4
            lin_force_x += pressure_profile[i] * member.radius * np.sin(angle - np.pi) * np.pi / 4
        lin_acceleration_y = lin_force_y / member.mass
        lin_acceleration_x = lin_force_x / member.mass
        print(f'----{pressure_profile=}')
        print(f'----{lin_acceleration_x=}')
        print(f'----{lin_acceleration_y=}')
        print(f'----location before: {member.location}')
        print(f'----velocity before: {member.velocity}')
        member.location['x'] += member.velocity['x'] * sim.dt + 0.5 * lin_acceleration_x * sim.dt ** 2
        member.velocity['x'] += lin_acceleration_x * sim.dt
        member.location['y'] += member.velocity['y'] * sim.dt + 0.5 * lin_acceleration_y * sim.dt ** 2
        member.velocity['y'] += lin_acceleration_y * sim.dt
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
    for i, angle in enumerate(np.arange(start=0, stop=2 * np.pi, step=np.pi / 4)):
        x = int((member.location['x'] + member.radius * np.cos(np.pi / 2 - angle)) * sim.resolution[
            0] / sim.length_x) + int(x_add[i])
        y = int((member.location['y'] + member.radius * np.sin(np.pi / 2 - angle)) * sim.resolution[
            1] / sim.length_y) + int(y_add[i])
        pressure_samples[i] = p.values.x[x].y[y]
    return pressure_samples
