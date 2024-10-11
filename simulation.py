import numpy as np

from data_structures import Simulation, Swarm, Inflow, Fluid, Member
from phi.flow import *
from datetime import datetime
from plotting import plot_save_current_step
import phi.field as field
import phi.math


def step(v: Field, p: Field, inflow_field: Field, inflow: Inflow, sim: Simulation, swarm: Swarm,
         t: float):
    rect_wave = 4 / np.pi * np.sin(inflow.frequency * t)
    for n in range(3, 100, 2):
        rect_wave += 4 / np.pi * 1 / n * np.sin(n * inflow.frequency * t)

    v_tensor_u = v.staggered_tensor()[0].numpy('x,y')
    v_tensor_u[:33, :] = 0.5 * inflow.amplitude * rect_wave + 0.5 * inflow.amplitude
    v_tensor_u[33:36, :] = 0.5 * inflow.amplitude * rect_wave + 0.5 * inflow.amplitude / 2
    v_tensor_u[36:39] = 0.5 * inflow.amplitude * rect_wave + 0.5 * inflow.amplitude / 4
    v_tensor_u[39:42] = 0.5 * inflow.amplitude * rect_wave + 0.5 * inflow.amplitude / 8
    v_tensor_u = tensor(v_tensor_u[:, :-1], spatial('x,y'))
    v_tensor_v = v.staggered_tensor()[1].numpy('x,y')
    v_tensor_v = tensor(v_tensor_v[:-1, :-2], spatial('x,y'))
    v = StaggeredGrid(math.stack([v_tensor_u, v_tensor_v], dual(vector='x,y')), boundary=v.boundary, bounds=v.bounds,
                      x=sim.resolution[0], y=sim.resolution[1])
    v = advect.semi_lagrangian(v, v, sim.dt)
    v, p = fluid.make_incompressible(velocity=v, obstacles=swarm.as_obstacle_list(),
                                     solve=Solve(rel_tol=1e-05, abs_tol=1e-05, x0=p, max_iterations=100_000))
    # Calculate movement and rotation of swarm members
    for member in swarm.members:
        pressure_profile = sample_pressure_around_obstacle(p=p, member=member, sim=sim)
        left_right = pressure_profile[1] - pressure_profile[3]
        up_down = pressure_profile[0] - pressure_profile[2]
        force_left_right = left_right * 2 * member.radius
        force_up_down = up_down * 2 * member.radius
    return v, p, inflow_field


def run_simulation(velocity_field: Field, pressure_field: Field | None, inflow_field: Field,
                   inflow: Inflow, sim: Simulation, swarm: Swarm, folder_name: str) -> None:
    for time_step in range(1, sim.time_steps + 1):
        print('Sim time:', time_step * sim.dt)
        calc_start = datetime.now()
        velocity_field, pressure_field, inflow_field = step(v=velocity_field, p=pressure_field,
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
    pressure_samples = np.zeros(4)
    for i, angle in enumerate(np.arange(start=0, stop=2 * np.pi, step=np.pi / 4)):
        x = int((member.location['x'] + member.radius * np.cos(np.pi / 2 - angle)) * sim.resolution[0] / sim.length_x)
        y = int((member.location['y'] + member.radius * np.sin(np.pi / 2 - angle)) * sim.resolution[1] / sim.length_y)
        pressure_samples[i] = p.values.x[x].y[y]
    return pressure_samples
