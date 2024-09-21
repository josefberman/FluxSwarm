from data_structures import Simulation, Swarm, Inflow, Fluid
from phi.flow import *
from datetime import datetime
from plotting import plot_save_current_step
import phi.field


def step(v: Field, p: Field, inflow_field: Field, inflow_sphere: Sphere, inflow: Inflow, sim: Simulation, swarm: Swarm,
         t: float):
    rect_wave = 4 / np.pi * np.sin(inflow.frequency * t)
    for n in range(3, 22, 2):
        rect_wave += 4 / np.pi * 1 / n * np.sin(n * inflow.frequency * t)
    inflow_field = advect.mac_cormack(inflow_field, v, sim.dt) + (
            0.5 * inflow.amplitude * rect_wave + 0.5 * inflow.amplitude) * resample(inflow_sphere, to=inflow_field,
                                                                                    soft=True)
    inflow_velocity = resample(inflow_field * (1, 0), to=v)
    v = advect.semi_lagrangian(v, v, sim.dt) + inflow_velocity * sim.dt
    v, p = fluid.make_incompressible(v, swarm.as_obstacle_list(),
                                     Solve(rel_tol=1e-05, abs_tol=1e-05, x0=p, max_iterations=100_000))
    return v, p, inflow_field


def run_simulation(velocity_field: Field, pressure_field: Field | None, inflow_field: Field, inflow_sphere: Sphere,
                   inflow: Inflow, sim: Simulation, swarm: Swarm, folder_name: str) -> None:
    for time_step in range(1, sim.time_steps + 1):
        print('Sim time:', time_step * sim.dt)
        calc_start = datetime.now()
        velocity_field, pressure_field, inflow_field = step(v=velocity_field, p=pressure_field,
                                                            inflow_field=inflow_field,
                                                            inflow_sphere=inflow_sphere, inflow=inflow, sim=sim,
                                                            swarm=swarm, t=time_step * sim.dt)
        print('Calculation time:', datetime.now() - calc_start)
        plot_save_current_step(time_step=time_step, folder_name=folder_name, v_field=velocity_field,
                               p_field=pressure_field,
                               inflow_field=inflow_field, sim=sim)
        phi.field.write(velocity_field, f'./run_{folder_name}/velocity/{time_step:04}')
        phi.field.write(pressure_field, f'./run_{folder_name}/pressure/{time_step:04}')
        phi.field.write(inflow_field, f'./run_{folder_name}/inflow/{time_step:04}')
    return None
