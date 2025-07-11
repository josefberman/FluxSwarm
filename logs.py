import os
from datetime import datetime
from os import makedirs

from data_structures import Simulation, Swarm, Inflow, Fluid


def create_run_name() -> str:
    """
    Creates a run name based on the date and time of the run in the format "run_%Y-%m-%d_%H-%M-%S".
    :return: Run name
    """
    return f'{datetime.now().year}-{datetime.now().month}-{datetime.now().day}_{datetime.now().hour}-{datetime.now().minute}-{datetime.now().second}'


def create_folders_for_run(folder_name) -> None:
    """
    Creates a main folder for the run with four subfolders for velocity, pressure, inflow data, and figures.
    :param folder_name:
    :return: None
    """
    os.makedirs(f'../runs/run_{folder_name}', exist_ok=True)
    os.makedirs(f'../runs/run_{folder_name}/velocity', exist_ok=True)
    os.makedirs(f'../runs/run_{folder_name}/pressure', exist_ok=True)
    os.makedirs(f'../runs/run_{folder_name}/inflow', exist_ok=True)
    os.makedirs(f'../runs/run_{folder_name}/figures', exist_ok=True)
    os.makedirs(f'../runs/run_{folder_name}/PPO', exist_ok=True)
    os.makedirs(f'../runs/run_{folder_name}/SAC', exist_ok=True)
    return None


def log_parameters(folder_name, sim: Simulation, swarm: Swarm, inflow: Inflow, fluid:Fluid) -> None:
    """
    Logs all initial parameters of the simulation, swarm members, inflow and fluid in a configuration.txt file.
    :param folder_name: Name of the run folder to save configuration file.
    :param sim: Simulation object to save simulation parameters.
    :param swarm: Swarm object to save swarm members parameters.
    :param inflow: Inflow object to save inflow parameters.
    :param fluid: Fluid object to save fluid properties.
    :return: None
    """
    with open(f'../runs/run_{folder_name}/configuration.txt', 'w') as f:
        f.write('Simulation:\n')
        f.write(f'--{sim.length_x=}\n')
        f.write(f'--{sim.length_y=}\n')
        f.write(f'--{sim.resolution=}\n')
        f.write(f'--{sim.dx=}\n')
        f.write(f'--{sim.dy=}\n')
        f.write(f'--{sim.dt=}\n')
        f.write(f'--{sim.total_time=}\n')
        f.write(f'--{sim.time_steps=}\n')
        f.write('\nInflow:\n')
        f.write(f'--{inflow.frequency=}\n')
        f.write(f'--{inflow.amplitude=}\n')
        f.write(f'--{inflow.h_shift=}\n')
        f.write(f'--{inflow.v_shift=}\n')
        f.write('\nFluid:\n')
        f.write(f'--{fluid.viscosity=}\n')
        f.write('\nSwarm:\n')
        f.write(f'--{swarm.num_x=}\n')
        f.write(f'--{swarm.num_y=}\n')
        f.write(f'--{swarm.left_location=}\n')
        f.write(f'--{swarm.bottom_location=}\n')
        f.write(f'--{swarm.member_interval_x=}\n')
        f.write(f'--{swarm.member_interval_y=}\n')
        f.write(f'--{swarm.member_radius=}\n')
        f.write(f'--{swarm.member_max_force=}\n')
        for i,member in enumerate(swarm.members):
            f.write(f'\n**Member {i}\n')
            f.write(f'  --{member.location=}\n')
            f.write(f'  --{member.velocity=}\n')
            f.write(f'  --{member.radius=}\n')
            f.write(f'  --{member.density=}\n')
            f.write(f'  --{member.mass=}\n')
            f.write(f'  --{member.max_force=}\n')
        return None