import datetime
import os

# PARALLELIZATION: Enable NumPy/OpenBLAS/MKL multi-threading for better CPU utilization
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 4)
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count() or 4)
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count() or 4)

import argparse

from phi.torch.flow import *
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from plotting import animate_save_simulation, plot_save_locations, plot_save_rewards, plot_save_velocities
from logs import create_run_name, create_folders_for_run, log_parameters
from data_structures import Simulation, Swarm, Inflow, Fluid
from RL import SwarmEnv, run_PPO, run_MOMAPPO
import warnings

warnings.filterwarnings("ignore")

assert backend.default_backend().set_default_device('GPU')


def main(args):
    print('Max force:', 3700)
    # -------------- Parameter Definition -------------
    # Simulation dimensions are length=mm and time=second, mass=mg
    sim = Simulation(length_x=100, length_y=4, resolution=(1000, 40), dt=0.05, total_time=3000)
    swarm = Swarm(num_x=4, num_y=4, left_location=49, bottom_location=0.5, member_interval_x=1, member_interval_y=1,
                    member_radius=0.25, member_density=5.150,
                    member_max_force=3700)  # density in mg/mm^3, force in mg*mm/s^2
    # inflow = Inflow(frequency=0.5, amplitude=10, h_shift=np.pi / 2, v_shift=25)
    inflow = Inflow(frequency=1, amplitude=162, upstroke=0.2, plateau=0.15, downstroke=0.2) # velocity in mm/s
    inflow.center_x = 0
    fluid = Fluid(viscosity=3)  # viscosity of blood in mg/(mm*s)

    # -------------- Container Generation --------------
    box = Box['x,y', 0:sim.length_x, 0:sim.length_y]

    # ---- initial v and p Vector Field Generation ----
    boundary = {'x': ZERO_GRADIENT, 'y': 0}
    velocity_field = StaggeredGrid(0, boundary=boundary, bounds=box, x=sim.resolution[0], y=sim.resolution[1])

    # ----------------- Calculation --------------------
    new_folder_indicator = input('Run in new folder? [y/n]: ')
    if new_folder_indicator=='y':
        folder_name = create_run_name()
        create_folders_for_run(folder_name)
        log_parameters(folder_name=folder_name, sim=sim, swarm=swarm, inflow=inflow, fluid=fluid)
    else:
        folder_name = input('Enter folder_name:')

    # ----------- Reinforcement Learning - PPO ------------------
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # PARALLELIZATION: Run 8 environments in parallel for better CPU/GPU utilization
    num_envs = 8
    
    # Create timestamp for this training run
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def make_env(env_id: int):
        """Factory function that creates an environment with unique timestamped ID."""
        def _init():
            # Each env gets a unique ID with timestamp for distinct folder names
            env_folder = f"{folder_name}/env_{env_id}_{run_timestamp}"
            return SwarmEnv(
                sim=sim, swarm=swarm, fluid=fluid, inflow=inflow,
                folder=env_folder, save_fields=args.save_fields, env_id=env_id
            )
        return _init
    
    # Create SubprocVecEnv with 8 parallel environments
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    print(f"[Parallelization] Running {num_envs} environments in parallel (timestamp: {run_timestamp})")
    
    # run_PPO(env, sim.time_steps)
    run_MOMAPPO(env, sim.time_steps, n_steps=32, batch_size=8, update_epochs=10)

    

    

    # ----------------- Animation --------------------


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    
    parser = argparse.ArgumentParser(description='Run FluxSwarm simulation')
    parser.add_argument('--save-fields', dest='save_fields', action='store_true', default=False,
                        help='Save velocity and pressure fields to npz files (default: False)')
    parser.add_argument('--no-save-fields', dest='save_fields', action='store_false',
                        help='Disable saving fields to npz files')
    
    args = parser.parse_args()
    main(args)
