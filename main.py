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
from logs import create_run_name, create_folders_for_run, log_parameters, log_hyperparameters
from data_structures import Simulation, Swarm, Inflow, Fluid
from RL import SwarmEnv, run_PPO, run_MOMAPPO
import warnings

warnings.filterwarnings("ignore")

assert backend.default_backend().set_default_device('GPU')


def main(args):
    # -------------- Parameter Definition -------------
    # Simulation dimensions are length=mm and time=second, mass=mg
    sim = Simulation(
        length_x=100,
        length_y=4,
        resolution=(1000, 40),
        dt=args.dt,
        total_time=args.total_time,
        substeps=args.dt_substeps,
    )
    swarm = Swarm(
        num_x=args.swarm_num_x,
        num_y=args.swarm_num_y,
        left_location=49,
        bottom_location=0.5,
        member_interval_x=1,
        member_interval_y=1,
        member_radius=0.25,
        member_density=5.150,
        member_max_force=args.swarm_max_force,
    )  # density in mg/mm^3, force in mg*mm/s^2
    # inflow = Inflow(frequency=0.5, amplitude=10, h_shift=np.pi / 2, v_shift=25)
    inflow = Inflow(
        frequency=1,
        amplitude=args.inflow_velocity,
        upstroke=0.2,
        plateau=0.15,
        downstroke=0.2,
    )  # velocity in mm/s
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
    
    # PARALLELIZATION: Run args.num_envs environments in parallel for better CPU/GPU utilization
    num_envs = args.num_envs
    
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
    
    # Create SubprocVecEnv with args.num_envs parallel environments
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    print(f"[Parallelization] Running {num_envs} environments in parallel (timestamp: {run_timestamp})")
    
    # run_PPO(env, sim.time_steps)
    # HYPERPARAMETERS: Balanced exploration/exploitation
    # - ent_coef=0.01: Mild entropy bonus
    # - clip_coef=0.2: Standard PPO clip range
    # - gamma=0.95: Standard discount factor for longer-horizon planning
    # - lr=3e-4: Standard learning rate
    print(f'Maximum propulsion force per swarm member: {swarm.members[0].max_force} mg*mm/s^2')
    print(f'Fluid velocity: {inflow.amplitude} mm/s')
    run_MOMAPPO(env, sim.time_steps, n_steps=args.n_steps, batch_size=args.batch_size, update_epochs=args.update_epochs, 
                ent_coef=args.ent_coef, clip_coef=args.clip_coef, gamma=args.gamma, lr=args.lr)
    
    # Log hyperparameters for reproducibility
    log_hyperparameters(folder_name, 
                       n_steps=args.n_steps, batch_size=args.batch_size, update_epochs=args.update_epochs,
                       ent_coef=args.ent_coef, clip_coef=args.clip_coef, gamma=args.gamma, lr=args.lr,
                       gae_lambda=0.95, vf_coef=0.5, total_timesteps=sim.time_steps,
                       num_envs=args.num_envs, log_std_init=-1.0)



if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()

    parser = argparse.ArgumentParser(
        description=(
            "Run the FluxSwarm reinforcement learning training loop.\n\n"
            "This script sets up the fluid simulation, swarm layout, and inflow profile, "
            "then trains a MOMAPPO agent using multiple parallel environments. "
            "Use the flags below to override key simulation, swarm, and inflow parameters."
        )
    )
    parser.add_argument(
        '--save-fields',
        dest='save_fields',
        action='store_true',
        default=False,
                        help='Save velocity and pressure fields to npz files (default: False)')
    parser.add_argument(
        '--no-save-fields',
        dest='save_fields',
        action='store_false',
                        help='Disable saving fields to npz files')

    # Simulation controls
    parser.add_argument(
        '--total-time',
        type=float,
        default=100.0,
        help='Total simulation time in seconds for each episode (default: 100.0)',
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=0.05,
        help='Base simulation timestep dt in seconds (default: 0.05)',
    )
    parser.add_argument(
        '--dt-substeps',
        type=int,
        default=25,
        help='Number of substeps per dt used in the simulator (default: 25)',
    )

    # Swarm layout controls
    parser.add_argument(
        '--swarm-num-x',
        type=int,
        default=4,
        help='Number of swarm members along the x-direction (default: 4)',
    )
    parser.add_argument(
        '--swarm-num-y',
        type=int,
        default=4,
        help='Number of swarm members along the y-direction (default: 4)',
    )
    parser.add_argument(
        '--swarm-max-force',
        type=float,
        default=2670/2, # 2670 mg*mm/s^2 is the equivalent force to overcome the drag
        help='Maximum propulsion force per swarm member in mg*mm/s^2 (default: 2670/2)',
    )

    # Inflow controls
    parser.add_argument(
        '--inflow-velocity',
        type=float,
        default=55,
        help='Peak inflow centerline velocity in mm/s (default: 55)',
    )

    # RL controls
    parser.add_argument(
        '--num-envs',
        type=int,
        default=16,
        help='Number of parallel environments (default: 16)',
    )
    parser.add_argument(
        '--n-steps',
        type=int,
        default=128,
        help='Number of steps per environment (default: 128)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size (default: 16)',
    )
    parser.add_argument(
        '--update-epochs',
        type=int,
        default=10,
        help='Number of update epochs (default: 10)',
    )
    parser.add_argument(
        '--ent-coef',
        type=float,
        default=0.01,
        help='Entropy coefficient (default: 0.01)',
    )
    parser.add_argument(
        '--clip-coef',
        type=float,
        default=0.2,
        help='Clip coefficient (default: 0.2)',
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.95,
        help='Discount factor (default: 0.95)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help='Learning rate (default: 3e-4)',
    )
    
    args = parser.parse_args()
    main(args)
