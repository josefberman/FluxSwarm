import datetime
import os

from phi.torch.flow import *
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from plotting import animate_save_simulation, plot_save_locations, plot_save_rewards, plot_save_velocities
from logs import create_run_name, create_folders_for_run, log_parameters
from data_structures import Simulation, Swarm, Inflow, Fluid
from RL import SwarmEnv, run_PPO, run_SAC
import warnings

warnings.filterwarnings("ignore")

assert backend.default_backend().set_default_device('GPU')


def main():
    for iter_max_force in range(30, 31, 10):
        print('Max force:', iter_max_force)
        # -------------- Parameter Definition -------------
        # Simulation dimensions are length=mm and time=second, mass=mg
        sim = Simulation(length_x=100, length_y=4, resolution=(1000, 40), dt=0.05, total_time=100)
        swarm = Swarm(num_x=2, num_y=2, left_location=49, bottom_location=1, member_interval_x=2, member_interval_y=2,
                      member_radius=0.25, member_density=5.150,
                      member_max_force=iter_max_force)  # density in mg/mm^3, force in mg*mm/s^2
        inflow = Inflow(frequency=0.5, amplitude=50, h_shift=np.pi / 2, v_shift=25)
        inflow.center_x = 0
        fluid = Fluid(viscosity=3.0)  # viscosity of blood in mg/(mm*s)

        # -------------- Container Generation --------------
        box = Box['x,y', 0:sim.length_x, 0:sim.length_y]

        # ---- initial v and p Vector Field Generation ----
        boundary = {'x': ZERO_GRADIENT, 'y': 0}
        velocity_field = StaggeredGrid(0, boundary=boundary, bounds=box, x=sim.resolution[0], y=sim.resolution[1])

        # ----------------- Calculation --------------------
        new_folder_indicator = input('Run in new folder? [y/n]')
        if new_folder_indicator=='y':
            folder_name = create_run_name()
            create_folders_for_run(folder_name)
            log_parameters(folder_name=folder_name, sim=sim, swarm=swarm, inflow=inflow, fluid=fluid)
        else:
            folder_name = input('Enter folder_name:')

        # ----------- Reinforcement Learning - PPO ------------------

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        def make_env():
            return SwarmEnv(sim=sim, swarm=swarm, fluid=fluid, inflow=inflow, folder=folder_name)

        num_envs = 1
        env = SubprocVecEnv([make_env for _ in range(num_envs)])
        run_PPO(env, sim.time_steps)

        # env = SwarmEnv(sim=sim, swarm=swarm, fluid=fluid, inflow=inflow, folder=folder_name)
        # run_SAC(env)

        # with open(f'../runs/{folder_name}/rewards.txt', 'w+') as f:
        #     for i,r in enumerate(env.rewards):
        #         f.write(f'{str(i)},{str(r)}\n')

        # ----------------- Animation --------------------
        # for env_i in range(env.num_envs):
        #     animate_save_simulation(sim=env.get_attr('sim')[env_i], swarm=env.get_attr('swarm')[env_i],
        #                             folder_name=f'{env.get_attr('folder')[env_i]}/PPO/{env_i}', inflow=env.get_attr('inflow')[env_i])
        # animate_save_simulation(sim=env.sim, swarm=env.swarm, folder_name=env.folder, inflow=env.inflow)


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()
