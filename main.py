import datetime
import os

from phi.flow import *
# from phi.torch.flow import *
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from plotting import animate_save_simulation, plot_save_locations, plot_save_rewards, plot_save_velocities
from logs import create_run_name, create_folders_for_run, log_parameters
from data_structures import Simulation, Swarm, Inflow, Fluid
from RL import SwarmEnv, run_PPO, run_SAC

# print(backend.default_backend().list_devices('GPU'))
# print(backend.default_backend().list_devices('CPU'))
# assert backend.default_backend().set_default_device('GPU')
assert backend.default_backend().set_default_device('CPU')


def main():
    # -------------- Parameter Definition -------------
    # Simulation dimensions are length=mm and time=second, mass=mg
    sim = Simulation(length_x=720, length_y=36, resolution=(1800, 90), dt=0.05, total_time=1)
    swarm = Swarm(num_x=3, num_y=3, left_location=480, bottom_location=8.1, member_interval_x=6.3, member_interval_y=6.3,
                  member_radius=1.8, member_density=5.150, member_max_force=100)  # density in mg/mm^3, force in mg*mm/s^2
    # max force 0.017 mg*mm/s^2
    inflow = Inflow(frequency=2 * np.pi, amplitude=20, radius=sim.length_y / 2, center_y=sim.length_y / 2)
    inflow.center_x = 0
    fluid = Fluid(viscosity=0.89)  # viscosity of water in mg/(mm*s)

    # -------------- Container Generation --------------
    box = Box['x,y', 0:sim.length_x, 0:sim.length_y]

    # ---- initial v and p Vector Field Generation ----
    boundary = {'x': ZERO_GRADIENT, 'y': 0}
    velocity_field = StaggeredGrid(0, boundary=boundary, bounds=box, x=sim.resolution[0], y=sim.resolution[1])

    # ----------------- Calculation --------------------
    folder_name = create_run_name()
    create_folders_for_run(folder_name)
    log_parameters(folder_name=folder_name, sim=sim, swarm=swarm, inflow=inflow, fluid=fluid)

    # run_simulation(velocity_field=velocity_field, pressure_field=None, inflow=inflow, sim=sim,
    #                swarm=swarm, fluid_obj=fluid, folder_name=folder_name)

    # ------------ Reinforcement learning - Random ------------

    # env = SwarmEnv(sim=sim, swarm=swarm, fluid=fluid, inflow=inflow, folder=folder_name)

    # obs, _ = env.reset()
    # for _ in range(int(sim.total_time / sim.dt)):
    #     action = env.action_space.sample()
    #     for i, member in enumerate(env.swarm.members):
    #         member.previous_forces.append(action[i])
    #     obs, reward, done, _, _ = env.step(action)

    # ----------- Reinforcement Learning - PPO ------------------

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    def make_env():
        return SwarmEnv(sim=sim, swarm=swarm, fluid=fluid, inflow=inflow, folder=folder_name)
    # env = SwarmEnv(sim=sim, swarm=swarm, fluid=fluid, inflow=inflow, folder=folder_name)
    num_envs = 6
    env = SubprocVecEnv([make_env for _ in range(num_envs)])
    run_PPO(env, sim.time_steps)

    # env = SwarmEnv(sim=sim, swarm=swarm, fluid=fluid, inflow=inflow, folder=folder_name)
    # run_SAC(env)

    # with open(f'../runs/run_{folder_name}/rewards.txt', 'w+') as f:
    #     for i,r in enumerate(env.rewards):
    #         f.write(f'{str(i)},{str(r)}\n')

    # ----------------- Animation --------------------
    # animate_save_simulation(sim=env.sim, swarm=env.swarm, folder_name=env.folder, inflow=env.inflow)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()