import os

from phi.flow import ZERO_GRADIENT, StaggeredGrid, Box
import numpy as np
from plotting import animate_save_simulation, plot_save_locations, plot_save_rewards
from logs import create_run_name, create_folders_for_run, log_parameters
from data_structures import Simulation, Swarm, Inflow, Fluid
from RL import SwarmEnv
from stable_baselines3 import PPO

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
env = SwarmEnv(sim=sim, swarm=swarm, fluid=fluid, inflow=inflow, folder=folder_name)

model = PPO('MlpPolicy', env, verbose=2)
model.learn(total_timesteps=env.sim.time_steps)
model.save(f'../runs/run_{env.folder}/swarm_rl_model')

# ----------------- Animation --------------------
plot_save_locations(folder_name=env.folder, sim=env.sim, swarm=env.swarm)
plot_save_rewards(folder_name=folder_name, rewards=env.rewards, sim=env.sim)
animate_save_simulation(sim=env.sim, swarm=env.swarm, folder_name=env.folder, inflow=env.inflow)
