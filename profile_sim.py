import cProfile
import pstats
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from data_structures import Simulation, Swarm, Inflow, Fluid
from RL import SwarmEnv
from phi.torch.flow import *
import numpy as np

def run_profiling():
    sim = Simulation(length_x=100, length_y=4, resolution=(1000, 40), dt=0.05, total_time=2000)
    swarm = Swarm(num_x=4, num_y=4, left_location=49, bottom_location=0.5, member_interval_x=1, member_interval_y=1,
                    member_radius=0.25, member_density=5.150,
                    member_max_force=3700)
    inflow = Inflow(frequency=1, amplitude=162, upstroke=0.2, plateau=0.15, downstroke=0.2)
    inflow.center_x = 0
    fluid = Fluid(viscosity=3)

    env = SwarmEnv(
        sim=sim, swarm=swarm, fluid=fluid, inflow=inflow,
        folder="test_folder", save_fields=False, env_id=0
    )
    env.reset()
    
    # Warmup
    for _ in range(2):
        env.step(np.zeros((16, 2)))
        
    print("Starting profile...")
    profiler = cProfile.Profile()
    profiler.enable()
    
    for _ in range(10):
        env.step(np.zeros((16, 2)))
        
    profiler.disable()
    
    with open('profile_results.txt', 'w') as stream:
        stats = pstats.Stats(profiler, stream=stream).sort_stats('cumtime')
        stats.print_stats(30)
    print("Profile written to profile_results.txt")

if __name__ == '__main__':
    run_profiling()
