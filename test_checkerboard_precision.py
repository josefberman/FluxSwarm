import numpy as np
import torch
import warnings
from phi.torch.flow import *
import phi
from data_structures import Simulation, Swarm, Inflow, Fluid

# Test with 64 bit precision to see if CG converges
phi.math.set_global_precision(64)

sim = Simulation(length_x=100, length_y=4, resolution=(2000, 80), dt=0.05, total_time=1)
fluid_obj = Fluid(viscosity=3)
inflow = Inflow(frequency=1, amplitude=162)
swarm = Swarm(num_x=4, num_y=4, left_location=49, bottom_location=0.5, member_interval_x=1, member_interval_y=1,
                member_radius=0.25, member_density=5.150,
                member_max_force=3700) 

v = StaggeredGrid(
    0, 
    boundary={'x': ZERO_GRADIENT, 'y': 0}, 
    bounds=Box['x,y', 0:sim.length_x, 0:sim.length_y], 
    x=sim.resolution[0], y=sim.resolution[1]
)
v = v + vec(x=100.0, y=0.0)
obstacles = swarm.as_obstacle_list()

print("Solving with float64...")
try:
    v_proj, p = fluid.make_incompressible(
        velocity=v,
        obstacles=tuple(obstacles),
        solve=Solve(method='CG', x0=None, rel_tol=1e-3, abs_tol=1e-5)
    )
    print("SUCCESS Float64")
except Exception as e:
    print(f"FAILED Float64: {e}")

phi.math.set_global_precision(32)

v = StaggeredGrid(
    0, 
    boundary={'x': ZERO_GRADIENT, 'y': 0}, 
    bounds=Box['x,y', 0:sim.length_x, 0:sim.length_y], 
    x=sim.resolution[0], y=sim.resolution[1]
)
v = v + vec(x=100.0, y=0.0)
obstacles = swarm.as_obstacle_list()

print("Solving with float32...")
try:
    v_proj, p = fluid.make_incompressible(
        velocity=v,
        obstacles=tuple(obstacles),
        solve=Solve(method='CG', x0=None, rel_tol=1e-3, abs_tol=1e-5)
    )
    print("SUCCESS Float32")
except Exception as e:
    print(f"FAILED Float32: {e}")
