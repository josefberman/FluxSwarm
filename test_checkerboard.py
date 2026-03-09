import numpy as np
import torch
from phi.torch.flow import *
import phi
from data_structures import Simulation, Inflow, Fluid

sim = Simulation(length_x=100, length_y=4, resolution=(2000, 80), dt=0.05, total_time=1)
fluid_obj = Fluid(viscosity=3)
inflow = Inflow(frequency=1, amplitude=162)

v = StaggeredGrid(
    0, 
    boundary={'x': ZERO_GRADIENT, 'y': 0}, 
    bounds=Box['x,y', 0:sim.length_x, 0:sim.length_y], 
    x=sim.resolution[0], y=sim.resolution[1]
)
# Add an initial delta
v = v + vec(x=100.0, y=0.0)

try:
    reynolds = inflow.amplitude * sim.length_y / fluid_obj.viscosity
    # Test diffusion
    v_diff = diffuse.explicit(v, 1.0 / reynolds, sim.dt)
    print(f"Diffusion max value: {math.max(v_diff.values)}")
    print(f"Diffusion min value: {math.min(v_diff.values)}")
    if math.max(v_diff.values) > 1000 or math.is_nan(math.max(v_diff.values)):
        print("DIFFUSION EXPLODED!")
    
    # Test project
    v_proj, p = fluid.make_incompressible(
        velocity=v,
        obstacles=(),
        solve=Solve(method='CG', x0=None, rel_tol=5e-3, abs_tol=1e-5)
    )
    print(f"Proj max value: {math.max(v_proj.values)}")
except Exception as e:
    print(f"Error: {e}")
