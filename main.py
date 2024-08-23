import numpy as np
from phi.flow import *
import matplotlib.pyplot as plt
from datetime import datetime

# -------------- Parameter Definition -------------
length_x = 20
length_y = 1
resolution = (512, 64)
swarm_num_x = 5
swarm_num_y = 5
swarm_member_rad = 0.04
freq = 0.25
amplitude = 0.2
viscosity = 0.1
dt = 1
total_time = 2

# -------------- Container Generation --------------
box = Box['x,y', 0:length_x, 0:length_y]

# -------------- Swarm Generation ------------------
swarm = []
for i in np.linspace(2, 3, swarm_num_x):
    for j in np.linspace(0.1, 0.9, swarm_num_y):
        swarm.append(Obstacle(Sphere(x=i, y=j, radius=swarm_member_rad)))


# -------------- Step Definition -------------------
def step(velocity_prev, pressure_prev, inflow, dt=0.1):
    velocity_tent = advect.semi_lagrangian(velocity_prev, velocity_prev, dt) + inflow
    velocity_tent = diffuse.explicit(velocity_tent, viscosity, dt, substeps=2000)
    velocity_next, pressure = fluid.make_incompressible(velocity_tent, swarm, solve=Solve(x0=pressure_prev))
    return velocity_next, pressure


# ------ u and p vector field Generation -----------
grid = CenteredGrid(0, extrapolation=extrapolation.BOUNDARY, bounds=box, x=resolution[0], y=resolution[1])
velocity_u = np.zeros([resolution[0], resolution[1]])
velocity_u[:, 0] = 1
velocity_v = np.zeros([resolution[0], resolution[1]])
velocity = np.stack([velocity_u, velocity_v], axis=-1)
velocity = StaggeredGrid(0, extrapolation=extrapolation.BOUNDARY, bounds=box, x=resolution[0], y=resolution[1])
inflow = StaggeredGrid(
    Box['x,y', 0:length_x / resolution[0], length_y / resolution[0]:length_y - length_y / resolution[0]],
    extrapolation=extrapolation.BOUNDARY, bounds=box, x=resolution[0], y=resolution[1])
velocity, pressure = fluid.make_incompressible(velocity, swarm)

# ----------------- Calculation --------------------
for i in range(total_time):
    print('Time Step:', i * dt)
    calc_start = datetime.now()
    velocity, pressure = step(velocity, pressure, inflow, dt)
    print('Calculation time:', datetime.now() - calc_start)
    vis.plot(velocity, size=(10, 1))
    vis.savefig(f'./frame_{i}.jpg', dpi=300)
