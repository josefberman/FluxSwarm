import os
import pickle

import phi.field
from phi.flow import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

# -------------- Parameter Definition -------------
length_x = 40  # cm
length_y = 3.2  # cm
resolution = (1024, 255)
swarm_num_x = 5
swarm_num_y = 5
swarm_member_rad = 0.04
inflow_freq = 1  # Hz
inflow_amplitude = 0.1  # cm/s
viscosity = 0.0089  # dyne*s/cm^2
dt = 0.05  # s
total_time = 50  # s

# -------------- Container Generation --------------
box = Box['x,y', 0:length_x, 0:length_y]

# -------------- Swarm Generation ------------------
swarm = []


# for i in np.linspace(2, 3, swarm_num_x):
#     for j in np.linspace(0.1, 0.9, swarm_num_y):
#         swarm.append(Obstacle(Sphere(x=i, y=j, radius=swarm_member_rad)))


# -------------- Step Definition -------------------
def step(velocity_prev, inflow, inflow_amplitude, inflow_freq, dt, t):
    advection_component = advect.semi_lagrangian(velocity_prev, velocity_prev, dt)
    # inflow_component = (inflow_amplitude * 0.5 * math.cos(inflow_freq * math.pi * t) + 0.5) * inflow * dt
    inflow_component = inflow * dt
    velocity_tent = advection_component + inflow_component
    velocity_tent = diffuse.explicit(velocity_tent, viscosity, dt, substeps=2000)
    velocity_next, pressure = fluid.make_incompressible(velocity_tent, swarm, Solve(rel_tol=1e-03, abs_tol=1e-03))
    return velocity_next, pressure, inflow


# ------------- Plotting functions ----------------
def plot_scalar_field_with_patches(field, box, ax, title):
    max_magnitude = np.max(np.abs(field.numpy()))
    im = ax.imshow(field.numpy().T, origin='lower', cmap='coolwarm_r', extent=[0, length_x, 0, length_y], aspect=4,
                   vmin=-max_magnitude, vmax=max_magnitude)
    # lower = box.lower.numpy()
    # upper = box.upper.numpy()
    # width = upper[0] - lower[0]
    # height = upper[1] - lower[1]
    # rect = patches.Rectangle(lower, width, height, linewidth=0, facecolor='white')
    # ax.add_patch(rect)
    ax.set_title(title)
    return im


# ---- initial u and p vector field Generation ----
velocity_boundaries = extrapolation.combine_sides(x=extrapolation.BOUNDARY, y=extrapolation.ZERO)
velocity = StaggeredGrid(0, extrapolation=extrapolation.BOUNDARY, bounds=box, x=resolution[0], y=resolution[1])
dx = length_x / resolution[0]
dy = length_y / resolution[1]
inflow_grid = StaggeredGrid(0, extrapolation=extrapolation.BOUNDARY, bounds=box, x=resolution[0], y=resolution[1])
# pressure_box = Box['x,y', length_x // 2:length_x // 2 + 20 * dx, 50 * dy:length_y - 49 * dy]
pressure_box = Sphere(x=length_x / 2, y=length_y / 2, radius=20 * dy)
inflow = resample(pressure_box, to=inflow_grid, soft=True)
velocity, pressure = fluid.make_incompressible(velocity=velocity, obstacles=swarm)

# ----------------- Calculation --------------------
folder_name = f'{datetime.now().year}-{datetime.now().month}-{datetime.now().day}_{datetime.now().hour}-{datetime.now().minute}-{datetime.now().second}'
os.makedirs(f'./run_{folder_name}', exist_ok=True)
os.makedirs(f'./run_{folder_name}/velocity', exist_ok=True)
os.makedirs(f'./run_{folder_name}/pressure', exist_ok=True)
os.makedirs(f'./run_{folder_name}/figures', exist_ok=True)
with open(f'./run_{folder_name}/configuration.txt', 'w') as f:
    f.write(f'{length_x=}\n')
    f.write(f'{length_y=}\n')
    f.write(f'{resolution=}\n')
    f.write(f'{swarm_num_x=}\n')
    f.write(f'{swarm_num_y=}\n')
    f.write(f'{swarm_member_rad=}\n')
    f.write(f'{inflow_freq=}\n')
    f.write(f'{inflow_amplitude=}\n')
    f.write(f'{viscosity=}\n')
    f.write(f'{dt=}\n')
    f.write(f'{total_time=}\n')

for time_step in range(total_time):
    print('Time Step:', time_step * dt)
    calc_start = datetime.now()
    velocity, pressure, inflow = step(velocity, inflow, inflow_amplitude, inflow_freq, dt, time_step * dt)
    print('Calculation time:', datetime.now() - calc_start)
    fig, axes = plt.subplots(3, 1, figsize=(10, 20))
    fields = [velocity['x'], velocity['y'], pressure]
    field_names = ['Velocity - x component', 'Velocity - y component', 'Pressure']
    ax_handlers = []
    # Velocity - x component
    ax_handlers.append(
        plot_scalar_field_with_patches(field=fields[0], box=pressure_box, ax=axes[0], title=field_names[0]))
    fig.colorbar(ax_handlers[-1], ax=axes[0], orientation='vertical', pad=0.04, fraction=0.02)
    # Velocity - y component
    ax_handlers.append(
        plot_scalar_field_with_patches(field=fields[1], box=pressure_box, ax=axes[1], title=field_names[1]))
    fig.colorbar(ax_handlers[-1], ax=axes[1], orientation='vertical', pad=0.04, fraction=0.02)
    # Pressure
    ax_handlers.append(
        plot_scalar_field_with_patches(field=fields[2], box=pressure_box, ax=axes[2], title=field_names[2]))
    fig.colorbar(ax_handlers[-1], ax=axes[2], orientation='vertical', pad=0.04, fraction=0.02)
    plt.savefig(f'./run_{folder_name}/figures/timestep_{time_step * dt:.3f}.jpg', dpi=300)
    plt.close(fig)
    phi.field.write(velocity, f'./run_{folder_name}/velocity/{time_step * dt:.3f}')
    phi.field.write(pressure, f'./run_{folder_name}/pressure/{time_step * dt:.3f}')
