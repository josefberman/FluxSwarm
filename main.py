import os
import pickle

import numpy as np
import phi.field
from phi.flow import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import matplotlib.animation as animation
from glob import glob

# -------------- Parameter Definition -------------
length_x = 100  # cm
length_y = 2.7  # cm
resolution = (1000, 270)
dx = length_x / resolution[0]
dy = length_y / resolution[1]
swarm_num_x = 5
swarm_num_y = 5
swarm_member_rad = 0.04
inflow_freq = 1  # Hz
inflow_amplitude = 23  # cm/s
inflow_radius = length_y / 4
inflow_center_x = inflow_radius + dx
inflow_center_y = length_y / 2
viscosity = 0.0089  # dyne*s/cm^2
dt = 0.1  # s
total_time = 30  # s

# -------------- Container Generation --------------
box = Box['x,y', 0:length_x, 0:length_y]

# -------------- Swarm Generation ------------------
swarm = []

# for i in np.linspace(2, 3, swarm_num_x):
#     for j in np.linspace(0.1, 0.9, swarm_num_y):
#         swarm.append(Obstacle(Sphere(x=i, y=j, radius=swarm_member_rad)))


# ---- initial u and p vector field Generation ----
boundary = {'x': extrapolation.ZERO_GRADIENT, 'y': extrapolation.ZERO}
velocity = StaggeredGrid(0, boundary=boundary, bounds=box, x=resolution[0], y=resolution[1])
inflow_sphere = Sphere(x=inflow_center_x, y=inflow_center_y, radius=inflow_radius)
inflow = CenteredGrid(0, boundary=boundary, bounds=box, x=resolution[0], y=resolution[1])


# -------------- Step Definition -------------------
def step(v, inflow, p, dt):
    inflow = advect.mac_cormack(inflow, v, dt) + inflow_amplitude * resample(inflow_sphere, to=inflow, soft=True)
    inflow_velocity = resample(inflow * (1, 0), to=v)
    v = advect.semi_lagrangian(v, v, dt) + inflow_velocity * dt
    v, p = fluid.make_incompressible(v, swarm, Solve(rel_tol=1e-03, abs_tol=1e-03, x0=p, max_iterations=100_000))
    return v, p, inflow


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


# ----------------- Calculation --------------------
folder_name = f'{datetime.now().year}-{datetime.now().month}-{datetime.now().day}_{datetime.now().hour}-{datetime.now().minute}-{datetime.now().second}'
os.makedirs(f'./run_{folder_name}', exist_ok=True)
os.makedirs(f'./run_{folder_name}/velocity', exist_ok=True)
os.makedirs(f'./run_{folder_name}/pressure', exist_ok=True)
os.makedirs(f'./run_{folder_name}/inflow', exist_ok=True)
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

for time_step in range(1, int(total_time / dt) + 1):
    print('Time Step:', time_step * dt)
    calc_start = datetime.now()
    velocity, pressure, inflow = step(velocity, inflow, None, dt)
    print('Calculation time:', datetime.now() - calc_start)
    fig, axes = plt.subplots(4, 1, figsize=(20, 20))
    fields = [velocity['x'], velocity['y'], pressure, inflow]
    field_names = ['Velocity - x component', 'Velocity - y component', 'Pressure', 'Inflow']
    ax_handlers = []
    # Velocity - x component
    ax_handlers.append(plot_scalar_field_with_patches(field=fields[0], box=None, ax=axes[0], title=field_names[0]))
    fig.colorbar(ax_handlers[-1], ax=axes[0], orientation='vertical', pad=0.04, fraction=0.02)
    # Velocity - y component
    ax_handlers.append(plot_scalar_field_with_patches(field=fields[1], box=None, ax=axes[1], title=field_names[1]))
    fig.colorbar(ax_handlers[-1], ax=axes[1], orientation='vertical', pad=0.04, fraction=0.02)
    # Pressure
    ax_handlers.append(plot_scalar_field_with_patches(field=fields[2], box=None, ax=axes[2], title=field_names[2]))
    fig.colorbar(ax_handlers[-1], ax=axes[2], orientation='vertical', pad=0.04, fraction=0.02)
    ax_handlers.append(plot_scalar_field_with_patches(field=fields[3], box=None, ax=axes[3], title=field_names[3]))
    fig.colorbar(ax_handlers[-1], ax=axes[3], orientation='vertical', pad=0.04, fraction=0.02)
    plt.savefig(f'./run_{folder_name}/figures/timestep_{time_step * dt:.3f}.jpg', dpi=300)
    plt.close(fig)
    phi.field.write(velocity, f'./run_{folder_name}/velocity/{time_step:04}')
    phi.field.write(pressure, f'./run_{folder_name}/pressure/{time_step:04}')
    phi.field.write(inflow, f'./run_{folder_name}/inflow/{time_step:04}')

velocity_file_list = sorted(glob(f'./run_{folder_name}/velocity/*.npz'))
pressure_file_list = sorted(glob(f'./run_{folder_name}/pressure/*.npz'))
inflow_file_list = sorted(glob(f'./run_{folder_name}/inflow/*.npz'))
velocity_data = [np.load(file) for file in velocity_file_list]
pressure_data = [np.load(file) for file in pressure_file_list]
inflow_data = [np.load(file) for file in inflow_file_list]
max_abs_velocity_x = np.max(np.abs([file['data'][:, :, 0] for file in velocity_data]))
max_abs_velocity_y = np.max(np.abs([file['data'][:, :, 1] for file in velocity_data]))
max_abs_pressure = np.max(np.abs([file['data'] for file in pressure_data]))
max_abs_inflow = np.max(np.abs([file['data'] for file in inflow_data]))
fig, ax = plt.subplots(nrows=4, figsize=(10, 20))
im1 = ax[0].imshow(velocity_data[0]['data'][:, :, 0].T, origin='lower', cmap='coolwarm_r', vmin=-max_abs_velocity_x,
                   vmax=max_abs_velocity_x)
im2 = ax[1].imshow(velocity_data[0]['data'][:, :, 1].T, origin='lower', cmap='coolwarm_r', vmin=-max_abs_velocity_y,
                   vmax=max_abs_velocity_y)
im3 = ax[2].imshow(pressure_data[0]['data'].T, origin='lower', cmap='coolwarm_r', vmin=-max_abs_pressure,
                   vmax=max_abs_pressure)
im4 = ax[3].imshow(inflow_data[0]['data'].T, origin='lower', cmap='coolwarm_r', vmin=-max_abs_inflow,
                   vmax=max_abs_inflow)


def update(frame):
    im1.set_data(velocity_data[frame]['data'][:, :, 0].T)
    im2.set_data(velocity_data[frame]['data'][:, :, 1].T)
    im3.set_data(pressure_data[frame]['data'].T)
    im4.set_data(inflow_data[frame]['data'].T)
    return [im1, im2, im3]


ani = animation.FuncAnimation(fig, update, frames=len(pressure_data), interval=200, blit=True)
ani.save(f'./run_{folder_name}/animation.gif', writer='pillow')
