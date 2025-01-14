import matplotlib as mpl
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import PatchCollection
import numpy as np
from phi.flow import *
from data_structures import Simulation, Swarm
from glob import glob
from scipy.signal import savgol_filter
from auxiliary import TO_MMHG


def plot_scalar_field(field, ax, length_x, length_y, title):
    max_magnitude = np.max(np.abs(field.numpy()))
    im = ax.imshow(field.numpy().T, origin='lower', cmap='coolwarm_r', extent=[0, length_x, 0, length_y], aspect=2,
                   vmin=-max_magnitude, vmax=max_magnitude, zorder=1)
    ax.set_title(title)
    return im


def plot_save_current_step(time_step: int, folder_name: str, v_field: Field, p_field: Field,
                           sim: Simulation, swarm: Swarm) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(20, 10))
    fields = [v_field['x'], v_field['y'], p_field * TO_MMHG]
    field_names = [u'Velocity - x component [mm/s]', u'Velocity = y component [mm/s]', 'Pressure [mmHg]']
    ax_handlers = []
    for i in range(0, 3):
        ax_handlers.append(plot_scalar_field(field=fields[i], ax=axes[i], length_x=sim.length_x, length_y=sim.length_y,
                                             title=field_names[i]))
        fig.colorbar(ax_handlers[-1], ax=axes[i], orientation='vertical', pad=0.04, fraction=0.02)
        for member in swarm.members:
            axes[i].add_patch(plt.Circle((member.location['x'], member.location['y']), member.radius, color='k'))
            axes[i].add_patch(
                plt.Arrow(member.location['x'], member.location['y'],
                          member.radius * np.cos(member.location['theta']),
                          member.radius * np.sin(member.location['theta']), color='white', linewidth=0.5))
        # for _x in np.arange(0, sim.length_x, sim.dx):
        #     axes[i].axvline(_x, c='k', linewidth=0.1, alpha=0.2)
        # for _y in np.arange(0, sim.length_y, sim.dy):
        #     axes[i].axhline(_y, c='k', linewidth=0.1, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'../runs/run_{folder_name}/figures/timestep_{time_step * sim.dt:.3f}.jpg', dpi=300)
    plt.close(fig)
    return None


def create_animation_frame_row(fig: plt.Figure, axis, sim: Simulation, swarm: Swarm, imshow_data: np.ndarray,
                               plot_data: np.ndarray, max_abs_value: float, title: str, x_label: str, y_label: str):
    im_handler = axis[0].imshow(imshow_data, origin='lower', cmap='coolwarm_r', vmin=-max_abs_value, vmax=max_abs_value,
                                extent=[0, sim.length_x, 0, sim.length_y], aspect=4, zorder=1)
    axis[0].plot([0, sim.length_x], [int(sim.length_y / 2), int(sim.length_y / 2)], c='k', linestyle='dashed', zorder=2)
    member_patches = []
    direction_patches = []
    for member in swarm.members:
        member_patches.append(axis[0].add_patch(
            plt.Circle((member.previous_locations[0]['x'], member.previous_locations[0]['y']), member.radius,
                       color='k', zorder=3)))
        direction_patches.append(
            axis[0].add_patch(
                matplotlib.patches.FancyArrow(x=member.previous_locations[0]['x'], y=member.previous_locations[0]['y'],
                                              dx=member.radius * np.cos(member.previous_locations[0]['theta']),
                                              dy=member.radius * np.sin(member.previous_locations[0]['theta']),
                                              head_width=0.5, width=0.5, color='white', linewidth=0.5, zorder=4)))
    fig.colorbar(im_handler, ax=axis[0], orientation='vertical', pad=0.04, fraction=0.02)
    axis[0].set_title(title)
    plot_handler, = axis[1].plot(np.linspace(0, sim.length_x, sim.resolution[0]), plot_data, c='k')
    axis[1].set_xlabel(x_label)
    axis[1].set_ylabel(y_label)
    axis[1].set_ylim(-max_abs_value, max_abs_value)
    return im_handler, plot_handler, member_patches, direction_patches


def animate_save_simulation(sim: Simulation, swarm: Swarm, folder_name: str) -> None:
    velocity_file_list = sorted(glob(f'../runs/run_{folder_name}/velocity/*.npz'))
    pressure_file_list = sorted(glob(f'../runs/run_{folder_name}/pressure/*.npz'))
    velocity_data = [np.load(file) for file in velocity_file_list]
    pressure_data = [np.load(file) for file in pressure_file_list]
    max_abs_velocity_x = np.max(np.abs([file['data'][:, :, 0] for file in velocity_data]))
    max_abs_velocity_y = np.max(np.abs([file['data'][:, :, 1] for file in velocity_data]))
    max_abs_pressure = np.max(np.abs([file['data'] for file in pressure_data]))
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 10), gridspec_kw={'width_ratios': [3, 1]})
    v_x_h = create_animation_frame_row(fig=fig, axis=ax[0], sim=sim, swarm=swarm,
                                       imshow_data=velocity_data[0]['data'][:, :, 0].T,
                                       plot_data=savgol_filter(velocity_data[0]['data'][:,
                                                               int(sim.resolution[1] / 2), 0], 100, 5),
                                       max_abs_value=max_abs_velocity_x, title=u'Velocity - x component',
                                       x_label='Tube length [mm]', y_label='Velocity [mm/s]')
    v_y_h = create_animation_frame_row(fig=fig, axis=ax[1], sim=sim, swarm=swarm,
                                       imshow_data=velocity_data[0]['data'][:, :, 1].T,
                                       plot_data=savgol_filter(velocity_data[0]['data'][:,
                                                               int(sim.resolution[1] / 2), 1], 100,
                                                               5), max_abs_value=max_abs_velocity_y,
                                       title=u'Velocity = y component', x_label='Tube length [mm]',
                                       y_label='Velocity [mm/s]')
    p_h = create_animation_frame_row(fig=fig, axis=ax[2], sim=sim, swarm=swarm,
                                     imshow_data=pressure_data[0]['data'].T * TO_MMHG,
                                     plot_data=savgol_filter(pressure_data[0]['data'][:,
                                                             int(sim.resolution[1] / 2)], 100, 5) * TO_MMHG,
                                     max_abs_value=max_abs_pressure * TO_MMHG, title='Pressure',
                                     x_label='Tube length [mm]', y_label='Pressure [mmHg]')
    fig.suptitle(f'Simulation time: {sim.dt} seconds')
    plt.tight_layout()

    def update(frame):
        print(f'{frame=}')
        v_x_h[0].set_data(velocity_data[frame]['data'][:, :, 0].T)
        v_y_h[0].set_data(velocity_data[frame]['data'][:, :, 1].T)
        p_h[0].set_data(pressure_data[frame]['data'].T * TO_MMHG)
        v_x_h[1].set_ydata(
            savgol_filter(velocity_data[frame]['data'][:, int(sim.resolution[1] / 2), 0], 100, 1))
        v_y_h[1].set_ydata(
            savgol_filter(velocity_data[frame]['data'][:, int(sim.resolution[1] / 2), 1], 100, 1))
        p_h[1].set_ydata(
            savgol_filter(pressure_data[frame]['data'][:, int(sim.resolution[1] / 2)] * TO_MMHG, 100, 1))
        for i, member in enumerate(swarm.members):
            v_x_h[2][i].center = member.previous_locations[frame]['x'], member.previous_locations[frame]['y']
            v_y_h[2][i].center = member.previous_locations[frame]['x'], member.previous_locations[frame]['y']
            p_h[2][i].center = member.previous_locations[frame]['x'], member.previous_locations[frame]['y']
            v_x_h[3][i].set_data(x=member.previous_locations[frame]['x'], y=member.previous_locations[frame]['y'],
                                 dx=member.radius * np.cos(member.previous_locations[frame]['theta']),
                                 dy=member.radius * np.sin(member.previous_locations[frame]['theta']))
            v_y_h[3][i].set_data(x=member.previous_locations[frame]['x'], y=member.previous_locations[frame]['y'],
                                 dx=member.radius * np.cos(member.previous_locations[frame]['theta']),
                                 dy=member.radius * np.sin(member.previous_locations[frame]['theta']))
            p_h[3][i].set_data(x=member.previous_locations[frame]['x'], y=member.previous_locations[frame]['y'],
                               dx=member.radius * np.cos(member.previous_locations[frame]['theta']),
                               dy=member.radius * np.sin(member.previous_locations[frame]['theta']))
        fig.suptitle(f'Simulation time: {frame * sim.dt:.2f} seconds')
        return [v_x_h[0], v_y_h[0], p_h[0], v_x_h[1], v_y_h[1], p_h[1], *v_x_h[2], *v_y_h[2], *p_h[2], *v_x_h[3],
                *v_y_h[3], *p_h[3]]

    mpl.rcParams['animation.ffmpeg_path'] = r"C:\Users\assaf\ffmpeg\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe"
    ffmpeg_writer = animation.FFMpegWriter(fps=10, codec='h264', bitrate=-1)
    ani = animation.FuncAnimation(fig, update, frames=len(pressure_data), blit=True, repeat=False)
    ani.save(f'../runs/run_{folder_name}/animation_fast.mp4', ffmpeg_writer, dpi=200)
    ffmpeg_writer = animation.FFMpegWriter(fps=1, codec='h264', bitrate=-1)
    ani = animation.FuncAnimation(fig, update, frames=len(pressure_data), blit=True, repeat=False)
    ani.save(f'../runs/run_{folder_name}/animation_slow.mp4', ffmpeg_writer, dpi=200)
    # ani.save(f'./run_{folder_name}/animation_slow.gif', writer='pillow', fps=1, dpi=300)
    # ani.save(f'./run_{folder_name}/animation_fast.gif', writer='pillow', fps=10, dpi=300)
    return None
