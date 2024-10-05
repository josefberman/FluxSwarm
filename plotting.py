import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from phi.flow import *
from data_structures import Simulation, Swarm
from glob import glob
from scipy.signal import savgol_filter, medfilt


def plot_scalar_field(field, ax, length_x, length_y, title):
    max_magnitude = np.max(np.abs(field.numpy()))
    im = ax.imshow(field.numpy().T, origin='lower', cmap='coolwarm_r', extent=[0, length_x, 0, length_y], aspect=2,
                   vmin=-max_magnitude, vmax=max_magnitude, zorder=1)
    ax.set_title(title)
    return im


def plot_save_current_step(time_step: int, folder_name: str, v_field: Field, p_field: Field, inflow_field: Field,
                           sim: Simulation, swarm: Swarm) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(20, 10))
    fields = [v_field['x'], v_field['y'], p_field * 7.5E-9, inflow_field]
    field_names = [u'Velocity - x component [\u03bcm/s]', u'Velocity = y component [\u03bcm/s]', 'Pressure [mmHg]',
                   'Inflow [\u03bcm/s']
    ax_handlers = []
    # axes[0].streamplot(v_field.cells,)
    for i in range(0, 4):
        ax_handlers.append(plot_scalar_field(field=fields[i], ax=axes[i], length_x=sim.length_x, length_y=sim.length_y,
                                             title=field_names[i]))
        fig.colorbar(ax_handlers[-1], ax=axes[i], orientation='vertical', pad=0.04, fraction=0.02)
        for member in swarm.members:
            axes[i].add_patch(plt.Circle((member.location['x'], member.location['y']), member.radius, color='k'))
            axes[i].add_patch(
                plt.Arrow(member.location['x'], member.location['y'], -member.radius * np.cos(member.direction),
                          member.radius * np.sin(member.direction), color='white', linewidth=0.5))
    plt.tight_layout()
    plt.savefig(f'./run_{folder_name}/figures/timestep_{time_step * sim.dt:.3f}.jpg', dpi=300)
    plt.close(fig)
    return None


def create_animation_frame_row(fig: plt.Figure, axis, sim: Simulation, swarm: Swarm, imshow_data: np.ndarray,
                               plot_data: np.ndarray, max_abs_value: float, title: str):
    im_handler = axis[0].imshow(imshow_data, origin='lower', cmap='coolwarm_r', vmin=-max_abs_value, vmax=max_abs_value,
                                extent=[0, sim.length_x, 0, sim.length_y], aspect=4)
    axis[0].plot([0, sim.length_x], [int(sim.length_y / 2), int(sim.length_y / 2)], c='k', linestyle='dashed', zorder=2)
    for member in swarm.members:
        axis[0].add_patch(plt.Circle((member.location['x'], member.location['y']), member.radius, color='k'))
        axis[0].add_patch(
            plt.Arrow(member.location['x'], member.location['y'], -member.radius * np.cos(member.direction),
                      member.radius * np.sin(member.direction), color='white', linewidth=0.5))
    fig.colorbar(im_handler, ax=axis[0], orientation='vertical', pad=0.04, fraction=0.02)
    axis[0].set_title(title)
    plot_handler, = axis[1].plot(np.linspace(0, sim.length_x, sim.resolution[0]), plot_data, c='k')
    return im_handler, plot_handler


def animate_save_simulation(sim: Simulation, swarm: Swarm, folder_name: str) -> None:
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
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(20, 10))
    im1, plot1 = create_animation_frame_row(fig=fig, axis=ax[0], sim=sim, swarm=swarm,
                                            imshow_data=velocity_data[0]['data'][:, :, 0].T,
                                            plot_data=savgol_filter(
                                                velocity_data[0]['data'][:-1, int(sim.resolution[1] / 2), 0], 100, 5),
                                            max_abs_value=max_abs_velocity_x,
                                            title=u'Velocity - x component [\u03bcm/s]')
    im2, plot2 = create_animation_frame_row(fig=fig, axis=ax[1], sim=sim, swarm=swarm,
                                            imshow_data=velocity_data[0]['data'][:, :, 1].T,
                                            plot_data=savgol_filter(
                                                velocity_data[0]['data'][:-1, int(sim.resolution[1] / 2), 1], 100, 5),
                                            max_abs_value=max_abs_velocity_y,
                                            title=u'Velocity = y component [\u03bcm/s]')
    im3, plot3 = create_animation_frame_row(fig=fig, axis=ax[2], sim=sim, swarm=swarm,
                                            imshow_data=pressure_data[0]['data'].T,
                                            plot_data=savgol_filter(
                                                pressure_data[0]['data'][:, int(sim.resolution[1] / 2)], 100,
                                                5) * 7.5E-9, max_abs_value=max_abs_pressure, title='Pressure [mmHg]')
    im4, plot4 = create_animation_frame_row(fig=fig, axis=ax[3], sim=sim, swarm=swarm,
                                            imshow_data=inflow_data[0]['data'].T,
                                            plot_data=savgol_filter(
                                                inflow_data[0]['data'][:, int(sim.resolution[1] / 2)], 100, 5),
                                            max_abs_value=max_abs_inflow, title='Inflow [\u03bcm/s]')
    fig.suptitle(f'Simulation time: 0.0 seconds')
    plt.tight_layout()

    def update(frame):
        im1.set_data(velocity_data[frame]['data'][:, :, 0].T)
        im2.set_data(velocity_data[frame]['data'][:, :, 1].T)
        im3.set_data(pressure_data[frame]['data'].T)
        im4.set_data(inflow_data[frame]['data'].T)
        plot1.set_ydata(savgol_filter(velocity_data[frame]['data'][:-1, int(sim.resolution[1] / 2), 0], 100, 1))
        plot2.set_ydata(savgol_filter(velocity_data[frame]['data'][:-1, int(sim.resolution[1] / 2), 1], 100, 1))
        plot3.set_ydata(savgol_filter(pressure_data[frame]['data'][:, int(sim.resolution[1] / 2)], 100, 1))
        plot4.set_ydata(savgol_filter(inflow_data[frame]['data'][:, int(sim.resolution[1] / 2)], 100, 1))
        fig.suptitle(f'Simulation time: {frame * sim.dt:.1f} seconds')
        for i in range(3):
            ax[i][1].relim()
            ax[i][1].autoscale()
        return [im1, im2, im3, im4, plot1, plot2, plot3, plot4]

    ani = animation.FuncAnimation(fig, update, frames=len(pressure_data), interval=2000, blit=True)
    ani.save(f'./run_{folder_name}/animation_slow.gif', writer='pillow', fps=1)
    ani.save(f'./run_{folder_name}/animation_fast.gif', writer='pillow', fps=10)
    return None
