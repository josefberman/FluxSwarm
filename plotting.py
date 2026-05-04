import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from phi.flow import *
from data_structures import Simulation, Swarm, Inflow
from glob import glob
from scipy.signal import savgol_filter
from auxiliary import TO_MMHG, trapezoidal_waveform

# Must match RL.py; stream files live under ``run/<parent_run>/env_<id>_<ts>/`` (not the MOMAPPO subfolder).
TRAINING_LOG_TRAJECTORY = "training_log_trajectory.csv"
TRAINING_LOG_REWARDS = "training_log_rewards.csv"


def _env_run_dir_from_plot_output_folder(folder_name: str) -> str:
    """Map ``.../MOMAPPO/<date>`` output path to the env run directory that holds training logs."""
    p = folder_name.replace("\\", "/")
    if "/MOMAPPO/" in p:
        return p.split("/MOMAPPO/", 1)[0]
    return p


def _path_training_trajectory(folder_name: str) -> str:
    return f"run/{_env_run_dir_from_plot_output_folder(folder_name)}/{TRAINING_LOG_TRAJECTORY}"


def _path_training_rewards(folder_name: str) -> str:
    return f"run/{_env_run_dir_from_plot_output_folder(folder_name)}/{TRAINING_LOG_REWARDS}"


def _read_trajectory_or_fail(folder_name: str) -> pd.DataFrame:
    p = _path_training_trajectory(folder_name)
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"Missing {TRAINING_LOG_TRAJECTORY} at {p}. "
            "Run training so SwarmEnv writes stream logs under run/<parent>/env_<id>_*/."
        )
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"Training trajectory log is empty: {p}")
    return df


def _read_rewards_log_or_fail(folder_name: str) -> pd.DataFrame:
    p = _path_training_rewards(folder_name)
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"Missing {TRAINING_LOG_REWARDS} at {p}. "
            "Run training so SwarmEnv writes stream logs under run/<parent>/env_<id>_*/."
        )
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"Training rewards log is empty: {p}")
    return df



def plot_save_fields(v: Field, p: Field, folder_name: str, pid: int, current_time: float, sim: Simulation):
    """
    This function generates and saves plots for the velocity field components (`x` and `y`) and the
    pressure field at a particular simulation timestep. The fields are visualized as 2D images, where
    the color scale is centered around zero and its range is determined by the maximum absolute value
    of each field component. The resulting plots are saved as a high-resolution image in a specified
    directory, where the directory structure is created if it does not already exist.

    :param v: The velocity field as a dictionary containing separate components ('x', 'y') represented
        as tensors.
    :type v: Field
    :param p: The pressure field represented as a tensor.
    :type p: Field
    :param folder_name: The name of the folder where plots will be saved.
    :type folder_name: str
    :param pid: The process ID or unique identifier for the simulation instance.
    :type pid: int
    :param current_time: The current timestep in the simulation, used for naming the output file.
    :type current_time: float
    :param sim: An object representing the simulation, containing spatial properties like `length_x`
        and `length_y`.
    :type sim: Simulation
    :return: None
    """
    os.makedirs(f'run/{folder_name}/PPO/figures/{pid}', exist_ok=True)
    max_abs_velocity_x = np.max(np.abs(v['x'].numpy()))
    max_abs_velocity_y = np.max(np.abs(v['y'].numpy()))
    max_abs_pressure = np.max(np.abs(p.numpy()))
    fig, ax = plt.subplots(3, 1, figsize=(20, 10))
    ax[0].imshow(v['x'].numpy().T, origin='lower', cmap='coolwarm_r', vmin=-max_abs_velocity_x, vmax=max_abs_velocity_x,
                 extent=[0, sim.length_x, 0, sim.length_y])
    ax[1].imshow(v['y'].numpy().T, origin='lower', cmap='coolwarm_r', vmin=-max_abs_velocity_y, vmax=max_abs_velocity_y,
                 extent=[0, sim.length_x, 0, sim.length_y])
    ax[2].imshow(p.numpy().T * TO_MMHG, origin='lower', cmap='coolwarm_r', vmin=-max_abs_pressure,
                 vmax=max_abs_pressure, extent=[0, sim.length_x, 0, sim.length_y])
    plt.tight_layout()
    plt.savefig(f'run/{folder_name}/PPO/figures/{pid}/timestep_{current_time:.3f}.jpg', dpi=300)
    plt.close(fig)


def plot_save_locations(folder_name: str, sim: Simulation, swarm: Swarm):
    """
    Read ``training_log_trajectory.csv`` from the parent env folder, write
    ``locations.csv`` and ``locations.jpg`` under ``run/{folder_name}/``.

    The trajectory log is produced by :class:`SwarmEnv` (one row per step, includes ``episode``).
    """
    df = _read_trajectory_or_fail(folder_name)
    n = len(swarm.members)
    data_dict: dict = {
        'timestep': df['current_time'].to_numpy(dtype=np.float64),
        'episode': df['episode'].to_numpy(dtype=np.int64),
    }
    for i in range(n):
        data_dict[f'location_{i}_x'] = df[f'location_{i}_x']
        data_dict[f'location_{i}_y'] = df[f'location_{i}_y']
    os.makedirs(f'run/{folder_name}', exist_ok=True)
    pd.DataFrame(data_dict).to_csv(f'run/{folder_name}/locations.csv', index=False)
    time_axis = df['current_time'].to_numpy(dtype=np.float64)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
    loc_cols_x = [df[f'location_{i}_x'].to_numpy() for i in range(n)]
    loc_cols_y = [df[f'location_{i}_y'].to_numpy() for i in range(n)]
    for lx in loc_cols_x:
        axes[0].plot(time_axis, lx, c='#bbbbbb', linewidth=0.5)
    mean_x = np.mean(np.stack(loc_cols_x, axis=0), axis=0)
    mean_y = np.mean(np.stack(loc_cols_y, axis=0), axis=0)
    axes[0].plot(time_axis, mean_x, c='k', linewidth=1)
    axes[0].set_title('x locations', fontweight='bold')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Location [mm]')
    axes[0].set_ylim(0, sim.length_x)
    for ly in loc_cols_y:
        axes[1].plot(time_axis, ly, c='#bbbbbb', linewidth=0.5)
    axes[1].plot(time_axis, mean_y, c='k', linewidth=1)
    axes[1].set_title('y locations', fontweight='bold')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Location [mm]')
    axes[1].set_ylim(0, sim.length_y)
    plt.tight_layout()
    plt.savefig(f'run/{folder_name}/locations.jpg', dpi=300)
    plt.close(fig)


def plot_save_actions(folder_name: str, sim: Simulation, swarm: Swarm):
    """
    Read ``training_log_trajectory.csv`` and write ``actions.csv`` / ``actions.jpg`` under ``run/{folder_name}/``.
    """
    _ = sim
    df = _read_trajectory_or_fail(folder_name)
    n = len(swarm.members)
    data_dict: dict = {
        'timestep': df['current_time'].to_numpy(dtype=np.float64),
        'episode': df['episode'].to_numpy(dtype=np.int64),
    }
    for i in range(n):
        data_dict[f'action_{i}_x'] = df[f'action_{i}_x']
        data_dict[f'action_{i}_y'] = df[f'action_{i}_y']
    os.makedirs(f'run/{folder_name}', exist_ok=True)
    pd.DataFrame(data_dict).to_csv(f'run/{folder_name}/actions.csv', index=False)
    time_axis = df['current_time'].to_numpy(dtype=np.float64)
    ax_x = [df[f'action_{i}_x'].to_numpy() for i in range(n)]
    ax_y = [df[f'action_{i}_y'].to_numpy() for i in range(n)]
    mean_x = np.mean(np.stack(ax_x, axis=0), axis=0)
    mean_y = np.mean(np.stack(ax_y, axis=0), axis=0)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
    for x in ax_x:
        axes[0].plot(time_axis, x, c='#bbbbbb', linewidth=0.5)
    axes[0].plot(time_axis, mean_x, c='k', linewidth=1)
    axes[0].set_title('x actions', fontweight='bold')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Action [mm/s]')
    for y in ax_y:
        axes[1].plot(time_axis, y, c='#bbbbbb', linewidth=0.5)
    axes[1].plot(time_axis, mean_y, c='k', linewidth=1)
    axes[1].set_title('y actions', fontweight='bold')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Action [mm/s]')
    plt.tight_layout()
    plt.savefig(f'run/{folder_name}/actions.jpg', dpi=300)
    plt.close(fig)


def plot_save_velocities(folder_name: str, sim: Simulation, swarm: Swarm):
    """
    Read ``training_log_trajectory.csv`` and write ``velocities.csv`` / ``velocities.jpg``.
    """
    _ = sim
    df = _read_trajectory_or_fail(folder_name)
    n = len(swarm.members)
    data_dict: dict = {
        'timestep': df['current_time'].to_numpy(dtype=np.float64),
        'episode': df['episode'].to_numpy(dtype=np.int64),
    }
    for i in range(n):
        data_dict[f'velocity_{i}_x'] = df[f'velocity_{i}_x']
        data_dict[f'velocity_{i}_y'] = df[f'velocity_{i}_y']
    os.makedirs(f'run/{folder_name}', exist_ok=True)
    pd.DataFrame(data_dict).to_csv(f'run/{folder_name}/velocities.csv', index=False)
    time_axis = df['current_time'].to_numpy(dtype=np.float64)
    vx = [df[f'velocity_{i}_x'].to_numpy() for i in range(n)]
    vy = [df[f'velocity_{i}_y'].to_numpy() for i in range(n)]
    mean_x = np.mean(np.stack(vx, axis=0), axis=0)
    mean_y = np.mean(np.stack(vy, axis=0), axis=0)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
    for x in vx:
        axes[0].plot(time_axis, x, c='#bbbbbb', linewidth=0.5)
    axes[0].plot(time_axis, mean_x, c='k', linewidth=1)
    axes[0].set_title('x velocities', fontweight='bold')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Velocity [mm/s]')
    for y in vy:
        axes[1].plot(time_axis, y, c='#bbbbbb', linewidth=0.5)
    axes[1].plot(time_axis, mean_y, c='k', linewidth=1)
    axes[1].set_title('y velocities', fontweight='bold')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Velocity [mm/s]')
    plt.tight_layout()
    plt.savefig(f'run/{folder_name}/velocities.jpg', dpi=300)
    plt.close(fig)


def plot_save_forces(folder_name: str, sim: Simulation, swarm: Swarm):
    """
    Read actions from ``training_log_trajectory.csv`` and save ``force_i = action_i * max_force`` per member.
    """
    _ = sim
    df = _read_trajectory_or_fail(folder_name)
    n = len(swarm.members)
    time_axis = df['current_time'].to_numpy(dtype=np.float64)
    data_dict: dict = {
        'timestep': time_axis,
        'episode': df['episode'].to_numpy(dtype=np.int64),
    }
    fx_cols: list = []
    fy_cols: list = []
    for i, member in enumerate(swarm.members):
        fmax = float(member.max_force)
        ax = df[f'action_{i}_x'].to_numpy(dtype=np.float64)
        ay = df[f'action_{i}_y'].to_numpy(dtype=np.float64)
        data_dict[f'force_{i}_x'] = ax * fmax
        data_dict[f'force_{i}_y'] = ay * fmax
        fx_cols.append(ax * fmax)
        fy_cols.append(ay * fmax)
    os.makedirs(f'run/{folder_name}', exist_ok=True)
    pd.DataFrame(data_dict).to_csv(f'run/{folder_name}/forces.csv', index=False)
    mean_fx = np.mean(np.stack(fx_cols, axis=0), axis=0)
    mean_fy = np.mean(np.stack(fy_cols, axis=0), axis=0)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
    for x in fx_cols:
        axes[0].plot(time_axis, x, c='#bbbbbb', linewidth=0.5)
    axes[0].plot(time_axis, mean_fx, c='k', linewidth=1)
    axes[0].set_title('x forces', fontweight='bold')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Force [mg*mm/s^2]')
    for y in fy_cols:
        axes[1].plot(time_axis, y, c='#bbbbbb', linewidth=0.5)
    axes[1].plot(time_axis, mean_fy, c='k', linewidth=1)
    axes[1].set_title('y forces', fontweight='bold')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Force [mg*mm/s^2]')
    plt.tight_layout()
    plt.savefig(f'run/{folder_name}/forces.jpg', dpi=300)
    plt.close(fig)


def plot_save_rewards(folder_name: str, sim: Simulation):
    """
    Read ``training_log_rewards.csv`` and write ``rewards.csv`` / ``rewards.jpg`` (no in-memory list).
    """
    _ = sim
    df = _read_rewards_log_or_fail(folder_name)
    t = df['current_time'].to_numpy(dtype=np.float64)
    rw = df['step_reward'].to_numpy(dtype=np.float64)
    os.makedirs(f'run/{folder_name}', exist_ok=True)
    out = pd.DataFrame({
        'timestep': t,
        'episode': df['episode'].to_numpy(dtype=np.int64),
        'reward': rw,
    })
    out.to_csv(f'run/{folder_name}/rewards.csv', index=False)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
    axes[0].plot(t, np.cumsum(rw), c='k', linewidth=0.5)
    axes[0].set_title('Cumulative reward', fontweight='bold')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Cumulative reward')
    axes[1].plot(t, rw, c='k', linewidth=0.5)
    axes[1].set_title('Step reward', fontweight='bold')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Step reward')
    plt.tight_layout()
    plt.savefig(f'run/{folder_name}/rewards.jpg', dpi=300)
    plt.close(fig)


def plot_save_rewards_objectives(folder_name: str, sim: Simulation, title_suffix: str = '',
                                 objective_weights: tuple[float, float, float] = (16.0, 1.0, 1.0)):
    """
    Read unweighted per-step objectives from ``training_log_rewards.csv``; write ``rewards_objectives.*``.
    """
    _ = sim
    df = _read_rewards_log_or_fail(folder_name)
    timesteps = df['current_time'].to_numpy(dtype=np.float64)
    prog = df['progress'].to_numpy(dtype=np.float64).tolist()
    energy = df['energy_efficiency'].to_numpy(dtype=np.float64).tolist()
    sm = df['smoothness'].to_numpy(dtype=np.float64).tolist()

    w_progress, w_energy, w_smooth = objective_weights
    if w_progress > 0 and any(abs(v) > 1.5 for v in prog):
        prog = [float(v) / float(w_progress) for v in prog]
    if w_energy > 0 and any(abs(v) > 1.5 for v in energy):
        energy = [float(v) / float(w_energy) for v in energy]
    if w_smooth > 0 and any(abs(v) > 1.5 for v in sm):
        sm = [float(v) / float(w_smooth) for v in sm]

    os.makedirs(f'run/{folder_name}', exist_ok=True)
    pd.DataFrame({
        'timestep': timesteps,
        'episode': df['episode'].to_numpy(dtype=np.int64),
        'progress': prog,
        'energy_efficiency': energy,
        'smoothness': sm,
    }).to_csv(f'run/{folder_name}/rewards_objectives.csv', index=False)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 12))
    axes[0].plot(timesteps, prog, c='tab:purple', linewidth=1.0, label='progress')
    axes[0].set_title(f'Normalized velocity over time{(" - " + title_suffix) if title_suffix else ""}', fontweight='bold')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Rel. u_x objective')

    axes[1].plot(timesteps, energy, c='tab:green', linewidth=1.0, label='energy_efficiency')
    axes[1].set_title(f'Energy efficiency over time{(" - " + title_suffix) if title_suffix else ""}', fontweight='bold')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Energy efficiency')

    axes[2].plot(timesteps, sm, c='tab:orange', linewidth=1.0, label='smoothness')
    axes[2].set_title(f'Smoothness reward over time{(" - " + title_suffix) if title_suffix else ""}', fontweight='bold')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('Smoothness')

    plt.tight_layout()
    plt.savefig(f'run/{folder_name}/rewards_objectives.jpg', dpi=300)
    plt.close(fig)


def create_animation_frame_row(fig: plt.Figure, axis, sim: Simulation, swarm: Swarm, imshow_data: np.ndarray,
                               plot_data: np.ndarray, max_abs_value: float, title: str, x_label: str, y_label: str):
    """
    Creates a single frame for an animation row consisting of an imshow plot and a line plot. The method configures the
    appearance and content of the provided axes by setting the image data, overlaying swarm member circular patches,
    configuring axis titles, labels, and color bars, and creating the line plot representation.

    :param fig: The matplotlib Figure object containing the entire animation.
    :type fig: plt.Figure
    :param axis: List of axes where the first axis corresponds to the imshow plot and the second to the line plot.
    :param sim: Simulation data with attributes such as length_x, length_y, and resolution required for plotting.
    :type sim: Simulation
    :param swarm: The swarm object containing a list of members and associated location information.
    :type swarm: Swarm
    :param imshow_data: Numpy array representing the 2D data to be displayed using imshow in the first plot.
    :type imshow_data: np.ndarray
    :param plot_data: Numpy array representing the line plot data for the second axis.
    :type plot_data: np.ndarray
    :param max_abs_value: Maximum absolute value for setting the color scale and y-axis limits.
    :type max_abs_value: float
    :param title: Title for the imshow plot.
    :type title: str
    :param x_label: X-axis label for the line plot.
    :type x_label: str
    :param y_label: Y-axis label for the line plot.
    :type y_label: str
    :return: A tuple consisting of the imshow image handler, the line plot handler, and a list of matplotlib.Circle
        patches representing swarm members.
    :rtype: tuple
    """
    im_handler = axis[0].imshow(imshow_data, origin='lower', cmap='coolwarm_r', vmin=-max_abs_value, vmax=max_abs_value,
                                extent=[0, sim.length_x, 0, sim.length_y], aspect=4, zorder=1)
    axis[0].plot([0, sim.length_x], [int(sim.length_y / 2), int(sim.length_y / 2)], c='k', linestyle='dashed', zorder=2)
    member_patches = []
    for member in swarm.members:
        member_patches.append(axis[0].add_patch(
            plt.Circle((member.previous_locations[0]['x'], member.previous_locations[0]['y']), member.radius, color='k',
                       zorder=3)))
    fig.colorbar(im_handler, ax=axis[0], orientation='vertical', pad=0.04, fraction=0.02)
    axis[0].set_title(title, fontweight='bold')
    plot_handler, = axis[1].plot(np.linspace(0, sim.length_x, sim.resolution[0]), plot_data, c='k')
    axis[1].set_xlabel(x_label)
    axis[1].set_ylabel(y_label)
    axis[1].set_ylim(-max_abs_value, max_abs_value)
    return im_handler, plot_handler, member_patches


def animate_save_simulation(sim: Simulation, swarm: Swarm, inflow: Inflow, folder_name: str) -> None:
    """
    Generates and saves animations of a simulation for visualizing velocity components and pressure data
    across time. It reads simulation data from specified file paths, dynamically updates plots for animation,
    and saves the output in specified formats.

    :param sim: The `Simulation` instance containing simulation parameters and resolution.
    :param swarm: The `Swarm` instance representing members moving in the simulation.
    :param inflow: The `Inflow` instance defining inflow characteristics such as amplitude.
    :param folder_name: The name of the folder containing simulation data files used for animation.
    :return: None
    """
    velocity_file_list = sorted(glob(f'run/{folder_name}/velocity/*.npz'))
    pressure_file_list = sorted(glob(f'run/{folder_name}/pressure/*.npz'))
    velocity_data = [np.load(file) for file in velocity_file_list]
    pressure_data = [np.load(file) for file in pressure_file_list]
    max_abs_velocity_x = np.max(np.abs([file['data'][:, :, 0] for file in velocity_data]))
    max_abs_velocity_y = np.max(np.abs([file['data'][:, :, 1] for file in velocity_data]))
    max_abs_pressure = np.max(np.abs([file['data'] for file in pressure_data]))
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 10), gridspec_kw={'width_ratios': [3, 1]})
    v_x_h = create_animation_frame_row(fig=fig, axis=ax[0], sim=sim, swarm=swarm,
                                       imshow_data=velocity_data[0]['data'][:, :, 0].T,
                                       plot_data=savgol_filter(velocity_data[0]['data'][:-1,
                                                               int(sim.resolution[1] / 2), 0], 100, 5),
                                       max_abs_value=max_abs_velocity_x, title=u'Velocity - x component',
                                       x_label='Tube length [mm]', y_label='Velocity [mm/s]')
    v_y_h = create_animation_frame_row(fig=fig, axis=ax[1], sim=sim, swarm=swarm,
                                       imshow_data=velocity_data[0]['data'][:, :, 1].T,
                                       plot_data=savgol_filter(velocity_data[0]['data'][:-1,
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
    inflow_mag = trapezoidal_waveform(t=sim.dt, a=inflow.amplitude, tau=2, h=1.5, v=inflow.amplitude / 2)
    fig.suptitle(f'Simulation time: {sim.dt} seconds.\nInflow: {inflow_mag:.2f} mm/s')
    plt.tight_layout()

    def update(frame):
        v_x_h[0].set_data(velocity_data[frame]['data'][:, :, 0].T)
        v_y_h[0].set_data(velocity_data[frame]['data'][:, :, 1].T)
        p_h[0].set_data(pressure_data[frame]['data'].T * TO_MMHG)
        v_x_h[1].set_ydata(
            savgol_filter(velocity_data[frame]['data'][:-1, int(sim.resolution[1] / 2), 0], 100, 1))
        v_y_h[1].set_ydata(
            savgol_filter(velocity_data[frame]['data'][:-1, int(sim.resolution[1] / 2), 1], 100, 1))
        p_h[1].set_ydata(
            savgol_filter(pressure_data[frame]['data'][:, int(sim.resolution[1] / 2)] * TO_MMHG, 100, 1))
        for i, member in enumerate(swarm.members):
            v_x_h[2][i].center = member.previous_locations[frame]['x'], member.previous_locations[frame]['y']
            v_y_h[2][i].center = member.previous_locations[frame]['x'], member.previous_locations[frame]['y']
            p_h[2][i].center = member.previous_locations[frame]['x'], member.previous_locations[frame]['y']
        inflow_mag = trapezoidal_waveform(t=frame * sim.dt * 5, a=inflow.amplitude, tau=2, h=1.5,
                                          v=inflow.amplitude / 2)
        fig.suptitle(f'Simulation time: {frame * sim.dt * 5:.2f} seconds.\nInflow: {inflow_mag:.2f} mm/s')
        return [v_x_h[0], v_y_h[0], p_h[0], v_x_h[1], v_y_h[1], p_h[1], *v_x_h[2], *v_y_h[2], *p_h[2]]

    mpl.rcParams['animation.ffmpeg_path'] = r"C:\Users\Josef\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffmpeg.exe"
    ffmpeg_writer = animation.FFMpegWriter(fps=10, codec='h264', bitrate=-1)
    ani = animation.FuncAnimation(fig, update, frames=len(pressure_data), blit=True, repeat=False)
    ani.save(f'run/{folder_name}/animation_fast.mp4', ffmpeg_writer, dpi=200)
    ffmpeg_writer = animation.FFMpegWriter(fps=1, codec='h264', bitrate=-1)
    ani = animation.FuncAnimation(fig, update, frames=len(pressure_data), blit=True, repeat=False)
    ani.save(f'run/{folder_name}/animation_slow.mp4', ffmpeg_writer, dpi=200)
    return None
