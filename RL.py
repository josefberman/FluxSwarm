import os
from gymnasium import spaces
import gymnasium as gym
from phi.field import write
from phi.flow import *
import phi.field as field
from data_structures import Simulation, Swarm, Fluid, Inflow
from simulation import step, sample_field_around_obstacles, NUM_PRESSURE_ANGLES
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from plotting import plot_save_locations, plot_save_velocities, plot_save_forces, plot_save_rewards, plot_save_actions, plot_save_fields, plot_save_rewards_objectives
from stable_baselines3 import PPO, SAC
import torch
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import subprocess
import math
import csv
import pandas as pd
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


class SwarmEnv(gym.Env):
    """
    SwarmEnv class for simulating robotic swarm behavior in fluid flow environments.

    This class is a subclass of gym.Env and defines a custom simulation environment
    for training and evaluating swarm robotics systems. It utilizes physics-based
    fluid simulations to evaluate the dynamics of swarm members interacting with
    their environment. The environment supports reinforcement learning agents,
    providing observation and action spaces tailored for swarm control. The environment
    is designed to process simulation steps, compute rewards, and maintain detailed
    state information for the swarm members.

    :ivar metadata: A dictionary specifying the rendering modes available for the
        environment.
    :type metadata: dict
    :ivar pid: Process ID of the current instance, used for unique directory creation.
    :type pid: int
    :ivar sim: The simulation object defining the fluid environment and parameters.
    :type sim: Simulation
    :ivar swarm: The swarm object representing the group of agents in the environment.
    :type swarm: Swarm
    :ivar fluid: The fluid object defining the fluid properties in the simulation.
    :type fluid: Fluid
    :ivar inflow: The inflow object representing boundary conditions for fluid flow.
    :type inflow: Inflow
    :ivar current_time: The current simulation time.
    :type current_time: float
    :ivar episode_time: The elapsed time of the current episode in simulation seconds.
    :type episode_time: float
    :ivar current_timestep: The current timestep within the simulation.
    :type current_timestep: int
    :ivar folder: The folder path used for saving simulation results.
    :type folder: str
    :ivar rewards: A list storing the reward values accumulated during an episode.
    :type rewards: list
    :ivar v: The velocity field of the fluid in the simulation.
    :type v: StaggeredGrid
    :ivar p: The pressure field of the fluid in the simulation, initialized as None.
    :type p: NoneType or numpy.ndarray
    :ivar observation_space: The observation space for the environment, representing
        the state of the swarm.
    :type observation_space: gym.spaces.Box
    :ivar action_space: The action space for the environment, defining possible force
        controls for the agents.
    :type action_space: gym.spaces.Box
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, sim: Simulation, swarm: Swarm, fluid: Fluid, inflow: Inflow, folder: str, save_fields: bool = False, episode_duration: float = 10.0, env_id: int = 0):
        super(SwarmEnv, self).__init__()
        self.env_id = env_id  # Environment index (0 = first env)
        self.pid = os.getpid()
        self.sim = sim
        self.swarm = swarm
        self.fluid = fluid
        self.inflow = inflow
        self.current_time = 0.0
        self.episode_time = 0.0
        self.current_timestep = 0
        self.folder = folder
        self.rewards = []
        # Multi-objective reward weights (can be tuned externally after init)
        self.w_progress = 1.0     # relative x vs fluid: mean(u_fluid_x - v_member_x), normalized
        self.w_energy = 1.0       # energy efficiency: mean(1 - ||F_i||/||F_max||)
        self.w_smooth = 1.0       # maximize action smoothness (cosine similarity)
        # Tracking for logging
        self.last_reward_components = {'progress': 0.0, 'energy_efficiency': 0.0, 'smoothness': 0.0}
        self.reward_components_history = []
        self.objectives_history = []
        box = Box['x,y', 0:sim.length_x, 0:sim.length_y]
        boundary = {'x': ZERO_GRADIENT, 'y': 0}
        self.v = StaggeredGrid(0, boundary=boundary, bounds=box, x=sim.resolution[0], y=sim.resolution[1])
        self.p = None
        self.episode_duration = 10.0

        # Per-episode tracking
        self.episode_index = 0
        self.episode_cum_reward = 0.0
        self.episode_cum_objectives = {'progress': 0.0, 'energy_efficiency': 0.0, 'smoothness': 0.0}
        self.episodes_csv_path = f"run/{self.folder}/episodes_summary_{self.pid}.csv"

        # Field saving for animations
        self.save_fields = save_fields
        self.field_step_counter = 0  # Counter for unique field file names
        self.num_members = len(swarm.members)
        self.last_reward_matrix = np.zeros((self.num_members, 3), dtype=np.float64)
        self.last_objectives_matrix = np.zeros((self.num_members, 3), dtype=np.float64)

        # Define observation space: num_of_members * (position x2, velocity x2, pressure x4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(swarm.members), 8), dtype=np.float32
        )
        # Define action space (force control x2 for each agent)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(swarm.members), 2), dtype=np.float32
        )
        os.makedirs(f'run/{self.folder}/PPO/velocity_{self.pid}', exist_ok=True)
        os.makedirs(f'run/{self.folder}/PPO/pressure_{self.pid}', exist_ok=True)

    def reset(self, seed=None, options=None):
        """
        Resets the simulation environment to an initial state.

        This method initializes or re-initializes the swarm, simulation grid, velocity field,
        and other parameters necessary to start a new episode. Additionally, it preserves
        certain attributes of the previous swarm members for continuity, such as their
        previous locations, velocities, and forces. It returns the initial observation of the
        environment along with an auxiliary dictionary.

        :param seed: A seed value for random number generation, if required.
        :type seed: Optional[int]
        :param options: Additional options for the reset, if applicable.
        :type options: Optional[dict]
        :return: A tuple where the first element is the initial observation of the
            environment, and the second element is an auxiliary dictionary for additional
            information.
        :rtype: Tuple[Any, dict]
        """
        prev_members = self.swarm.members
        prev_swarm = self.swarm
        self.swarm = Swarm(num_x=prev_swarm.num_x, num_y=prev_swarm.num_y, left_location=prev_swarm.left_location,
                           bottom_location=prev_swarm.bottom_location, member_interval_x=prev_swarm.member_interval_x,
                           member_interval_y=prev_swarm.member_interval_y, member_radius=prev_swarm.member_radius,
                           member_density=prev_members[0].density,
                           member_max_force=prev_swarm.member_max_force)  # density in kg/m^3, force in kg*m/s^2
        box = Box['x,y', 0:self.sim.length_x, 0:self.sim.length_y]
        boundary = {'x': ZERO_GRADIENT, 'y': 0}
        self.v = StaggeredGrid(0, boundary=boundary, bounds=box, x=self.sim.resolution[0], y=self.sim.resolution[1])
        from simulation import generate_parabolic_profile_mask
        from phiml.math._tensors import TensorStack
        mask, v_u, v_v = generate_parabolic_profile_mask(self.v, self.sim, self.inflow, t=0.0)
        v_tensor_u = phi.math.expand(mask, v_u.shape['x'])
        stacked_v = TensorStack((v_tensor_u, v_v), dual(vector='x,y'))
        self.v = self.v.with_values(stacked_v)
        self.p = None
        for i, member in enumerate(self.swarm.members):
            member.previous_locations = prev_members[i].previous_locations.copy()
            member.previous_velocities = prev_members[i].previous_velocities.copy()
            member.previous_actions = prev_members[i].previous_actions.copy()
        self.episode_time = 0.0
        # Reset per-episode accumulators
        self.episode_cum_reward = 0.0
        self.episode_cum_objectives = {'progress': 0.0, 'energy_efficiency': 0.0, 'smoothness': 0.0}

        return self._get_observation(), {}

    def step(self, action):
        """
        Advances the simulation by one step based on the given action and computes the observed
        state, reward, and termination conditions.

        The function updates the velocity, pressure, swarm attributes, and the current time based
        on the input action. It also computes the reward for the current timestep, determines whether
        the simulation is complete, and prints the state of each swarm member, including location,
        velocity, and pressure gradient around obstacles.

        :param action: The control action applied to modify or influence the state of the simulation.
        :type action: Any
        :return: A tuple containing the observed state, reward, simulation completion flag, placeholder
            `False` for compatibility, and an empty dictionary.
        :rtype: tuple
        """
        # Advance the simulation
        v_temp, p_temp, swarm_temp = step(
            v=self.v, p=self.p, inflow=self.inflow, sim=self.sim, swarm=self.swarm, fluid_obj=self.fluid,
            t=self.episode_time, force_actions=action
        )
        self.v = v_temp
        self.p = p_temp
        self.swarm = swarm_temp

        self.current_time += self.sim.dt
        self.episode_time += self.sim.dt
        self.current_timestep += 1

        # if self.v is not None:
        #     if self.current_timestep % 10 == 0:
        #         plot_save_current_step(current_time=self.current_time, folder_name=self.folder, v_field=self.v,
        #                                p_field=self.p, sim=self.sim, swarm=self.swarm)

        # Compute reward (scalarized multi-objective)
        reward = self._compute_reward()
        self.rewards.append(reward)
        # Update per-episode accumulators
        self.episode_cum_reward += float(reward)
        self.episode_cum_objectives['progress'] += float(self.last_objectives.get('progress', 0.0))
        self.episode_cum_objectives['energy_efficiency'] += float(self.last_objectives.get('energy_efficiency', 0.0))
        self.episode_cum_objectives['smoothness'] += float(self.last_objectives.get('smoothness', 0.0))

        # Save fields every step, only for first environment (env_id=0) to reduce I/O
        if self.save_fields and self.env_id == 0 and self.v is not None and self.p is not None:
            # Extract velocity components and pressure as numpy arrays
            v_data = self.v.staggered_tensor()  # Returns tuple of (vx, vy)
            vx_np = v_data[0].numpy('x,y').astype(np.float64)
            vy_np = v_data[1].numpy('x,y').astype(np.float64)
            p_np = self.p.values.numpy('x,y').astype(np.float64)
            
            # Create output directory
            output_dir = f"run/{self.folder}/fields"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save each field snapshot
            step_id = f"{self.field_step_counter:06d}"
            field_filename = f"{output_dir}/step_{step_id}.npz"
            np.savez_compressed(
                field_filename,
                vx=vx_np,
                vy=vy_np,
                p=p_np,
                timestep=self.current_time,
                length_x=self.sim.length_x,
                length_y=self.sim.length_y,
                resolution=self.sim.resolution
            )
            
            self.field_step_counter += 1

        # Compute termination signals
        terminated = self._compute_terminated()
        truncated = self._compute_truncated()

        if terminated or truncated:
            self._finalize_episode(terminated=terminated, truncated=truncated)

        # if self.current_timestep % 10 == 0:
        #     write(self.v, f'run/{self.folder}/velocity/velocity_{self.current_timestep}')
        #     write(self.p, f'run/{self.folder}/pressure/pressure_{self.current_timestep}')

        info = {
            'reward_components': self.last_reward_components,
            'objectives': self.last_objectives,
            'reward_matrix': np.asarray(self.last_reward_matrix, dtype=np.float32),
            'objectives_matrix': np.asarray(self.last_objectives_matrix, dtype=np.float32),
            'terminated': terminated,
            'truncated': truncated,
        }
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        """
        Generates and returns the observation array by sampling environmental
        fields and extracting information from members of the swarm.

        The observation array contains the current `x` and `y` positions, `x`
        and `y` components of velocity, and sampled pressure profiles for
        each swarm member. If no pressure field is provided (`self.p` is None),
        the pressure profile values default to an array of zeros.

        :returns: A NumPy array of shape (number_of_members, observation_features)
                  with `float64` as data type, where each row represents the
                  observation of a single member in the swarm.
        :rtype: numpy.ndarray
        """
        obs = []
        if self.p is not None:
            pressure_profiles = sample_field_around_obstacles(f=self.p, swarm=self.swarm, sim=self.sim, n=4)
            if torch.is_tensor(pressure_profiles):
                pressure_profiles = pressure_profiles.detach().cpu().numpy()
        else:
            pressure_profiles = np.zeros((len(self.swarm.members), 4), dtype=np.float32)
        for i, member in enumerate(self.swarm.members):
            pressure_profile = pressure_profiles[i]
            obs.append([
                member.location['x'], member.location['y'],
                member.velocity['x'], member.velocity['y'],
                *pressure_profile
            ])
        return np.array(obs, dtype=np.float32)

    def _compute_truncated(self) -> bool:
        # Truncate if simulation diverged / fields invalid
        return self.v is None

    def _compute_terminated(self) -> bool:
        # Terminate after episode_duration
        return self.episode_time > self.episode_duration

    def _finalize_episode(self, terminated: bool, truncated: bool) -> None:
        """Append a row to episodes CSV with cumulative per-objective rewards and status."""
        status = 'terminated' if terminated and not truncated else ('truncated' if truncated and not terminated else 'both')
        os.makedirs(f'run/{self.folder}', exist_ok=True)
        file_exists = os.path.exists(self.episodes_csv_path)
        with open(self.episodes_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['episode', 'cum_progress', 'cum_energy_efficiency', 'cum_smoothness', 'cum_total_reward', 'status'])
            writer.writerow([
                self.episode_index,
                f"{self.episode_cum_objectives['progress']:.6f}",
                f"{self.episode_cum_objectives['energy_efficiency']:.6f}",
                f"{self.episode_cum_objectives['smoothness']:.6f}",
                f"{self.episode_cum_reward:.6f}",
                status
            ])
        self.episode_index += 1

    def _compute_reward(self):
        n = len(self.swarm.members)
        z = np.zeros((n, 3), dtype=np.float64)

        # Warmup first step: no previous history to compare
        if self.episode_time <= self.sim.dt:
            self.last_reward_matrix = z.copy()
            self.last_objectives_matrix = z.copy()
            self.last_reward_components = {'progress': 0.0, 'energy_efficiency': 0.0, 'smoothness': 0.0}
            self.last_objectives = {'progress': 0.0, 'energy_efficiency': 0.0, 'smoothness': 0.0}
            self.reward_components_history.append(self.last_reward_components.copy())
            self.objectives_history.append(self.last_objectives.copy())
            return 0.0

        if self.v is None:
            self.last_reward_matrix = z.copy()
            self.last_objectives_matrix = z.copy()
            self.last_reward_components = {'progress': 0.0, 'energy_efficiency': 0.0, 'smoothness': 0.0}
            self.last_objectives = {'progress': 0.0, 'energy_efficiency': 0.0, 'smoothness': 0.0}
            self.reward_components_history.append(self.last_reward_components.copy())
            self.objectives_history.append(self.last_objectives.copy())
            return 0.0

        # --- Progress reward ---

        # --- Option 1: use relative velocity to progress reward ---

        # velocity_profiles_far = sample_field_around_obstacles(
        #     f=self.v, swarm=self.swarm, sim=self.sim, n=8, radius_factor=2.0
        # )
        # velocity_profiles_near = sample_field_around_obstacles(
        #     f=self.v, swarm=self.swarm, sim=self.sim, n=8, radius_factor=1.0
        # )
        # if torch.is_tensor(velocity_profiles_far):
        #     vp_far = velocity_profiles_far.detach().cpu().numpy()
        # else:
        #     vp_far = np.asarray(velocity_profiles_far)
        # if torch.is_tensor(velocity_profiles_near):
        #     vp_near = velocity_profiles_near.detach().cpu().numpy()
        # else:
        #     vp_near = np.asarray(velocity_profiles_near)
        # # Average both near and far velocity profiles to estimate a more precise ux_mean
        # ux_far = np.mean(vp_far[:, :, 0], axis=1).astype(np.float64)
        # ux_near = np.mean(vp_near[:, :, 0], axis=1).astype(np.float64)
        # ux_mean = 0.5 * (ux_far + ux_near)
        # vx_mem = np.array([m.velocity['x'] for m in self.swarm.members], dtype=np.float64)
        # r_per_member = ux_mean - vx_mem
        # v_ref = max(abs(float(self.inflow.amplitude)), 1e-9)
        # progress_unw = np.clip(r_per_member / v_ref, -1.0, 1.0)
        # r_prog_w = self.w_progress * progress_unw

        # --- Option 2: use position to progress reward ---

        # progress_unw = np.zeros(n, dtype=np.float64)
        # x_inlet = 0.0
        # x_outlet = self.sim.length_x
        # for idx, member in enumerate(self.swarm.members):
        #     x_mem_initial = member.previous_locations[0]['x']
        #     x_mem_t = member.location['x']
        #     if x_mem_t < x_mem_initial:
        #         progress_unw[idx] = (x_mem_t - x_inlet) / (x_inlet - x_mem_initial) + 1.0  # r(x_inlet)=1.0, r(x_initial)=0.0
        #     else:
        #         progress_unw[idx] = (x_mem_t - x_outlet) / (x_mem_initial - x_outlet) - 1.0  # r(x_outlet)=-1.0, r(x_initial)=0.0
        # r_prog_w = self.w_progress * progress_unw

        # --- Option 3: use absolute velocity to progress reward ---

        vx_mem = np.array([m.velocity['x'] for m in self.swarm.members], dtype=np.float64)
        r_per_member = -1.0 * vx_mem
        v_ref = max(abs(float(self.inflow.amplitude)), 1e-9)
        # progress_unw = np.clip(r_per_member / v_ref, -1.0, 1.0)
        progress_unw = np.clip(r_per_member, -1.0, 1.0)
        r_prog_w = self.w_progress * progress_unw

        # --- Energy efficiency reward ---

        energy_unw = np.zeros(n, dtype=np.float64)
        for idx, member in enumerate(self.swarm.members):
            if len(member.previous_actions) < 1:
                energy_unw[idx] = 1.0
                continue
            a = member.previous_actions[-1]
            ax, ay = float(a['x']), float(a['y'])
            fmax = float(member.max_force)
            f_mag = fmax * math.sqrt(ax * ax + ay * ay)
            energy_unw[idx] = float(np.clip(1.0 - f_mag / f_max, 0.0, 1.0))
        r_en_w = self.w_energy * energy_unw

        # --- Smoothness reward ---

        smooth_unw = np.zeros(n, dtype=np.float64)
        for idx, member in enumerate(self.swarm.members):
            if len(member.previous_actions) >= 2:
                a_prev = member.previous_actions[-2]
                a_curr = member.previous_actions[-1]
                v1 = np.array([a_prev['x'], a_prev['y']], dtype=float)
                v2 = np.array([a_curr['x'], a_curr['y']], dtype=float)
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 < 1e-8 or n2 < 1e-8:
                    cos_sim = 1.0
                else:
                    val = np.dot(v1, v2) / (n1 * n2)
                    if not np.isfinite(val):
                        print("[WARNING: RL.py] NaN/Inf detected in smoothness reward calculation.")
                        val = 1.0
                    cos_sim = float(np.clip(val, -1.0, 1.0))
                smooth_unw[idx] = cos_sim
            else:
                smooth_unw[idx] = 0.0
        r_sm_w = self.w_smooth * smooth_unw

        # Columns: progress, energy_efficiency, smoothness (weighted, for CTDE learning)
        self.last_reward_matrix = np.stack([r_prog_w, r_en_w, r_sm_w], axis=1).astype(np.float64)
        self.last_objectives_matrix = np.stack([progress_unw, energy_unw, smooth_unw], axis=1).astype(np.float64)

        # Scalar summaries (mean over agents) for logging / CSV / callbacks
        self.last_reward_components = {
            'progress': float(np.mean(r_prog_w)),
            'energy_efficiency': float(np.mean(r_en_w)),
            'smoothness': float(np.mean(r_sm_w))
        }
        self.last_objectives = {
            'progress': float(np.mean(progress_unw)),
            'energy_efficiency': float(np.mean(energy_unw)),
            'smoothness': float(np.mean(smooth_unw))
        }
        self.reward_components_history.append(self.last_reward_components.copy())
        self.objectives_history.append(self.last_objectives.copy())
        # Gym scalar: mean over agents of total weighted reward per agent
        total_reward = float(np.mean(np.sum(self.last_reward_matrix, axis=1)))
        return total_reward


    def save_fields_to_disk(self):
        """Legacy method - fields are now saved immediately after each step."""
        # Fields are saved incrementally in step(), so this method is a no-op
        pass

    def render(self, mode='human'):
        pass


class RewardLoggerCallback(BaseCallback):
    """
    Logs rewards and other custom metrics during the training process.

    This class extends the BaseCallback class and is used in reinforcement
    learning environments to log statistics like the mean, minimum, and
    maximum rewards per step. Additionally, it fetches and processes
    custom attributes related to the training environment. Detailed
    behavior is defined within the `_on_step` method.

    :ivar locals: Dictionary of local variables used in the callback during training.
                  This attribute provides access to information like rewards and other
                  runtime data.
    :type locals: dict
    :ivar logger: Logger instance used to record the custom metrics during each step.
    :type logger: Logger
    :ivar training_env: The training environment used during the reinforcment learning
                        process from which custom attributes are fetched.
    :type training_env: Environment
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        rewards = self.locals['rewards']
        mean_r = float(np.mean(rewards))
        min_r = float(np.min(rewards))
        max_r = float(np.max(rewards))
        self.logger.record('custom/step_reward_mean', mean_r)
        self.logger.record('custom/step_reward_min', min_r)
        self.logger.record('custom/step_reward_max', max_r)
        # Log multi-objective components if available
        try:
            comps = self.training_env.get_attr('last_reward_components')
            # VecEnv returns list over envs; single env returns list with one element
            progress_vals = []
            energy_vals = []
            smooth_vals = []
            for c in comps:
                if isinstance(c, dict):
                    progress_vals.append(c.get('progress', 0.0))
                    energy_vals.append(c.get('energy_efficiency', 0.0))
                    smooth_vals.append(c.get('smoothness', 0.0))
            if len(progress_vals) > 0:
                self.logger.record('custom/r_progress', float(np.mean(progress_vals)))
                self.logger.record('custom/r_energy_efficiency', float(np.mean(energy_vals)))
                self.logger.record('custom/r_smooth', float(np.mean(smooth_vals)))
        except Exception:
            pass
        v_attr = self.training_env.get_attr('v')
        p_attr = self.training_env.get_attr('p')
        folder_attr = self.training_env.get_attr('folder')
        pid_attr = self.training_env.get_attr('pid')
        sim_attr = self.training_env.get_attr('sim')
        current_time_attr = self.training_env.get_attr('current_time')
        # if np.round(current_time_attr[0],2) % 1.0 == 0:
        #     for i in range(len(v_attr)):
        #         if v_attr[i] is not None:
        #             phi.field.write(v_attr[i],
        #                             f'run/{folder_attr[i]}/PPO/velocity_{pid_attr[i]}/velocity_{current_time_attr[i]:.3f}')
        #         if p_attr[i] is not None:
        #             phi.field.write(p_attr[i],
        #                             f'run/{folder_attr[i]}/PPO/pressure_{pid_attr[i]}/pressure_{current_time_attr[i]:.3f}')
        # plot_save_fields(v_attr[i], p_attr[i], folder_attr[i], pid_attr[i], current_time_attr[i], sim_attr[i])
        return True


class ActorCriticMO(nn.Module):
    """
    CTDE: decentralized actor (shared π(a_i|o_i)), centralized critic V_i^k(s_joint).
    """
    def __init__(self, num_members: int, obs_local_dim: int = 8, hidden_sizes: tuple[int, int] = (256, 256)):
        super().__init__()
        self.num_members = num_members
        self.obs_local_dim = obs_local_dim
        joint_dim = num_members * obs_local_dim
        h0, h1 = hidden_sizes
        self.actor_torso = nn.Sequential(
            nn.Linear(obs_local_dim, h0),
            nn.Tanh(),
            nn.Linear(h0, h1),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(h1, 2)
        self.log_std = nn.Parameter(torch.ones(2) * -1.0)
        self.critic_torso = nn.Sequential(
            nn.Linear(joint_dim, h0),
            nn.Tanh(),
            nn.Linear(h0, h1),
            nn.Tanh(),
        )
        self.v_progress = nn.Linear(h1, num_members)
        self.v_energy = nn.Linear(h1, num_members)
        self.v_smooth = nn.Linear(h1, num_members)

    def forward(self, obs_local: torch.Tensor):
        """obs_local: (B, N, 8) -> mu, std each (B, N, 2) factorized Gaussian."""
        B, N, D = obs_local.shape
        assert N == self.num_members and D == self.obs_local_dim
        h = self.actor_torso(obs_local.reshape(B * N, D))
        mu = self.mu_head(h).reshape(B, N, 2)
        clamped = torch.clamp(self.log_std, min=-20.0, max=2.0)
        std = torch.exp(clamped).reshape(1, 1, 2).expand(B, N, 2)
        return mu, std

    def values(self, obs_joint: torch.Tensor):
        """obs_joint: (B, N*8) -> three (B, N) value tensors per objective."""
        x = self.critic_torso(obs_joint)
        return self.v_progress(x), self.v_energy(x), self.v_smooth(x)


class RolloutBufferMO:
    def __init__(
        self,
        buffer_size: int,
        num_members: int,
        obs_local_dim: int,
        device: torch.device,
        num_envs: int = 1,
        n_steps: int = 1,
    ):
        self.buffer_size = buffer_size
        self.num_members = num_members
        self.obs_local_dim = obs_local_dim
        self.device = device
        self.num_envs = num_envs
        self.n_steps = n_steps
        assert buffer_size == n_steps * num_envs, "buffer_size must equal n_steps * num_envs"

        self.obs_actor = torch.zeros((buffer_size, num_members, obs_local_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, num_members, 2), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.rew_progress = torch.zeros((buffer_size, num_members), dtype=torch.float32, device=device)
        self.rew_energy = torch.zeros((buffer_size, num_members), dtype=torch.float32, device=device)
        self.rew_smooth = torch.zeros((buffer_size, num_members), dtype=torch.float32, device=device)
        self.val_progress = torch.zeros((buffer_size, num_members), dtype=torch.float32, device=device)
        self.val_energy = torch.zeros((buffer_size, num_members), dtype=torch.float32, device=device)
        self.val_smooth = torch.zeros((buffer_size, num_members), dtype=torch.float32, device=device)
        self.adv_progress = torch.zeros((buffer_size, num_members), dtype=torch.float32, device=device)
        self.adv_energy = torch.zeros((buffer_size, num_members), dtype=torch.float32, device=device)
        self.adv_smooth = torch.zeros((buffer_size, num_members), dtype=torch.float32, device=device)
        self.ret_progress = torch.zeros((buffer_size, num_members), dtype=torch.float32, device=device)
        self.ret_energy = torch.zeros((buffer_size, num_members), dtype=torch.float32, device=device)
        self.ret_smooth = torch.zeros((buffer_size, num_members), dtype=torch.float32, device=device)
        self.ptr = 0

    def add(
        self,
        obs_actor: torch.Tensor,
        action: torch.Tensor,
        logprob: float,
        done: bool,
        rew_progress: torch.Tensor,
        rew_energy: torch.Tensor,
        rew_smooth: torch.Tensor,
        val_progress: torch.Tensor,
        val_energy: torch.Tensor,
        val_smooth: torch.Tensor,
    ):
        i = self.ptr
        dev = self.device
        self.obs_actor[i] = obs_actor.to(dev)
        self.actions[i] = action.to(dev)
        self.logprobs[i] = logprob
        self.dones[i] = float(done)
        self.rew_progress[i] = rew_progress.to(dev)
        self.rew_energy[i] = rew_energy.to(dev)
        self.rew_smooth[i] = rew_smooth.to(dev)
        self.val_progress[i] = val_progress.to(dev)
        self.val_energy[i] = val_energy.to(dev)
        self.val_smooth[i] = val_smooth.to(dev)
        self.ptr += 1

    def _gae_one_objective(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        last_v: torch.Tensor,
        last_done: torch.Tensor,
        gamma: float,
        lam: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """rewards, values: (T, E, N); last_v: (E, N); last_done: (E,). Returns adv, ret (T*E, N)."""
        T, E, N = rewards.shape
        device = self.device
        adv_buf = torch.zeros((T, E, N), dtype=torch.float32, device=device)
        for e in range(E):
            adv_n = torch.zeros(N, dtype=torch.float32, device=device)
            for t in range(T - 1, -1, -1):
                idx = t * E + e
                if t < T - 1:
                    next_v = values[t + 1, e]
                    dn = self.dones[idx]
                else:
                    next_v = last_v[e]
                    dn = last_done[e]
                next_nonterminal = 1.0 - dn
                delta = rewards[t, e] + gamma * next_v * next_nonterminal - values[t, e]
                adv_n = delta + gamma * lam * next_nonterminal * adv_n
                adv_buf[t, e] = adv_n
        adv = adv_buf.reshape(T * E, N)
        ret = adv + values.reshape(T * E, N)
        return adv, ret

    def compute_gae(
        self,
        last_values: tuple,
        last_done: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        """last_values: three tensors (E, N); last_done: (E,) float."""
        T, E = self.n_steps, self.num_envs
        N = self.num_members
        ptr = self.ptr
        last_vp, last_ve, last_vs = last_values

        def norm(x: torch.Tensor):
            if x.std() > 1e-8:
                return (x - x.mean()) / (x.std() + 1e-8)
            return x - x.mean()

        val_flat = self.val_progress[:ptr].view(T, E, N)
        rew = self.rew_progress[:ptr].view(T, E, N)
        adv, ret = self._gae_one_objective(rew, val_flat, last_vp, last_done, gamma, lam)
        self.adv_progress[:ptr] = norm(adv)
        self.ret_progress[:ptr] = ret

        val_flat = self.val_energy[:ptr].view(T, E, N)
        rew = self.rew_energy[:ptr].view(T, E, N)
        adv, ret = self._gae_one_objective(rew, val_flat, last_ve, last_done, gamma, lam)
        self.adv_energy[:ptr] = norm(adv)
        self.ret_energy[:ptr] = ret

        val_flat = self.val_smooth[:ptr].view(T, E, N)
        rew = self.rew_smooth[:ptr].view(T, E, N)
        adv, ret = self._gae_one_objective(rew, val_flat, last_vs, last_done, gamma, lam)
        self.adv_smooth[:ptr] = norm(adv)
        self.ret_smooth[:ptr] = ret

    def get(self, batch_size: int):
        idxs = torch.randperm(self.ptr, device=self.device)
        for start in range(0, self.ptr, batch_size):
            end = min(start + batch_size, self.ptr)
            batch_idx = idxs[start:end]
            obs_joint = self.obs_actor[batch_idx].reshape(len(batch_idx), -1)
            yield (
                self.obs_actor[batch_idx],
                obs_joint,
                self.actions[batch_idx],
                self.logprobs[batch_idx],
                self.dones[batch_idx],
                self.rew_progress[batch_idx],
                self.rew_energy[batch_idx],
                self.rew_smooth[batch_idx],
                self.val_progress[batch_idx],
                self.val_energy[batch_idx],
                self.val_smooth[batch_idx],
                self.adv_progress[batch_idx],
                self.adv_energy[batch_idx],
                self.adv_smooth[batch_idx],
                self.ret_progress[batch_idx],
                self.ret_energy[batch_idx],
                self.ret_smooth[batch_idx],
            )


def pcgrad_merge(model: nn.Module, losses: list[torch.Tensor]):
    """Apply PCGrad to combine gradients from multiple objective losses for actor update."""
    params = [p for p in model.parameters() if p.requires_grad]
    grads = []
    for i, loss in enumerate(losses):
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        g = []
        for p in params:
            if p.grad is None:
                g.append(torch.zeros_like(p).view(-1))
            else:
                g.append(p.grad.view(-1).clone())
        grads.append(torch.cat(g))
    grads = [g for g in grads]
    # PCGrad projection
    merged = grads[0].clone()
    for i in range(len(grads)):
        gi = grads[i].clone()
        for j in range(len(grads)):
            if i == j:
                continue
            gj = grads[j]
            dot = torch.dot(gi, gj)
            if dot < 0:
                norm_sq = gj.norm()**2
                # SAFEGUARD: Prevent divide by zero or NaN if gj norm exploded
                if not torch.isfinite(norm_sq) or norm_sq < 1e-12:
                    print(f"[WARNING: RL.py] PCGrad exploded norm detected on grad {j} - skipping projection")
                    continue
                proj = (dot / (norm_sq + 1e-12)) * gj
                if torch.isnan(proj).any() or torch.isinf(proj).any():
                    print(f"[WARNING: RL.py] PCGrad NaN/Inf projection generated - skipping")
                    continue
                gi = gi - proj
        if i == 0:
            merged = gi
        else:
            merged = merged + gi
    merged = merged / len(grads)
    # Set merged grads back to params
    offset = 0
    for p in params:
        numel = p.numel()
        g = merged[offset:offset+numel].view_as(p)
        if p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.copy_(g)
        offset += numel


def run_PPO(env: SwarmEnv | VecEnv, timesteps: int):
    """
    Executes the Proximal Policy Optimization (PPO) algorithm on a given environment and saves training
    models, visualizations, and logs. The function supports single Swarm Environment instances as well
    as vectorized multiple environment instances. Training progress and logs are saved to TensorBoard,
    and analysis plots are saved to the specified directory.

    :param env: The environment on which PPO will be executed. Can either be a single SwarmEnv or a VecEnv instance.
    :type env: SwarmEnv | VecEnv

    :param timesteps: Total number of timesteps for which the PPO model will be trained. For VecEnv,
        this value is multiplied by the number of environments.
    :type timesteps: int

    :return: None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_steps = 128
    if isinstance(env, VecEnv):
        if os.path.exists(f"run/{env.get_attr('folder')[0]}/swarm_rl_ppo.zip"):
            model = PPO.load(f"run/{env.get_attr('folder')[0]}/swarm_rl_ppo.zip", env=env)
            print('Successfully loaded model')
        else:
            model = PPO('MlpPolicy', env, verbose=2, n_steps=num_steps, batch_size=32,
                        device=device, gamma=0.95, learning_rate=0.0003, ent_coef=0.01, n_epochs=10,
                        tensorboard_log=f"run/{env.get_attr('folder')[0]}/swarm_rl_ppo_tb")
        model.learn(total_timesteps=timesteps * env.num_envs, log_interval=1, progress_bar=True,
                    callback=RewardLoggerCallback(), reset_num_timesteps=False)
        model.save(f"run/{env.get_attr('folder')[0]}/swarm_rl_ppo")
        for env_i in range(env.num_envs):
            date_stamp = f'{datetime.now().year}-{datetime.now().month}-{datetime.now().day}_{datetime.now().hour}-{datetime.now().minute}-{datetime.now().second}'
            os.makedirs(f"run/{env.get_attr('folder')[env_i]}/PPO/{date_stamp}_{env.get_attr('pid')[env_i]}", exist_ok=True)
            plot_save_locations(folder_name=f"{env.get_attr('folder')[env_i]}/PPO/{date_stamp}_{env.get_attr('pid')[env_i]}",
                                sim=env.get_attr('sim')[env_i], swarm=env.get_attr('swarm')[env_i])
            plot_save_velocities(folder_name=f"{env.get_attr('folder')[env_i]}/PPO/{date_stamp}_{env.get_attr('pid')[env_i]}",
                                 sim=env.get_attr('sim')[env_i], swarm=env.get_attr('swarm')[env_i])
            plot_save_rewards(folder_name=f"{env.get_attr('folder')[env_i]}/PPO/{date_stamp}_{env.get_attr('pid')[env_i]}",
                              rewards=env.get_attr('rewards')[env_i], sim=env.get_attr('sim')[env_i])
            plot_save_actions(folder_name=f"{env.get_attr('folder')[env_i]}/PPO/{date_stamp}_{env.get_attr('pid')[env_i]}",
                              sim=env.get_attr('sim')[env_i], swarm=env.get_attr('swarm')[env_i])
            # plot_save_fields(folder_name=f'{env.get_attr('folder')[env_i]}/PPO/', pid=env.get_attr('pid')[env_i])
    elif isinstance(env, SwarmEnv):
        if os.path.exists(f"run/{env.folder}/swarm_rl_ppo.zip"):
            model = PPO.load(f"run/{env.folder}/swarm_rl_ppo.zip", env=env)
            print('Successfully loaded model')
        else:
            model = PPO('MlpPolicy', env, verbose=2, n_steps=num_steps, batch_size=num_steps, device=device, gamma=0.95,
                        tensorboard_log=f"run/{env.folder}/swarm_rl_ppo_tb")
        model.learn(total_timesteps=timesteps, log_interval=1, progress_bar=True, callback=RewardLoggerCallback(),
                    reset_num_timesteps=False)
        model.save(f"run/{env.folder}/swarm_rl_ppo")
        date_stamp = f'{datetime.now().year}-{datetime.now().month}-{datetime.now().day}_{datetime.now().hour}-{datetime.now().minute}-{datetime.now().second}'
        os.makedirs(f"run/{env.folder}/PPO/{date_stamp}", exist_ok=True)
        plot_save_locations(folder_name=f"{env.folder}/PPO/{date_stamp}", sim=env.sim, swarm=env.swarm)
        plot_save_velocities(folder_name=f"{env.folder}/PPO/{date_stamp}", sim=env.sim, swarm=env.swarm)
        plot_save_rewards(folder_name=f"{env.folder}/PPO/{date_stamp}", rewards=env.rewards, sim=env.sim)
        plot_save_actions(folder_name=f"{env.folder}/PPO/{date_stamp}", sim=env.sim, swarm=env.swarm)

def run_SAC(env: SwarmEnv):
    """
    Trains and saves a Soft Actor-Critic (SAC) model for the given swarm environment. This function
    also generates and saves plots for location trajectories, velocities, and rewards during the
    simulation.

    :param env: The swarm environment where the SAC model is trained. Must be an instance of
        `SwarmEnv`.
    :return: None
    """
    model = SAC('MlpPolicy', env, verbose=2, device='cpu', gamma=0.95, tau=0.1)
    model.learn(total_timesteps=env.sim.time_steps, progress_bar=True)
    model.save(f'run/{env.folder}/swarm_rl_sac')

    plot_save_locations(folder_name=f'{env.folder}/SAC', sim=env.sim, swarm=env.swarm)
    plot_save_velocities(folder_name=f'{env.folder}/SAC', sim=env.sim, swarm=env.swarm)
    plot_save_rewards(folder_name=f'{env.folder}/SAC', rewards=env.rewards, sim=env.sim)


def run_MOMAPPO(env, total_timesteps: int,
                n_steps: int = 1024, batch_size: int = 256, update_epochs: int = 10,
                gamma: float = 0.95, gae_lambda: float = 0.95, clip_coef: float = 0.2,
                ent_coef: float = 0.0, vf_coef: float = 0.5, lr: float = 3e-4,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu', open_tensorboard: bool = True,
                resume: bool = True):
    """
    Multi-objective MAPPO (CTDE): shared decentralized actor π(a_i|o_i), centralized
    critics V_i^k(s_joint), per-member reward_matrix (N,3), PCGrad on three policy objectives.
    Supports multiple parallel environments.
    """
    dev = torch.device(device)
    is_vec = isinstance(env, VecEnv)
    
    # Get number of parallel environments
    num_envs = env.num_envs if is_vec else 1
    print(f"[MOMAPPO] Training with {num_envs} parallel environment(s)")
    
    # Reset all environments
    if is_vec:
        obs_all = env.reset()  # shape: (num_envs, num_members, obs_per_member)
    else:
        obs_all, _ = env.reset()
        obs_all = np.expand_dims(obs_all, 0)
    # Resolve folder - use parent folder for shared resources
    if is_vec:
        env_folder = env.get_attr('folder')[0]
        folder_parts = env_folder.split('/')
        folder = '/'.join(folder_parts[:-1]) if len(folder_parts) > 1 else env_folder
    else:
        folder = env.folder
    tb_run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_root_log_dir = f"run/{folder}/MOMAPPO_tb"
    tb_log_dir = f"{tb_root_log_dir}/{tb_run_stamp}"
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    tb_proc = None

    # Try to open TensorBoard server
    if open_tensorboard:
        try:
            tb_proc = subprocess.Popen([
                "tensorboard", "--logdir", tb_root_log_dir, "--port", "6006", "--host", "127.0.0.1"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"TensorBoard: http://127.0.0.1:6006 (logdir {tb_root_log_dir})")
        except Exception as e:
            print(f"Could not launch TensorBoard automatically: {e}. You can run: tensorboard --logdir {tb_root_log_dir}")
    num_members = int(env.action_space.shape[0])
    obs_local_dim = int(env.observation_space.shape[1])
    model = ActorCriticMO(num_members, obs_local_dim).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Checkpointing (resume if requested)
    ckpt_dir = f"run/{folder}/MOMAPPO/models"
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_ckpt = f"{ckpt_dir}/model_latest.pt"
    if resume and os.path.exists(latest_ckpt):
        try:
            ckpt = torch.load(latest_ckpt, map_location=dev)
            model.load_state_dict(ckpt.get('model', {}))
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            print(f"Resumed MOMAPPO from {latest_ckpt}")
        except Exception as e:
            print(f"Warning: failed to load checkpoint {latest_ckpt}: {e}")

    # Adjust total timesteps for parallel envs
    effective_total_timesteps = int(total_timesteps)
    num_updates = max(1, int(math.ceil(effective_total_timesteps / float(n_steps))))
    step_count = 0

    # Create progress bar tracking timesteps
    if tqdm is not None:
        pbar = tqdm(total=num_updates * n_steps * num_envs, desc="MOMAPPO", unit="step", dynamic_ncols=True)
    else:
        pbar = None
    
    # Track last done state for all environments
    last_dones = np.zeros(num_envs, dtype=np.float32)
    # For TensorBoard curves that can be compared directly to end-of-run plots
    # (which use env.rewards and env.objectives_history from env_idx=0).
    env0_cum_reward = 0.0

    for update in range(num_updates):
        # Create buffer sized for all environments
        buffer = RolloutBufferMO(
            n_steps * num_envs, num_members, obs_local_dim, dev,
            num_envs=num_envs, n_steps=n_steps,
        )
        rollout_obj_progress = []
        rollout_obj_energy = []
        rollout_obj_smooth = []

        for t in range(n_steps):
            # Sample actions for ALL environments in parallel
            actions_all = []
            logprobs_all = []
            vals_progress_all = []
            vals_energy_all = []
            vals_smooth_all = []
            obs_tensors = []
            step_reward_env0 = None
            step_obj_env0_progress = 0.0
            step_obj_env0_energy = 0.0
            step_obj_env0_smooth = 0.0
            rm_env0 = None

            for env_idx in range(num_envs):
                obs = obs_all[env_idx]
                obs_actor = torch.tensor(obs, dtype=torch.float32, device=dev)
                obs_joint = obs_actor.reshape(-1)
                obs_tensors.append((obs_actor, obs_joint))

                with torch.no_grad():
                    mu, std = model(obs_actor.unsqueeze(0))
                    dist = Normal(mu, std)
                    action = dist.sample()[0]
                    logprob = dist.log_prob(action).sum()
                    val_progress, val_energy, val_smooth = model.values(obs_joint.unsqueeze(0))

                actions_all.append(action.cpu().numpy())
                logprobs_all.append(float(logprob.cpu()))
                vals_progress_all.append(val_progress[0].cpu())
                vals_energy_all.append(val_energy[0].cpu())
                vals_smooth_all.append(val_smooth[0].cpu())
            
            # Format actions for VecEnv (shape: (num_envs, num_members, 2))
            actions_np = np.array(actions_all)  # (num_envs, num_members, 2)
            actions_np = np.clip(actions_np, -1.0, 1.0)
            actions_np = actions_np.reshape(num_envs, *env.action_space.shape)
            
            # Step all environments
            if is_vec:
                next_obs_all, rewards_batch, dones_batch, infos_batch = env.step(actions_np)
            else:
                next_obs, _, done, _, info = env.step(actions_np[0])
                next_obs_all = np.expand_dims(next_obs, 0)
                dones_batch = np.array([done])
                infos_batch = [info]
            
            # Add experiences from ALL environments to buffer
            for env_idx in range(num_envs):
                obs_actor, _ = obs_tensors[env_idx]
                action_t = torch.tensor(actions_all[env_idx], dtype=torch.float32, device=dev)

                info = infos_batch[env_idx]
                comps = info.get('reward_components', {'progress': 0.0, 'energy_efficiency': 0.0, 'smoothness': 0.0})
                rm = info.get('reward_matrix')
                if rm is None:
                    inv = max(num_members, 1)
                    p = float(comps.get('progress', 0.0)) / inv
                    e = float(comps.get('energy_efficiency', 0.0)) / inv
                    s = float(comps.get('smoothness', 0.0)) / inv
                    rm = np.full((num_members, 3), [p, e, s], dtype=np.float32)
                rm = np.asarray(rm, dtype=np.float32)
                rew_progress = torch.tensor(rm[:, 0], dtype=torch.float32)
                rew_energy = torch.tensor(rm[:, 1], dtype=torch.float32)
                rew_smooth = torch.tensor(rm[:, 2], dtype=torch.float32)
                reward_total = float(np.mean(np.sum(rm, axis=1)))
                obj = info.get('objectives', {'progress': 0.0, 'energy_efficiency': 0.0, 'smoothness': 0.0})
                obj_progress = float(obj.get('progress', 0.0))
                obj_energy = float(obj.get('energy_efficiency', 0.0))
                obj_smooth = float(obj.get('smoothness', 0.0))
                rollout_obj_progress.append(obj_progress)
                rollout_obj_energy.append(obj_energy)
                rollout_obj_smooth.append(obj_smooth)
                if env_idx == 0:
                    rm_env0 = np.array(rm, dtype=np.float32, copy=True)
                    step_reward_env0 = reward_total
                    step_obj_env0_progress = obj_progress
                    step_obj_env0_energy = obj_energy
                    step_obj_env0_smooth = obj_smooth
                buffer.add(
                    obs_actor.cpu(),
                    action_t.cpu(),
                    logprobs_all[env_idx],
                    bool(dones_batch[env_idx]),
                    rew_progress,
                    rew_energy,
                    rew_smooth,
                    vals_progress_all[env_idx],
                    vals_energy_all[env_idx],
                    vals_smooth_all[env_idx],
                )
                step_count += 1

            # Step-level logging for env_idx=0 (to match end-of-run reward plots).
            if step_reward_env0 is not None:
                env0_cum_reward += float(step_reward_env0)
                # writer.add_scalar('rewards/step_total_env0', float(step_reward_env0), step_count)
                # writer.add_scalar('rewards/cumulative_total_env0', float(env0_cum_reward), step_count)
                writer.add_scalar('objectives/env0/progress', float(step_obj_env0_progress), step_count)
                writer.add_scalar('objectives/env0/energy_efficiency', float(step_obj_env0_energy), step_count)
                writer.add_scalar('objectives/env0/smoothness', float(step_obj_env0_smooth), step_count)
                writer.add_scalar('physical/env0/location_x', float(env.get_attr('swarm')[0].members[0].previous_locations[-1]['x']), step_count)
                writer.add_scalar('physical/env0/location_y', float(env.get_attr('swarm')[0].members[0].previous_locations[-1]['y']), step_count)
                writer.add_scalar('physical/env0/velocity_x', float(env.get_attr('swarm')[0].members[0].previous_velocities[-1]['x']), step_count)
                writer.add_scalar('physical/env0/velocity_y', float(env.get_attr('swarm')[0].members[0].previous_velocities[-1]['y']), step_count)
                writer.add_scalar('physical/env0/action_x', float(env.get_attr('swarm')[0].members[0].previous_actions[-1]['x']), step_count)
                writer.add_scalar('physical/env0/action_y', float(env.get_attr('swarm')[0].members[0].previous_actions[-1]['y']), step_count)
                # if rm_env0 is not None and rm_env0.shape[0] == num_members:
                #     for mi in range(num_members):
                #         writer.add_scalar(
                #             f'rewards/env0/member_{mi}/total_weighted',
                #             float(np.sum(rm_env0[mi])),
                #             step_count,
                #         )
                #         writer.add_scalar(
                #             f'rewards/env0/member_{mi}/progress',
                #             float(rm_env0[mi, 0]),
                #             step_count,
                #         )
                #         writer.add_scalar(
                #             f'rewards/env0/member_{mi}/energy_efficiency',
                #             float(rm_env0[mi, 1]),
                #             step_count,
                #         )
                #         writer.add_scalar(
                #             f'rewards/env0/member_{mi}/smoothness',
                #             float(rm_env0[mi, 2]),
                #             step_count,
                #         )
            
            # Update observations for next iteration
            obs_all = next_obs_all
            last_dones = dones_batch.astype(np.float32)
            
            if pbar is not None:
                pbar.update(num_envs)

        last_vp = torch.zeros(num_envs, num_members, dtype=torch.float32, device=dev)
        last_ve = torch.zeros(num_envs, num_members, dtype=torch.float32, device=dev)
        last_vs = torch.zeros(num_envs, num_members, dtype=torch.float32, device=dev)
        with torch.no_grad():
            for env_idx in range(num_envs):
                oj = torch.tensor(obs_all[env_idx].reshape(-1), dtype=torch.float32, device=dev)
                vp, ve, vs = model.values(oj.unsqueeze(0))
                last_vp[env_idx] = vp[0]
                last_ve[env_idx] = ve[0]
                last_vs[env_idx] = vs[0]
        last_done_t = torch.tensor(last_dones, dtype=torch.float32, device=dev)
        buffer.compute_gae((last_vp, last_ve, last_vs), last_done_t, gamma=gamma, lam=gae_lambda)

        # Log unweighted per-objective means from this rollout
        with torch.no_grad():
            r_progress_mean = float(np.mean(rollout_obj_progress)) if len(rollout_obj_progress) > 0 else 0.0
            rc_mean = float(np.mean(rollout_obj_energy)) if len(rollout_obj_energy) > 0 else 0.0
            rs_mean = float(np.mean(rollout_obj_smooth)) if len(rollout_obj_smooth) > 0 else 0.0
        # writer.add_scalar('objectives/progress', r_progress_mean, step_count)
        # writer.add_scalar('objectives/energy_efficiency', rc_mean, step_count)
        # writer.add_scalar('objectives/smoothness', rs_mean, step_count)

        # PPO updates
        epoch_entropies = []
        epoch_value_losses = []
        for epoch in range(update_epochs):
            for batch in buffer.get(batch_size):
                (b_oa, b_oj, b_actions, b_logp_old, _b_dones,
                 _rp, _re, _rs, _b_vp, _b_ve, _b_vs,
                 b_adv_pr, b_adve, b_advs, b_ret_pr, b_rete, b_rets) = batch

                b_oa = b_oa.to(dev)
                b_oj = b_oj.to(dev)
                b_actions = b_actions.to(dev)
                b_logp_old = b_logp_old.to(dev)
                b_adv_pr = b_adv_pr.to(dev)
                b_adve = b_adve.to(dev)
                b_advs = b_advs.to(dev)
                b_ret_pr = b_ret_pr.to(dev)
                b_rete = b_rete.to(dev)
                b_rets = b_rets.to(dev)

                if torch.isnan(b_oa).any() or torch.isnan(b_logp_old).any():
                    print("[MOMAPPO] WARNING: NaN in batch obs/logprobs — skipping batch")
                    continue

                mu, std = model(b_oa)

                if torch.isnan(mu).any():
                    print("[MOMAPPO] WARNING: NaN in actor output (mu) — skipping batch")
                    continue

                dist = Normal(mu, std)
                logp = dist.log_prob(b_actions).sum(-1).sum(-1)
                ratio = torch.exp(logp - b_logp_old).unsqueeze(-1)

                def ppo_obj(adv):
                    unclipped = ratio * adv
                    clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv
                    return -torch.mean(torch.min(unclipped, clipped))

                loss_pi_progress = ppo_obj(b_adv_pr)
                loss_pi_energy = ppo_obj(b_adve)
                loss_pi_smooth = ppo_obj(b_advs)

                entropy = dist.entropy().sum(-1).sum(-1).mean()

                v_progress, v_energy, v_smooth = model.values(b_oj)
                loss_v = 0.5 * (
                    torch.mean((v_progress - b_ret_pr) ** 2) +
                    torch.mean((v_energy - b_rete) ** 2) +
                    torch.mean((v_smooth - b_rets) ** 2)
                )

                # Combine actor gradients via PCGrad (3 objectives), then add critic and entropy
                optimizer.zero_grad(set_to_none=True)
                pcgrad_merge(model, [loss_pi_progress, loss_pi_energy, loss_pi_smooth])
                # Add critic and entropy losses on top of actor grads
                total_aux = vf_coef * loss_v - ent_coef * entropy
                total_aux.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                # Guard: skip optimizer step if any gradient is NaN
                # (clip_grad_norm_ cannot rescue NaN grads — NaN/NaN = NaN scaling)
                grad_nan = any(
                    p.grad is not None and torch.isnan(p.grad).any()
                    for p in model.parameters()
                )
                if grad_nan:
                    print("[MOMAPPO] WARNING: NaN gradient detected — skipping optimizer step")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()

                epoch_entropies.append(float(entropy.detach()))
                epoch_value_losses.append(float(loss_v.detach()))

        # Log training stats per update
        # if len(epoch_entropies) > 0:
        #     writer.add_scalar('loss/value', float(np.mean(epoch_value_losses)), step_count)
        #     writer.add_scalar('stats/entropy', float(np.mean(epoch_entropies)), step_count)

        # Update progress bar with mean rewards from this rollout
        if pbar is not None:
            pbar.set_postfix({
                'steps': step_count,
                'prog': f"{r_progress_mean:.3f}",
                'energy': f"{rc_mean:.3f}",
                'sm': f"{rs_mean:.3f}"
            })
        elif tqdm is None:
            print(f"MOMAPPO update {update+1}/{num_updates}, steps so far {step_count}, prog {r_progress_mean:.3f}, energy {rc_mean:.3f}, sm {rs_mean:.3f}")

    # Fields are saved incrementally after each step, so no need to save here
    # (SubprocVecEnv doesn't expose envs attribute, and fields are already saved)

    # Save plots for each environment
    date_stamp = f'{datetime.now().year}-{datetime.now().month}-{datetime.now().day}_{datetime.now().hour}-{datetime.now().minute}-{datetime.now().second}'
    
    if is_vec:
        # Save plots for EACH parallel environment
        for env_idx in range(num_envs):
            try:
                env_folder = env.get_attr('folder')[env_idx]
                sim_attr = env.get_attr('sim')[env_idx]
                swarm_attr = env.get_attr('swarm')[env_idx]
                rewards_attr = env.get_attr('rewards')[env_idx]
                objectives_attr = env.get_attr('objectives_history')[env_idx]
                pid_attr = env.get_attr('pid')[env_idx]
                
                run_dir = f"run/{env_folder}/MOMAPPO/{date_stamp}"
                os.makedirs(run_dir, exist_ok=True)
                
                plot_save_locations(folder_name=f"{env_folder}/MOMAPPO/{date_stamp}", sim=sim_attr, swarm=swarm_attr)
                plot_save_velocities(folder_name=f"{env_folder}/MOMAPPO/{date_stamp}", sim=sim_attr, swarm=swarm_attr)
                plot_save_forces(folder_name=f"{env_folder}/MOMAPPO/{date_stamp}", sim=sim_attr, swarm=swarm_attr)
                plot_save_rewards(folder_name=f"{env_folder}/MOMAPPO/{date_stamp}", rewards=rewards_attr, sim=sim_attr)
                plot_save_rewards_objectives(folder_name=f"{env_folder}/MOMAPPO/{date_stamp}", sim=sim_attr, objective_history=objectives_attr)
                print(f"Saved plots for env {env_idx} (pid={pid_attr}) to {run_dir}")
            except Exception as e:
                print(f"Warning: Failed to save plots for env {env_idx}: {e}")
    else:
        folder = env.folder
        sim_attr = env.sim
        swarm_attr = env.swarm
        rewards_attr = env.rewards
        objectives_attr = getattr(env, 'objectives_history', [])

        run_dir = f"run/{folder}/MOMAPPO/{date_stamp}"
        os.makedirs(run_dir, exist_ok=True)
        plot_save_locations(folder_name=f"{folder}/MOMAPPO/{date_stamp}", sim=sim_attr, swarm=swarm_attr)
        plot_save_velocities(folder_name=f"{folder}/MOMAPPO/{date_stamp}", sim=sim_attr, swarm=swarm_attr)
        plot_save_forces(folder_name=f"{folder}/MOMAPPO/{date_stamp}", sim=sim_attr, swarm=swarm_attr)
        plot_save_rewards(folder_name=f"{folder}/MOMAPPO/{date_stamp}", rewards=rewards_attr, sim=sim_attr)
        plot_save_rewards_objectives(folder_name=f"{folder}/MOMAPPO/{date_stamp}", sim=sim_attr, objective_history=objectives_attr)

    # Save checkpoints (timestamped and latest)
    ts_ckpt = f"{ckpt_dir}/model_{date_stamp}.pt"
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'num_members': num_members,
        'obs_local_dim': obs_local_dim,
        'total_timesteps': total_timesteps,
    }, ts_ckpt)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'num_members': num_members,
        'obs_local_dim': obs_local_dim,
        'total_timesteps': total_timesteps,
    }, latest_ckpt)
    print(f"Saved MOMAPPO checkpoints to {ts_ckpt} and {latest_ckpt}")
    
    # Close progress bar if it exists
    if pbar is not None:
        pbar.close()
    
    writer.close()
    if tb_proc is not None:
        try:
            tb_proc.terminate()
            tb_proc.wait(timeout=5)
        except Exception:
            try:
                tb_proc.kill()
            except Exception:
                pass
