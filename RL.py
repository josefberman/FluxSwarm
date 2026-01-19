import os
from gymnasium import spaces
import gymnasium as gym
from phi.field import write
from phi.flow import *
import phi.field as field
from data_structures import Simulation, Swarm, Fluid, Inflow
from simulation import step, sample_field_around_obstacle
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from plotting import plot_save_locations, plot_save_velocities, plot_save_rewards, plot_save_actions, plot_save_fields, plot_save_rewards_objectives
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

    def __init__(self, sim: Simulation, swarm: Swarm, fluid: Fluid, inflow: Inflow, folder: str, save_fields: bool = False, episode_duration: float = 10.0):
        super(SwarmEnv, self).__init__()
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
        self.w_prog = 1.0         # center-of-mass progress along x
        self.w_cohesion = 1.0     # minimize average distance to COM
        self.w_smooth = 1.0        # maximize action smoothness (cosine similarity)
        # Tracking for logging
        self.last_reward_components = {'progress': 0.0, 'cohesion': 0.0, 'smoothness': 0.0}
        self.reward_components_history = []
        self.objectives_history = []
        box = Box['x,y', 0:sim.length_x, 0:sim.length_y]
        boundary = {'x': ZERO_GRADIENT, 'y': 0}
        self.v = StaggeredGrid(0, boundary=boundary, bounds=box, x=sim.resolution[0], y=sim.resolution[1])
        self.p = None
        self.episode_duration = 50.0

        # Per-episode tracking
        self.episode_index = 0
        self.episode_cum_reward = 0.0
        self.episode_cum_objectives = {'progress': 0.0, 'cohesion': 0.0, 'smoothness': 0.0}
        self.episodes_csv_path = f"../runs/{self.folder}/episodes_summary_{self.pid}.csv"
        
        # Field saving for animations
        self.save_fields = save_fields
        self.field_step_counter = 0  # Counter for unique field file names

        # Define observation space: num_of_members * (position x2, velocity x2, pressure x4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(swarm.members), 8), dtype=np.float64
        )
        # Define action space (force control x2 for each agent)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(swarm.members), 2), dtype=np.float64
        )
        os.makedirs(f'../runs/{self.folder}/PPO/velocity_{self.pid}', exist_ok=True)
        os.makedirs(f'../runs/{self.folder}/PPO/pressure_{self.pid}', exist_ok=True)

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
        self.p = None
        for i, member in enumerate(self.swarm.members):
            member.previous_locations = prev_members[i].previous_locations.copy()
            member.previous_velocities = prev_members[i].previous_velocities.copy()
            member.previous_actions = prev_members[i].previous_actions.copy()
        self.episode_time = 0.0
        # Reset per-episode accumulators
        self.episode_cum_reward = 0.0
        self.episode_cum_objectives = {'progress': 0.0, 'cohesion': 0.0, 'smoothness': 0.0}
        
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
        self.episode_cum_objectives['cohesion'] += float(self.last_objectives.get('cohesion', 0.0))
        self.episode_cum_objectives['smoothness'] += float(self.last_objectives.get('smoothness', 0.0))

        # Save fields immediately if requested
        if self.save_fields and self.v is not None and self.p is not None:
            # Extract velocity components and pressure as numpy arrays
            v_data = self.v.staggered_tensor()  # Returns tuple of (vx, vy)
            vx_np = v_data[0].numpy('x,y').astype(np.float64)
            vy_np = v_data[1].numpy('x,y').astype(np.float64)
            p_np = self.p.values.numpy('x,y').astype(np.float64)
            
            # Create output directory
            output_dir = f"../runs/{self.folder}/fields"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save each field snapshot immediately
            step_id = f"{self.field_step_counter:06d}"
            field_filename = f"{output_dir}/step_{step_id}.npz"
            np.savez_compressed(
                field_filename,
                vx=vx_np,
                vy=vy_np,
                p=p_np,
                timestep=self.episode_time,
                length_x=self.sim.length_x,
                length_y=self.sim.length_y,
                resolution=self.sim.resolution
            )

            # # Also export to Excel with separate sheets for each field component
            # excel_path = f"{output_dir}/step_{step_id}.xlsx"
            # with pd.ExcelWriter(excel_path) as writer:
            #     pd.DataFrame(vx_np).to_excel(writer, sheet_name="v_u", index=False, header=False)
            #     pd.DataFrame(vy_np).to_excel(writer, sheet_name="v_v", index=False, header=False)
            #     pd.DataFrame(p_np).to_excel(writer, sheet_name="p", index=False, header=False)
            
            self.field_step_counter += 1

        # Compute termination signals
        terminated = self._compute_terminated()
        truncated = self._compute_truncated()

        if terminated or truncated:
            self._finalize_episode(terminated=terminated, truncated=truncated)

        # if self.current_timestep % 10 == 0:
        #     write(self.v, f'../runs/{self.folder}/velocity/velocity_{self.current_timestep}')
        #     write(self.p, f'../runs/{self.folder}/pressure/pressure_{self.current_timestep}')

        info = {
            'reward_components': self.last_reward_components,
            'objectives': self.last_objectives,  # unweighted objective vector for MOMARL
            'terminated': terminated,
            'truncated': truncated,
            'member_progress': getattr(self, 'last_member_progress_norm', [])
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
        for member in self.swarm.members:
            if self.p is not None:
                pressure_profile = sample_field_around_obstacle(f=self.p, member=member, sim=self.sim, n=4)
            else:
                pressure_profile = np.zeros(4, dtype=object)
            obs.append([
                member.location['x'], member.location['y'],
                member.velocity['x'], member.velocity['y'],
                *pressure_profile
            ])
        return np.array(obs, dtype=np.float64)

    def _compute_truncated(self) -> bool:
        # Truncate if simulation diverged / fields invalid
        return self.v is None

    def _compute_terminated(self) -> bool:
        # Terminate after 10 seconds
        return self.episode_time > self.episode_duration

    def _finalize_episode(self, terminated: bool, truncated: bool) -> None:
        """Append a row to episodes CSV with cumulative per-objective rewards and status."""
        status = 'terminated' if terminated and not truncated else ('truncated' if truncated and not terminated else 'both')
        os.makedirs(f'../runs/{self.folder}', exist_ok=True)
        file_exists = os.path.exists(self.episodes_csv_path)
        with open(self.episodes_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['episode', 'cum_progress', 'cum_cohesion', 'cum_smoothness', 'cum_total_reward', 'status'])
            writer.writerow([
                self.episode_index,
                f"{self.episode_cum_objectives['progress']:.6f}",
                f"{self.episode_cum_objectives['cohesion']:.6f}",
                f"{self.episode_cum_objectives['smoothness']:.6f}",
                f"{self.episode_cum_reward:.6f}",
                status
            ])
        self.episode_index += 1

    def _compute_reward(self):
        # Warmup first step: no previous history to compare
        if self.episode_time <= self.sim.dt:
            self.last_reward_components = {'progress': 0.0, 'cohesion': 0.0, 'smoothness': 0.0}
            self.last_objectives = {'progress': 0.0, 'cohesion': 0.0, 'smoothness': 0.0}
            self.reward_components_history.append(self.last_reward_components.copy())
            self.objectives_history.append(self.last_objectives.copy())
            return 0.0

        # 1) Progress of COM along x (normalized by inflow amplitude)
        v_ref = float(self.inflow.amplitude) if getattr(self.inflow, 'amplitude', 0.0) != 0 else 1.0
        dv_members = []
        for member in self.swarm.members:
            dv_members.append(-(member.location['x'] - member.previous_locations[-2]['x']) / self.sim.dt)
        dv_com = float(np.mean(dv_members)) / v_ref
        r_progress = self.w_prog * dv_com

        # 2) Cohesion: average 2D distance to center of mass, normalized
        xs = np.array([m.location['x'] for m in self.swarm.members], dtype=float)
        ys = np.array([m.location['y'] for m in self.swarm.members], dtype=float)
        x_com = float(np.mean(xs))
        y_com = float(np.mean(ys))
        dists = np.sqrt((xs - x_com) ** 2 + (ys - y_com) ** 2)
        avg_dist = float(np.mean(dists))
        # Normalize by half-diagonal of domain to keep within ~[0,1]
        norm_scale = float(np.sqrt((self.sim.length_x / 2) ** 2 + (self.sim.length_y / 2) ** 2))
        avg_dist_norm = avg_dist / norm_scale if norm_scale > 0 else 0.0
        avg_dist_norm = float(np.clip(avg_dist_norm, 0.0, 1.0))
        r_cohesion = -1.0 * self.w_cohesion * avg_dist_norm

        # 3) Smoothness: cosine similarity between actions at t and t-1, averaged over members
        smooth_vals = []
        for member in self.swarm.members:
            if len(member.previous_actions) >= 2:
                a_prev = member.previous_actions[-2]
                a_curr = member.previous_actions[-1]
                v1 = np.array([a_prev['x'], a_prev['y']], dtype=float)
                v2 = np.array([a_curr['x'], a_curr['y']], dtype=float)
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 == 0.0 or n2 == 0.0:
                    cos_sim = 1.0
                else:
                    cos_sim = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                # Map [-1,1] -> [0,1]
                smooth_vals.append((cos_sim + 1.0) / 2.0)
            else:
                smooth_vals.append(1.0)
        smoothness = float(np.mean(smooth_vals))
        r_smooth = self.w_smooth * smoothness

        total_reward = r_progress + r_cohesion + r_smooth
        self.last_reward_components = {
            'progress': float(r_progress),
            'cohesion': float(r_cohesion),
            'smoothness': float(r_smooth)
        }
        # Unweighted objective vector for MOMARL (all to be maximized)
        self.last_objectives = {
            'progress': float(r_progress/self.w_prog),
            'cohesion': float(r_cohesion/self.w_cohesion),
            'smoothness': float(r_smooth/self.w_smooth)
        }
        # Per-member normalized progress (unweighted), for per-member critic head
        self.last_member_progress_norm = [float(dv_i / v_ref) for dv_i in dv_members]
        self.reward_components_history.append(self.last_reward_components.copy())
        self.objectives_history.append(self.last_objectives.copy())
        return float(total_reward)


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
            prog_vals = []
            coh_vals = []
            smooth_vals = []
            for c in comps:
                if isinstance(c, dict):
                    prog_vals.append(c.get('progress', 0.0))
                    coh_vals.append(c.get('cohesion', 0.0))
                    smooth_vals.append(c.get('smoothness', 0.0))
            if len(prog_vals) > 0:
                self.logger.record('custom/r_progress', float(np.mean(prog_vals)))
                self.logger.record('custom/r_cohesion', float(np.mean(coh_vals)))
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
        #                             f'../runs/{folder_attr[i]}/PPO/velocity_{pid_attr[i]}/velocity_{current_time_attr[i]:.3f}')
        #         if p_attr[i] is not None:
        #             phi.field.write(p_attr[i],
        #                             f'../runs/{folder_attr[i]}/PPO/pressure_{pid_attr[i]}/pressure_{current_time_attr[i]:.3f}')
        # plot_save_fields(v_attr[i], p_attr[i], folder_attr[i], pid_attr[i], current_time_attr[i], sim_attr[i])
        return True


class ActorCriticMO(nn.Module):
    """
    Multi-objective actor-critic with shared torso, one critic head per objective,
    and a Gaussian policy over joint actions (all agents).
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: tuple[int, int] = (256, 256), num_members: int | None = None):
        super().__init__()
        self.torso = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden_sizes[1], act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        # Three critic heads: progress, cohesion, smoothness
        self.v_prog = nn.Linear(hidden_sizes[1], 1)
        self.v_coh = nn.Linear(hidden_sizes[1], 1)
        self.v_smooth = nn.Linear(hidden_sizes[1], 1)
        # Optional per-member progress heads
        self.num_members = num_members
        if num_members is not None and num_members > 0:
            self.v_member_prog = nn.Linear(hidden_sizes[1], num_members)

    def forward(self, obs: torch.Tensor):
        x = self.torso(obs)
        mu = self.mu(x)
        std = torch.exp(self.log_std)
        return mu, std

    def values(self, obs: torch.Tensor):
        x = self.torso(obs)
        v_p = self.v_prog(x).squeeze(-1)
        v_c = self.v_coh(x).squeeze(-1)
        v_s = self.v_smooth(x).squeeze(-1)
        if hasattr(self, 'v_member_prog'):
            v_mp = self.v_member_prog(x)
        else:
            v_mp = None
        return v_p, v_c, v_s, v_mp


class RolloutBufferMO:
    def __init__(self, buffer_size: int, obs_dim: int, act_dim: int, device: torch.device, num_members: int | None = None):
        self.buffer_size = buffer_size
        self.obs = torch.zeros((buffer_size, obs_dim), dtype=torch.float64)
        self.actions = torch.zeros((buffer_size, act_dim), dtype=torch.float64)
        self.logprobs = torch.zeros((buffer_size,), dtype=torch.float64)
        self.dones = torch.zeros((buffer_size,), dtype=torch.float64)
        # Per-objective rewards and values
        self.rew_prog = torch.zeros((buffer_size,), dtype=torch.float64)
        self.rew_coh = torch.zeros((buffer_size,), dtype=torch.float64)
        self.rew_smooth = torch.zeros((buffer_size,), dtype=torch.float64)
        self.val_prog = torch.zeros((buffer_size,), dtype=torch.float64)
        self.val_coh = torch.zeros((buffer_size,), dtype=torch.float64)
        self.val_smooth = torch.zeros((buffer_size,), dtype=torch.float64)
        # Per-member progress
        self.num_members = int(num_members) if num_members is not None else 0
        if self.num_members > 0:
            self.rew_member_prog = torch.zeros((buffer_size, self.num_members), dtype=torch.float64)
            self.val_member_prog = torch.zeros((buffer_size, self.num_members), dtype=torch.float64)
            self.adv_member_prog = torch.zeros((buffer_size, self.num_members), dtype=torch.float64)
            self.ret_member_prog = torch.zeros((buffer_size, self.num_members), dtype=torch.float64)
        # Advantages and returns per objective
        self.adv_prog = torch.zeros((buffer_size,), dtype=torch.float64)
        self.adv_coh = torch.zeros((buffer_size,), dtype=torch.float64)
        self.adv_smooth = torch.zeros((buffer_size,), dtype=torch.float64)
        self.ret_prog = torch.zeros((buffer_size,), dtype=torch.float64)
        self.ret_coh = torch.zeros((buffer_size,), dtype=torch.float64)
        self.ret_smooth = torch.zeros((buffer_size,), dtype=torch.float64)
        self.ptr = 0
        self.device = device

    def add(self, obs, action, logprob, done, rew_prog, rew_coh, rew_smooth, val_prog, val_coh, val_smooth, member_prog: np.ndarray | None = None, val_member_prog: np.ndarray | None = None):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.logprobs[i] = logprob
        self.dones[i] = float(done)
        self.rew_prog[i] = rew_prog
        self.rew_coh[i] = rew_coh
        self.rew_smooth[i] = rew_smooth
        self.val_prog[i] = val_prog
        self.val_coh[i] = val_coh
        self.val_smooth[i] = val_smooth
        if self.num_members > 0 and member_prog is not None and val_member_prog is not None:
            mp = torch.as_tensor(member_prog, dtype=torch.float64)
            vm = torch.as_tensor(val_member_prog, dtype=torch.float64)
            if mp.numel() == self.num_members:
                self.rew_member_prog[i] = mp
            if vm.numel() == self.num_members:
                self.val_member_prog[i] = vm
        self.ptr += 1

    def compute_gae(self, last_values: tuple, last_done: float,
                    gamma: float = 0.99, lam: float = 0.95):
        # last_values may include per-member tensor at index 3
        last_v_prog, last_v_coh, last_v_smooth = last_values[:3]
        last_v_members = last_values[3] if len(last_values) > 3 else None
        adv_p, adv_c, adv_s = 0.0, 0.0, 0.0
        adv_members = torch.zeros(self.num_members, dtype=torch.float64)
        for t in reversed(range(self.ptr)):
            next_nonterminal = 1.0 - (self.dones[t+1] if t < self.ptr - 1 else last_done)
            next_v_prog = self.val_prog[t+1] if t < self.ptr - 1 else last_v_prog
            next_v_coh = self.val_coh[t+1] if t < self.ptr - 1 else last_v_coh
            next_v_smooth = self.val_smooth[t+1] if t < self.ptr - 1 else last_v_smooth
            if self.num_members > 0:
                next_v_members = self.val_member_prog[t+1] if t < self.ptr - 1 else (last_v_members if last_v_members is not None else torch.zeros(self.num_members))

            delta_p = self.rew_prog[t] + gamma * next_v_prog * next_nonterminal - self.val_prog[t]
            delta_c = self.rew_coh[t] + gamma * next_v_coh * next_nonterminal - self.val_coh[t]
            delta_s = self.rew_smooth[t] + gamma * next_v_smooth * next_nonterminal - self.val_smooth[t]
            if self.num_members > 0:
                delta_members = self.rew_member_prog[t] + gamma * next_v_members * next_nonterminal - self.val_member_prog[t]

            adv_p = float(delta_p) + gamma * lam * next_nonterminal * adv_p
            adv_c = float(delta_c) + gamma * lam * next_nonterminal * adv_c
            adv_s = float(delta_s) + gamma * lam * next_nonterminal * adv_s
            if self.num_members > 0:
                adv_members = delta_members + gamma * lam * next_nonterminal * adv_members

            self.adv_prog[t] = adv_p
            self.adv_coh[t] = adv_c
            self.adv_smooth[t] = adv_s
            if self.num_members > 0:
                self.adv_member_prog[t] = adv_members

        self.ret_prog = self.adv_prog + self.val_prog
        self.ret_coh = self.adv_coh + self.val_coh
        self.ret_smooth = self.adv_smooth + self.val_smooth
        if self.num_members > 0:
            self.ret_member_prog = self.adv_member_prog + self.val_member_prog
        # Normalize advantages per objective
        def norm(x: torch.Tensor):
            if x.std() > 1e-8:
                return (x - x.mean()) / (x.std() + 1e-8)
            return x - x.mean()
        self.adv_prog = norm(self.adv_prog)
        self.adv_coh = norm(self.adv_coh)
        self.adv_smooth = norm(self.adv_smooth)
        if self.num_members > 0:
            # normalize per member independently
            for m in range(self.num_members):
                self.adv_member_prog[:, m] = norm(self.adv_member_prog[:, m])

    def get(self, batch_size: int):
        idxs = torch.randperm(self.ptr)
        for start in range(0, self.ptr, batch_size):
            end = min(start + batch_size, self.ptr)
            batch_idx = idxs[start:end]
            yield (
                self.obs[batch_idx], self.actions[batch_idx], self.logprobs[batch_idx], self.dones[batch_idx],
                self.rew_prog[batch_idx], self.rew_coh[batch_idx], self.rew_smooth[batch_idx],
                self.val_prog[batch_idx], self.val_coh[batch_idx], self.val_smooth[batch_idx],
                self.adv_prog[batch_idx], self.adv_coh[batch_idx], self.adv_smooth[batch_idx],
                self.ret_prog[batch_idx], self.ret_coh[batch_idx], self.ret_smooth[batch_idx],
                (self.rew_member_prog[batch_idx] if self.num_members > 0 else None),
                (self.val_member_prog[batch_idx] if self.num_members > 0 else None),
                (self.adv_member_prog[batch_idx] if self.num_members > 0 else None),
                (self.ret_member_prog[batch_idx] if self.num_members > 0 else None),
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
                proj = (dot / (gj.norm()**2 + 1e-12)) * gj
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
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    num_steps = 128
    if isinstance(env, VecEnv):
        if os.path.exists(f"../runs/{env.get_attr('folder')[0]}/swarm_rl_ppo.zip"):
            model = PPO.load(f"../runs/{env.get_attr('folder')[0]}/swarm_rl_ppo.zip", env=env)
            print('Successfully loaded model')
        else:
            model = PPO('MlpPolicy', env, verbose=2, n_steps=num_steps, batch_size=32,
                        device=device, gamma=0.95, learning_rate=0.0003, ent_coef=0.01, n_epochs=10,
                        tensorboard_log=f"../runs/{env.get_attr('folder')[0]}/swarm_rl_ppo_tb")
        model.learn(total_timesteps=timesteps * env.num_envs, log_interval=1, progress_bar=True,
                    callback=RewardLoggerCallback(), reset_num_timesteps=False)
        model.save(f"../runs/{env.get_attr('folder')[0]}/swarm_rl_ppo")
        for env_i in range(env.num_envs):
            date_stamp = f'{datetime.now().year}-{datetime.now().month}-{datetime.now().day}_{datetime.now().hour}-{datetime.now().minute}-{datetime.now().second}'
            os.makedirs(f"../runs/{env.get_attr('folder')[env_i]}/PPO/{date_stamp}_{env.get_attr('pid')[env_i]}", exist_ok=True)
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
        if os.path.exists(f"../runs/{env.folder}/swarm_rl_ppo.zip"):
            model = PPO.load(f"../runs/{env.folder}/swarm_rl_ppo.zip", env=env)
            print('Successfully loaded model')
        else:
            model = PPO('MlpPolicy', env, verbose=2, n_steps=num_steps, batch_size=num_steps, device=device, gamma=0.95,
                        tensorboard_log=f"../runs/{env.folder}/swarm_rl_ppo_tb")
        model.learn(total_timesteps=timesteps, log_interval=1, progress_bar=True, callback=RewardLoggerCallback(),
                    reset_num_timesteps=False)
        model.save(f"../runs/{env.folder}/swarm_rl_ppo")
        date_stamp = f'{datetime.now().year}-{datetime.now().month}-{datetime.now().day}_{datetime.now().hour}-{datetime.now().minute}-{datetime.now().second}'
        os.makedirs(f"../runs/{env.folder}/PPO/{date_stamp}", exist_ok=True)
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
    model.save(f'../runs/{env.folder}/swarm_rl_sac')

    plot_save_locations(folder_name=f'{env.folder}/SAC', sim=env.sim, swarm=env.swarm)
    plot_save_velocities(folder_name=f'{env.folder}/SAC', sim=env.sim, swarm=env.swarm)
    plot_save_rewards(folder_name=f'{env.folder}/SAC', rewards=env.rewards, sim=env.sim)


def run_MOMAPPO(env, total_timesteps: int,
                n_steps: int = 1024, batch_size: int = 256, update_epochs: int = 10,
                gamma: float = 0.95, gae_lambda: float = 0.95, clip_coef: float = 0.2,
                ent_coef: float = 0.0, vf_coef: float = 0.5, lr: float = 3e-4,
                device: str = 'cpu', open_tensorboard: bool = True,
                resume: bool = True):
    """
    Multi-objective MAPPO training with PCGrad on the actor across three objectives
    (progress, cohesion, smoothness). Supports multiple parallel environments.
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
    obs_flat = obs_all[0].reshape(-1).astype(np.float64)

    # Resolve folder - use parent folder for shared resources
    if is_vec:
        env_folder = env.get_attr('folder')[0]
        folder_parts = env_folder.split('/')
        folder = '/'.join(folder_parts[:-1]) if len(folder_parts) > 1 else env_folder
    else:
        folder = env.folder
    tb_log_dir = f"../runs/{folder}/MOMAPPO_tb"
    os.makedirs(os.path.dirname(tb_log_dir), exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)

    # Try to open TensorBoard server
    if open_tensorboard:
        try:
            subprocess.Popen([
                "tensorboard", "--logdir", tb_log_dir, "--port", "6006", "--host", "127.0.0.1"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"TensorBoard: http://127.0.0.1:6006 (logdir {tb_log_dir})")
        except Exception as e:
            print(f"Could not launch TensorBoard automatically: {e}. You can run: tensorboard --logdir {tb_log_dir}")
    obs_dim = int(obs_flat.shape[0])
    act_dim = int(env.action_space.shape[0] * env.action_space.shape[1])

    num_members = int(env.action_space.shape[0])
    model = ActorCriticMO(obs_dim, act_dim, num_members=num_members).to(dev).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Checkpointing (resume if requested)
    ckpt_dir = f"../runs/{folder}/MOMAPPO/models"
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

    for update in range(num_updates):
        # Create buffer sized for all environments
        buffer = RolloutBufferMO(n_steps * num_envs, obs_dim, act_dim, dev, num_members=num_members)

        for t in range(n_steps):
            # Sample actions for ALL environments in parallel
            actions_all = []
            logprobs_all = []
            vals_prog_all = []
            vals_coh_all = []
            vals_smooth_all = []
            vals_member_all = []
            obs_tensors = []
            
            for env_idx in range(num_envs):
                obs = obs_all[env_idx]
                obs_t = torch.tensor(obs.reshape(-1), dtype=torch.float64, device=dev)
                obs_tensors.append(obs_t)
                
                with torch.no_grad():
                    mu, std = model(obs_t.unsqueeze(0))
                    dist = Normal(mu, std)
                    action = dist.sample()[0]
                    logprob = dist.log_prob(action).sum()
                    value_tuple = model.values(obs_t.unsqueeze(0))
                    val_prog, val_coh, val_smooth, val_member_prog = value_tuple
                
                actions_all.append(action.cpu().numpy())
                logprobs_all.append(float(logprob.cpu()))
                vals_prog_all.append(float(val_prog[0].cpu()))
                vals_coh_all.append(float(val_coh[0].cpu()))
                vals_smooth_all.append(float(val_smooth[0].cpu()))
                vals_member_all.append(val_member_prog[0].cpu().numpy() if val_member_prog is not None else None)
            
            # Format actions for VecEnv (shape: (num_envs, num_members, 2))
            actions_np = np.array(actions_all)  # (num_envs, act_dim)
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
                obs_t = obs_tensors[env_idx]
                action_t = torch.tensor(actions_all[env_idx], dtype=torch.float64)
                
                info = infos_batch[env_idx]
                obj = info.get('objectives', {'progress': 0.0, 'cohesion': 0.0, 'smoothness': 0.0})
                rew_prog = float(obj.get('progress', 0.0))
                rew_coh = float(obj.get('cohesion', 0.0))
                rew_smooth = float(obj.get('smoothness', 0.0))
                member_prog = info.get('member_progress', None)
                
                buffer.add(
                    obs_t.cpu(), action_t, logprobs_all[env_idx],
                    bool(dones_batch[env_idx]),
                    rew_prog, rew_coh, rew_smooth,
                    vals_prog_all[env_idx], vals_coh_all[env_idx], vals_smooth_all[env_idx],
                    member_prog, vals_member_all[env_idx]
                )
                step_count += 1
            
            # Update observations for next iteration
            obs_all = next_obs_all
            last_dones = dones_batch.astype(np.float32)
            
            if pbar is not None:
                pbar.update(num_envs)

        # Bootstrap last values (use first env's observation as representative)
        last_obs_t = torch.tensor(obs_all[0].reshape(-1), dtype=torch.float64, device=dev)
        with torch.no_grad():
            last_vals = model.values(last_obs_t.unsqueeze(0))
            lvp = last_vals[0][0].cpu()
            lvc = last_vals[1][0].cpu()
            lvs = last_vals[2][0].cpu()
            lvm = (last_vals[3][0].cpu() if last_vals[3] is not None else None)
            last_vals = (lvp, lvc, lvs, lvm)
        buffer.compute_gae(last_vals, float(last_dones[0]), gamma=gamma, lam=gae_lambda)

        # Log per-objective means from this rollout
        with torch.no_grad():
            rp_mean = float(buffer.rew_prog[:buffer.ptr].mean()) if buffer.ptr > 0 else 0.0
            rc_mean = float(buffer.rew_coh[:buffer.ptr].mean()) if buffer.ptr > 0 else 0.0
            rs_mean = float(buffer.rew_smooth[:buffer.ptr].mean()) if buffer.ptr > 0 else 0.0
        writer.add_scalar('objectives/progress', rp_mean, step_count)
        writer.add_scalar('objectives/cohesion', rc_mean, step_count)
        writer.add_scalar('objectives/smoothness', rs_mean, step_count)

        # PPO updates
        epoch_entropies = []
        epoch_value_losses = []
        for epoch in range(update_epochs):
            for batch in buffer.get(batch_size):
                (b_obs, b_actions, b_logp_old, _b_dones,
                 _rp, _rc, _rs, b_vp, b_vc, b_vs,
                 b_advp, b_advc, b_advs, b_retp, b_retc, b_rets,
                 b_rmp, b_vmp, b_amp, b_rtmp) = batch

                b_obs = b_obs.to(dev)
                b_actions = b_actions.to(dev)
                b_logp_old = b_logp_old.to(dev)
                b_advp = b_advp.to(dev)
                b_advc = b_advc.to(dev)
                b_advs = b_advs.to(dev)
                b_retp = b_retp.to(dev)
                b_retc = b_retc.to(dev)
                b_rets = b_rets.to(dev)

                mu, std = model(b_obs)
                dist = Normal(mu, std)
                logp = dist.log_prob(b_actions).sum(-1)
                ratio = torch.exp(logp - b_logp_old)

                def ppo_obj(adv):
                    unclipped = ratio * adv
                    clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv
                    return -torch.mean(torch.min(unclipped, clipped))

                # Per-objective policy losses
                loss_pi_prog = ppo_obj(b_advp)
                loss_pi_coh = ppo_obj(b_advc)
                loss_pi_smooth = ppo_obj(b_advs)

                # Entropy (encourage exploration)
                entropy = dist.entropy().sum(-1).mean()

                # Critic loss (sum across objectives)
                v_prog, v_coh, v_smooth, v_member_prog = model.values(b_obs)
                loss_v = 0.5 * (
                    torch.mean((v_prog - b_retp) ** 2) +
                    torch.mean((v_coh - b_retc) ** 2) +
                    torch.mean((v_smooth - b_rets) ** 2)
                )
                # Add per-member critic loss if enabled
                if v_member_prog is not None and b_rtmp is not None:
                    loss_v = loss_v + 0.5 * torch.mean((v_member_prog - b_rtmp) ** 2)

                # Combine actor gradients via PCGrad, then add critic and entropy
                optimizer.zero_grad(set_to_none=True)
                pcgrad_merge(model, [loss_pi_prog, loss_pi_coh, loss_pi_smooth])
                # Add critic and entropy losses on top of actor grads
                total_aux = vf_coef * loss_v - ent_coef * entropy
                total_aux.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                epoch_entropies.append(float(entropy.detach()))
                epoch_value_losses.append(float(loss_v.detach()))

        # Log training stats per update
        if len(epoch_entropies) > 0:
            writer.add_scalar('loss/value', float(np.mean(epoch_value_losses)), step_count)
            writer.add_scalar('stats/entropy', float(np.mean(epoch_entropies)), step_count)

        # Update progress bar with mean rewards from this rollout
        if pbar is not None:
            pbar.set_postfix({
                'steps': step_count,
                'prog': f"{rp_mean:.3f}",
                'coh': f"{rc_mean:.3f}",
                'sm': f"{rs_mean:.3f}"
            })
        elif tqdm is None:
            print(f"MOMAPPO update {update+1}/{num_updates}, steps so far {step_count}, prog {rp_mean:.3f}, coh {rc_mean:.3f}, sm {rs_mean:.3f}")

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
                
                run_dir = f"../runs/{env_folder}/MOMAPPO/{date_stamp}"
                os.makedirs(run_dir, exist_ok=True)
                
                plot_save_locations(folder_name=f"{env_folder}/MOMAPPO/{date_stamp}", sim=sim_attr, swarm=swarm_attr)
                plot_save_velocities(folder_name=f"{env_folder}/MOMAPPO/{date_stamp}", sim=sim_attr, swarm=swarm_attr)
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

        run_dir = f"../runs/{folder}/MOMAPPO/{date_stamp}"
        os.makedirs(run_dir, exist_ok=True)
        plot_save_locations(folder_name=f"{folder}/MOMAPPO/{date_stamp}", sim=sim_attr, swarm=swarm_attr)
        plot_save_velocities(folder_name=f"{folder}/MOMAPPO/{date_stamp}", sim=sim_attr, swarm=swarm_attr)
        plot_save_rewards(folder_name=f"{folder}/MOMAPPO/{date_stamp}", rewards=rewards_attr, sim=sim_attr)
        plot_save_rewards_objectives(folder_name=f"{folder}/MOMAPPO/{date_stamp}", sim=sim_attr, objective_history=objectives_attr)

    # Save checkpoints (timestamped and latest)
    ts_ckpt = f"{ckpt_dir}/model_{date_stamp}.pt"
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'obs_dim': obs_dim,
        'act_dim': act_dim,
        'num_members': num_members,
        'total_timesteps': total_timesteps,
    }, ts_ckpt)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'obs_dim': obs_dim,
        'act_dim': act_dim,
        'num_members': num_members,
        'total_timesteps': total_timesteps,
    }, latest_ckpt)
    print(f"Saved MOMAPPO checkpoints to {ts_ckpt} and {latest_ckpt}")
    
    # Close progress bar if it exists
    if pbar is not None:
        pbar.close()
    
    writer.close()
