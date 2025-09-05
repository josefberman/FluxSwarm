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
from plotting import plot_save_locations, plot_save_velocities, plot_save_rewards, plot_save_actions, plot_save_fields
from stable_baselines3 import PPO, SAC
import torch
from datetime import datetime


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

    def __init__(self, sim: Simulation, swarm: Swarm, fluid: Fluid, inflow: Inflow, folder: str):
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
        box = Box['x,y', 0:sim.length_x, 0:sim.length_y]
        boundary = {'x': ZERO_GRADIENT, 'y': 0}
        self.v = StaggeredGrid(0, boundary=boundary, bounds=box, x=sim.resolution[0], y=sim.resolution[1])
        self.p = None

        # Define observation space: num_of_members * (position x2, velocity x2, pressure x4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(swarm.members), 8), dtype=np.float32
        )
        # Define action space (force control x2 for each agent)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(swarm.members), 2), dtype=np.float32
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
                           member_max_force=prev_swarm.member_max_force)  # density in mg/mm^3, force in mg*mm/s^2
        box = Box['x,y', 0:self.sim.length_x, 0:self.sim.length_y]
        boundary = {'x': ZERO_GRADIENT, 'y': 0}
        self.v = StaggeredGrid(0, boundary=boundary, bounds=box, x=self.sim.resolution[0], y=self.sim.resolution[1])
        self.p = None
        for i, member in enumerate(self.swarm.members):
            member.previous_locations = prev_members[i].previous_locations.copy()
            member.previous_velocities = prev_members[i].previous_velocities.copy()
            member.previous_actions = prev_members[i].previous_actions.copy()
        self.episode_time = 0.0
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

        # Compute rewards
        reward = self._compute_reward()
        self.rewards.append(reward)
        done = self._compute_done()

        # if self.current_timestep % 10 == 0:
        #     write(self.v, f'../runs/{self.folder}/velocity/velocity_{self.current_timestep}')
        #     write(self.p, f'../runs/{self.folder}/pressure/pressure_{self.current_timestep}')

        return self._get_observation(), reward, done, False, {}

    def _get_observation(self):
        """
        Generates and returns the observation array by sampling environmental
        fields and extracting information from members of the swarm.

        The observation array contains the current `x` and `y` positions, `x`
        and `y` components of velocity, and sampled pressure profiles for
        each swarm member. If no pressure field is provided (`self.p` is None),
        the pressure profile values default to an array of zeros.

        :returns: A NumPy array of shape (number_of_members, observation_features)
                  with `float32` as data type, where each row represents the
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
        return np.array(obs, dtype=np.float32)

    def _compute_done(self):
        """
        Computes whether a task is considered complete based on swarm members'
        locations and the state of the object.

        This function determines the completion status (`done`) based on certain
        conditions. If the attribute `v` is `None`, the task is immediately marked
        as done. Otherwise, the function iterates through the `members` of the
        `swarm` and evaluates their x-coordinate. If any member's x-coordinate
        falls outside the range of 200 to 550 (inclusive boundaries not considered),
        the task is marked as done. The function returns the computed `done` status.

        :return: A boolean indicating whether the task is done.
        :rtype: bool
        """
        done = False
        if self.v is None:
            done = True
        else:
            count_members_finished = 0
            for member in self.swarm.members:
                if member.location['x'] <= 20:
                    count_members_finished += 1
                if member.location['x'] >= 80:
                    done = True
            if count_members_finished == len(self.swarm.members):
                done = True
        return done

    def _compute_reward(self):
        """
        Computes the reward based on the movement of members in the swarm.

        This method evaluates the movement of each member in the swarm. It assigns a reward
        based on whether their current x-coordinate has increased or decreased compared to
        their previous x-coordinate. In case the value of the v attribute is None, a negative reward
        is given.

        :return: The calculated reward based on the members' movements.
        :rtype: float
        """
        # If episode time is 0 or first timestep (dt), return 0 to avoid referencing to previous episode
        if (self.episode_time == 0) or (self.episode_time == self.sim.dt):
            return 0
        
        # If velocity field is None (calculation did not converge), return 0
        if self.v is None:
            return 0

        # Calculate reward
        reward = 0
        distance_factor = 1.0  # Distance reward at unit magnitude
        direction_factor = 1.0  # Direction-change penalty at unit magnitude
        discount_factor = 0.99  # Discount factor of PPO

        for member in self.swarm.members:
            # Distance reward: how far left (decrease in x) from initial x
            prev_location_x = member.previous_locations[-2]['x']  # Assuming 0 is the first location
            cur_location_x = member.location['x']
            distance_delta = prev_location_x - discount_factor * cur_location_x
            # distance_delta = max(0, prev_location_x - discount_factor * cur_location_x)  # Only reward for moving left

            # Penalty for rapid direction change in action
            if hasattr(member, 'previous_actions') and len(member.previous_actions) >= 2:
                prev_action = member.previous_actions[-2]
                curr_action = member.previous_actions[-1]
                prev_vec = np.array([prev_action['x'], prev_action['y']])
                curr_vec = np.array([curr_action['x'], curr_action['y']])
                norm_prev = np.linalg.norm(prev_vec)
                norm_curr = np.linalg.norm(curr_vec)
                if norm_prev > 0 and norm_curr > 0:
                    # Cosine similarity in [-1, 1]; 1 means aligned, -1 means opposite
                    cosine_sim = np.clip(np.dot(prev_vec, curr_vec) / (norm_prev * norm_curr), -1.0, 1.0)
                else:
                    cosine_sim = 1.0  # If one vector is zero, treat as aligned (no change)
            else:
                cosine_sim = 1.0

            # Convert similarity to change penalty in [0, 1]; 0 when aligned, 1 when opposite
            direction_penalty = (1.0 - cosine_sim) / 2.0

            member_reward = distance_delta * distance_factor - direction_penalty * direction_factor
            reward += member_reward
        # for member in self.swarm.members:
            # if member.location['y'] >= 1.8 and member.location['y'] <= 2.2:
            #     reward += 1
                
        return reward

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
    num_steps = 10
    if isinstance(env, VecEnv):
        if os.path.exists(f'../runs/{env.get_attr('folder')[0]}/swarm_rl_ppo'):
            model = PPO.load(f'../runs/{env.get_attr('folder')[0]}/swarm_rl_ppo')
        else:
            model = PPO('MlpPolicy', env, verbose=2, n_steps=num_steps, batch_size=(num_steps * env.num_envs),
                        device=device, gamma=0.95,
                        tensorboard_log=f'../runs/{env.get_attr('folder')[0]}/swarm_rl_ppo_tb')
        model.learn(total_timesteps=timesteps * env.num_envs, log_interval=1, progress_bar=True,
                    callback=RewardLoggerCallback(), reset_num_timesteps=False)
        model.save(f'../runs/{env.get_attr('folder')[0]}/swarm_rl_ppo')
        for env_i in range(env.num_envs):
            date_stamp = f'{datetime.now().year}-{datetime.now().month}-{datetime.now().day}_{datetime.now().hour}-{datetime.now().minute}-{datetime.now().second}'
            os.makedirs(f'../runs/{env.get_attr('folder')[env_i]}/PPO/{date_stamp}_{env.get_attr('pid')[env_i]}', exist_ok=True)
            plot_save_locations(folder_name=f'{env.get_attr('folder')[env_i]}/PPO/{date_stamp}_{env.get_attr('pid')[env_i]}',
                                sim=env.get_attr('sim')[env_i], swarm=env.get_attr('swarm')[env_i])
            plot_save_velocities(folder_name=f'{env.get_attr('folder')[env_i]}/PPO/{date_stamp}_{env.get_attr('pid')[env_i]}',
                                 sim=env.get_attr('sim')[env_i], swarm=env.get_attr('swarm')[env_i])
            plot_save_rewards(folder_name=f'{env.get_attr('folder')[env_i]}/PPO/{date_stamp}_{env.get_attr('pid')[env_i]}',
                              rewards=env.get_attr('rewards')[env_i], sim=env.get_attr('sim')[env_i])
            plot_save_actions(folder_name=f'{env.get_attr('folder')[env_i]}/PPO/{date_stamp}_{env.get_attr('pid')[env_i]}',
                              sim=env.get_attr('sim')[env_i], swarm=env.get_attr('swarm')[env_i])
            # plot_save_fields(folder_name=f'{env.get_attr('folder')[env_i]}/PPO/', pid=env.get_attr('pid')[env_i])
    elif isinstance(env, SwarmEnv):
        if os.path.exists(f'../runs/{env.folder}/swarm_rl_ppo'):
            model = PPO.load(f'../runs/{env.folder}/swarm_rl_ppo')
        else:
            model = PPO('MlpPolicy', env, verbose=2, n_steps=num_steps, batch_size=num_steps, device=device, gamma=0.95,
                        tensorboard_log=f'../runs/{env.folder}/swarm_rl_ppo_tb')
        model.learn(total_timesteps=timesteps, log_interval=1, progress_bar=True, callback=RewardLoggerCallback(),
                    reset_num_timesteps=False)
        model.save(f'../runs/{env.folder}/swarm_rl_ppo')
        date_stamp = f'{datetime.now().year}-{datetime.now().month}-{datetime.now().day}_{datetime.now().hour}-{datetime.now().minute}-{datetime.now().second}'
        os.makedirs(f'../runs/{env.folder}/PPO/{date_stamp}', exist_ok=True)
        plot_save_locations(folder_name=f'{env.folder}/PPO/{date_stamp}', sim=env.sim, swarm=env.swarm)
        plot_save_velocities(folder_name=f'{env.folder}/PPO/{date_stamp}', sim=env.sim, swarm=env.swarm)
        plot_save_rewards(folder_name=f'{env.folder}/PPO/{date_stamp}', rewards=env.rewards, sim=env.sim)
        plot_save_actions(folder_name=f'{env.folder}/PPO/{date_stamp}', sim=env.sim, swarm=env.swarm)

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
