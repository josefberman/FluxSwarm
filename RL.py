import os
from gymnasium import spaces
import gymnasium as gym
from phi.flow import *
import phi.field as field
from data_structures import Simulation, Swarm, Fluid, Inflow
from simulation import step, sample_field_around_obstacle
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from plotting import plot_save_current_step, plot_save_locations, plot_save_velocities, plot_save_rewards, \
    plot_save_fields
from stable_baselines3 import PPO, SAC


class SwarmEnv(gym.Env):
    """
    Custom Multi-Agent RL environment for swarm movement optimization.
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
        os.makedirs(f'../runs/run_{self.folder}/PPO/velocity_{self.pid}', exist_ok=True)
        os.makedirs(f'../runs/run_{self.folder}/PPO/pressure_{self.pid}', exist_ok=True)

    def reset(self, seed=None, options=None):
        # self.current_time = 0
        prev_members = self.swarm.members
        self.swarm = Swarm(num_x=3, num_y=3, left_location=480, bottom_location=8.1, member_interval_x=6.3,
                           member_interval_y=6.3, member_radius=1.8, member_density=5.150, member_max_force=0.017)
        box = Box['x,y', 0:self.sim.length_x, 0:self.sim.length_y]
        boundary = {'x': ZERO_GRADIENT, 'y': 0}
        self.v = StaggeredGrid(0, boundary=boundary, bounds=box, x=self.sim.resolution[0], y=self.sim.resolution[1])
        self.p = None
        for i, member in enumerate(self.swarm.members):
            member.previous_locations = prev_members[i].previous_locations.copy()
            member.previous_velocities = prev_members[i].previous_velocities.copy()
            member.previous_forces = prev_members[i].previous_forces.copy()
        # reynolds = self.inflow.amplitude * self.sim.length_y / self.fluid.viscosity
        # print(f'{reynolds=}')
        self.episode_time = 0.0
        return self._get_observation(), {}

    def step(self, action):
        """Apply actions to swarm members and update the simulation."""
        # Advance the simulation
        v_temp, p_temp, swarm_temp = step(
            v=self.v, p=self.p, inflow=self.inflow, sim=self.sim, swarm=self.swarm, fluid_obj=self.fluid,
            t=self.episode_time, force_actions=action
        )
        self.v = v_temp
        self.p = p_temp
        self.swarm = swarm_temp
        # self.v, self.p, self.swarm = step(
        #     v=self.v, p=self.p, inflow=self.inflow, sim=self.sim, swarm=self.swarm, fluid_obj=self.fluid,
        #     t=self.episode_time, force_actions=action
        # )

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
        # done = self.current_time >= self.sim.total_time
        done = self._compute_done()

        return self._get_observation(), reward, done, False, {}

    def _get_observation(self):
        """Retrieve the current state of the swarm."""
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
        done = False
        if self.v is None:
            done = True
        else:
            for member in self.swarm.members:
                if (member.location['x'] <= 200) or (member.location['x'] >= 550):
                    done = True
        return done

    def _compute_reward(self):
        """Reward agents for traveling upstream."""
        reward = 0
        if self.v is None:
            reward = -100
        else:
            for i, member in enumerate(self.swarm.members):
                if member.location['x'] < member.previous_locations[0]['x']:
                    reward += member.location['x'] - member.previous_locations[0]['x']
                else:
                    reward += 10 * (member.location['x'] - member.previous_locations[0]['x'])
        return reward

    def render(self, mode='human'):
        pass


class RewardLoggerCallback(BaseCallback):
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
        #                             f'../runs/run_{folder_attr[i]}/PPO/velocity_{pid_attr[i]}/velocity_{current_time_attr[i]:.3f}')
        #         if p_attr[i] is not None:
        #             phi.field.write(p_attr[i],
        #                             f'../runs/run_{folder_attr[i]}/PPO/pressure_{pid_attr[i]}/pressure_{current_time_attr[i]:.3f}')
        # plot_save_fields(v_attr[i], p_attr[i], folder_attr[i], pid_attr[i], current_time_attr[i], sim_attr[i])
        return True


def run_PPO(env: SwarmEnv | VecEnv, timesteps: int):
    num_steps = 10
    if isinstance(env, VecEnv):
        model = PPO('MlpPolicy', env, verbose=2, n_steps=num_steps, batch_size=(num_steps * env.num_envs) // 4,
                    device='cpu', gamma=0.95,
                    tensorboard_log=f'../runs/run_{env.get_attr('folder')[0]}/swarm_rl_ppo_tb')
        model.learn(total_timesteps=timesteps * env.num_envs, log_interval=1, progress_bar=True,
                    callback=RewardLoggerCallback(), reset_num_timesteps=False)
        model.save(f'../runs/run_{env.get_attr('folder')[0]}/swarm_rl_ppo')
        for env_i in range(env.num_envs):
            # model.learn(total_timesteps=timesteps, log_interval=timesteps//(num_steps*env.num_envs))
            os.makedirs(f'../runs/run_{env.get_attr('folder')[env_i]}/PPO/{env.get_attr('pid')[env_i]}', exist_ok=True)
            plot_save_locations(folder_name=f'{env.get_attr('folder')[env_i]}/PPO/{env.get_attr('pid')[env_i]}',
                                sim=env.get_attr('sim')[env_i], swarm=env.get_attr('swarm')[env_i])
            plot_save_velocities(folder_name=f'{env.get_attr('folder')[env_i]}/PPO/{env.get_attr('pid')[env_i]}',
                                 sim=env.get_attr('sim')[env_i], swarm=env.get_attr('swarm')[env_i])
            plot_save_rewards(folder_name=f'{env.get_attr('folder')[env_i]}/PPO/{env.get_attr('pid')[env_i]}',
                              rewards=env.get_attr('rewards')[env_i], sim=env.get_attr('sim')[env_i])
            # plot_save_fields(folder_name=f'{env.get_attr('folder')[env_i]}/PPO/', pid=env.get_attr('pid')[env_i])
    elif isinstance(env, SwarmEnv):
        model = PPO('MlpPolicy', env, verbose=2, n_steps=num_steps, batch_size=num_steps // 4, device='cpu', gamma=0.95,
                    tensorboard_log=f'../runs/run_{env.folder}/swarm_rl_ppo_tb')
        model.learn(total_timesteps=timesteps, log_interval=1, progress_bar=True, callback=RewardLoggerCallback(),
                    reset_num_timesteps=False)
        model.save(f'../runs/run_{env.folder}/swarm_rl_ppo')
        plot_save_locations(folder_name=f'{env.folder}/PPO', sim=env.sim, swarm=env.swarm)
        plot_save_velocities(folder_name=f'{env.folder}/PPO', sim=env.sim, swarm=env.swarm)
        plot_save_rewards(folder_name=f'{env.folder}/PPO', rewards=env.rewards, sim=env.sim)


def run_SAC(env: SwarmEnv):
    model = SAC('MlpPolicy', env, verbose=2, device='cpu', gamma=0.95, tau=0.1)
    model.learn(total_timesteps=env.sim.time_steps, progress_bar=True)
    model.save(f'../runs/run_{env.folder}/swarm_rl_sac')

    plot_save_locations(folder_name=f'{env.folder}/SAC', sim=env.sim, swarm=env.swarm)
    plot_save_velocities(folder_name=f'{env.folder}/SAC', sim=env.sim, swarm=env.swarm)
    plot_save_rewards(folder_name=f'{env.folder}/SAC', rewards=env.rewards, sim=env.sim)
