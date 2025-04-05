import gymnasium as gym
from phi.flow import *
import phi.field as field
from gymnasium import spaces
from data_structures import Simulation, Swarm, Fluid, Inflow
from plotting import plot_save_current_step, plot_save_locations, plot_save_velocities, plot_save_rewards
from simulation import step, sample_field_around_obstacle
from stable_baselines3 import PPO, SAC


class SwarmEnv(gym.Env):
    """
    Custom Multi-Agent RL environment for swarm movement optimization.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, sim: Simulation, swarm: Swarm, fluid: Fluid, inflow: Inflow, folder: str):
        super(SwarmEnv, self).__init__()
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

        # Define observation space (position x2, velocity x2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(swarm.members), 12), dtype=np.float32
        )

        # Define action space (velocity control x2 and force control for each agent)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(swarm.members), 2), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        # self.current_time = 0
        prev_members = self.swarm.members
        self.swarm = Swarm(num_x=3, num_y=3, left_location=480, bottom_location=8.1, member_interval_x=6.3,
                           member_interval_y=6.3, member_radius=1.8, member_density=5.150, member_max_force=100)
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
        print(f'Time: {self.current_time:.3f}')
        if self.current_timestep % 10 == 0:
            for i, member in enumerate(self.swarm.members):
                acc_x = action[i, 0] * member.max_force / member.mass
                acc_y = action[i, 1] * member.max_force / member.mass
                member.velocity['x'] += acc_x * self.sim.dt
                member.velocity['y'] += acc_y * self.sim.dt

        # Advance the simulation
        self.v, self.p, self.swarm = step(
            v=self.v, p=self.p, inflow=self.inflow, sim=self.sim, swarm=self.swarm, fluid_obj=self.fluid,
            t=self.episode_time
        )

        self.current_time += self.sim.dt
        self.episode_time += self.sim.dt
        self.current_timestep += 1

        # if self.v is not None:
        #     if self.current_timestep % 5 == 0:
        #         plot_save_current_step(current_time=self.current_time, folder_name=self.folder, v_field=self.v,
        #                                p_field=self.p, sim=self.sim, swarm=self.swarm)
        #         phi.field.write(self.v, f'../runs/run_{self.folder}/velocity/velocity_{self.current_time:.3f}')
        #         phi.field.write(self.p, f'../runs/run_{self.folder}/pressure/pressure_{self.current_time:.3f}')

        # Compute rewards
        reward = self._compute_reward()
        self.rewards.append(reward)
        # done = self.current_time >= self.sim.total_time
        done = self._compute_done()
        # print(f'reward: {reward}, done: {done}.')

        return self._get_observation(), reward, done, False, {}

    def _get_observation(self):
        """Retrieve the current state of the swarm."""
        obs = []
        for member in self.swarm.members:
            if self.p is not None:
                pressure_profile = sample_field_around_obstacle(f=self.p, member=member, sim=self.sim)
            else:
                pressure_profile = np.zeros(8, dtype=object)
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
                if member.location['x'] < member.previous_locations[-2]['x']:
                    reward += 1
                else:
                    reward -= 1
                if member.location['x'] <= 200:
                    reward = 100
                if member.location['x'] >= 550:
                    reward = -100
        return reward

    def render(self, mode='human'):
        pass


def run_PPO(env: SwarmEnv):
    model = PPO('MlpPolicy', env, verbose=2, n_steps=64, device='cpu', gamma=0.95)
    model.learn(total_timesteps=env.sim.time_steps)
    model.save(f'../runs/run_{env.folder}/swarm_rl_ppo')

    plot_save_locations(folder_name=f'{env.folder}/PPO', sim=env.sim, swarm=env.swarm)
    plot_save_velocities(folder_name=f'{env.folder}/PPO', sim=env.sim, swarm=env.swarm)
    plot_save_rewards(folder_name=f'{env.folder}/PPO', rewards=env.rewards, sim=env.sim)


def run_SAC(env: SwarmEnv):
    model = SAC('MlpPolicy', env, verbose=2, device='cpu', gamma=0.95, tau=0.1)
    model.learn(total_timesteps=env.sim.time_steps)
    model.save(f'../runs/run_{env.folder}/swarm_rl_sac')

    plot_save_locations(folder_name=f'{env.folder}/SAC', sim=env.sim, swarm=env.swarm)
    plot_save_velocities(folder_name=f'{env.folder}/SAC', sim=env.sim, swarm=env.swarm)
    plot_save_rewards(folder_name=f'{env.folder}/SAC', rewards=env.rewards, sim=env.sim)
