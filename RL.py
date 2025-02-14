import numpy as np
import gymnasium as gym
from phi.flow import *
import phi.field as field
from gymnasium import spaces
from data_structures import Simulation, Swarm, Fluid, Inflow, Member
from plotting import plot_save_current_step
from simulation import step


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
        self.current_time = 0
        self.folder = folder
        box = Box['x,y', 0:sim.length_x, 0:sim.length_y]
        boundary = {'x': ZERO_GRADIENT, 'y': 0}
        self.v = StaggeredGrid(0, boundary=boundary, bounds=box, x=sim.resolution[0], y=sim.resolution[1])
        self.p = None

        # Define observation space (position x2, velocity x2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(swarm.members), 4), dtype=np.float32
        )

        # Define action space (velocity control x2 and force control for each agent)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(swarm.members), 2), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.current_time = 0
        for member in self.swarm.members:
            member.velocity = {'x': 0, 'y': 0}
        reynolds = self.inflow.amplitude * self.sim.length_y / self.fluid.viscosity
        print(f'{reynolds=}')
        return self._get_observation(), {}

    def step(self, action):
        """Apply actions to swarm members and update the simulation."""
        print(f'Time: {self.current_time:.3f}')
        for i, member in enumerate(self.swarm.members):
            acc_x = action[i, 0] * member.max_force / member.mass
            acc_y = action[i, 1] * member.max_force / member.mass
            member.velocity['x'] += acc_x * self.sim.dt
            member.velocity['y'] += acc_y * self.sim.dt

        # Advance the simulation
        self.v, self.p, self.swarm = step(
            v=self.v, p=self.p, inflow=self.inflow, sim=self.sim, swarm=self.swarm, fluid_obj=self.fluid,
            t=self.current_time
        )
        plot_save_current_step(current_time=self.current_time, folder_name=self.folder, v_field=self.v,
                               p_field=self.p, sim=self.sim, swarm=self.swarm)
        phi.field.write(self.v, f'../runs/run_{self.folder}/velocity/velocity_{self.current_time:.3f}')
        phi.field.write(self.p, f'../runs/run_{self.folder}/pressure/pressure_{self.current_time:.3f}')

        self.current_time += self.sim.dt

        # Compute rewards
        reward = self._compute_reward()
        # done = self.current_time >= self.sim.total_time
        done = self._compute_done()

        return self._get_observation(), reward, done, False, {}

    def _get_observation(self):
        """Retrieve the current state of the swarm."""
        obs = []
        for member in self.swarm.members:
            obs.append([
                member.location['x'], member.location['y'],
                member.velocity['x'], member.velocity['y']
            ])
        return np.array(obs, dtype=np.float32)

    def _compute_done(self):
        done = False
        for member in self.swarm.members:
            if member.location['x'] <= 200:
                done = True
        return done

    def _compute_reward(self):
        """Reward agents for traveling upstream."""
        reward = 0
        for i, member in enumerate(self.swarm.members):
            if member.location['x'] > member.previous_locations[-1]['x']:
                reward += 1
            else:
                reward -= 10
        return reward

    def render(self, mode='human'):
        pass
