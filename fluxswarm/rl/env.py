"""Reinforcement learning environment using new optimized structures."""

import os
import csv
from typing import Dict, Tuple, Optional, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from phi.flow import StaggeredGrid, Box, ZERO_GRADIENT

from fluxswarm.core.simulation import SimulationParams, SimulationEngine
from fluxswarm.core.swarm import Swarm
from fluxswarm.core.fluid import FluidParams, InflowParams
from fluxswarm.core.sim_step import SimulationStep
from fluxswarm.core.physics import PhysicsEngine


class SwarmEnv(gym.Env):
    """
    SwarmEnv class for simulating robotic swarm behavior in fluid flow environments.
    
    Uses the new optimized structures and simulation engine.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        sim_params: SimulationParams,
        swarm: Swarm,
        fluid_params: FluidParams,
        inflow_params: InflowParams,
        folder: str,
        save_fields: bool = False,
        episode_duration: float = 10.0
    ):
        super().__init__()
        
        self.sim_params = sim_params
        self.swarm = swarm
        self.fluid_params = fluid_params
        self.inflow_params = inflow_params
        self.folder = folder
        self.save_fields = save_fields
        self.episode_duration = episode_duration
        
        # Multi-objective reward weights
        self.w_prog = 1.0
        self.w_energy = 1.0
        self.w_smooth = 1.0
        
        # Tracking
        self.last_reward_components = {'progress': 0.0, 'energy_efficiency': 0.0, 'smoothness': 0.0}
        self.last_objectives = {'progress': 0.0, 'energy_efficiency': 0.0, 'smoothness': 0.0}
        self.reward_components_history = []
        self.objectives_history = []
        
        # Initialize simulation
        self.sim_engine = SimulationEngine(sim_params)
        self.sim_step = SimulationStep(sim_params, fluid_params, inflow_params)
        self.physics = PhysicsEngine(sim_params, fluid_params, inflow_params)
        
        # Create initial velocity field
        self.v = self.sim_engine.create_velocity_field()
        self.p = None
        
        # Episode tracking
        self.current_time = 0.0
        self.episode_time = 0.0
        self.current_timestep = 0
        self.episode_index = 0
        self.episode_cum_reward = 0.0
        self.episode_cum_objectives = {'progress': 0.0, 'energy_efficiency': 0.0, 'smoothness': 0.0}
        self.episodes_csv_path = f"run/{self.folder}/episodes_summary.csv"
        # run_dir will be set by run_MOMAPPO after initialization for incremental CSV saving
        self.run_dir = None
        self.rewards = []
        self.locations_csv_initialized = False
        self.velocities_csv_initialized = False
        
        # Field saving
        self.fields_vx_list = []
        self.fields_vy_list = []
        self.fields_p_list = []
        self.field_timesteps = []
    
        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(swarm.members), 8),
            dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(swarm.members), 2),
            dtype=np.float64
        )
        

    def _get_csv_paths(self):
        """
        Return the paths to the CSV files for storing locations and velocities of swarm members.
        If 'run_dir' is set (used for unique experiment/episode directories), use that,
        otherwise use the folder name assigned for this run.
        Returns:
            Tuple[str, str]: Paths for locations.csv and velocities.csv.
        """
        base_path = self.run_dir if self.run_dir else self.folder
        return (
            f"run/{base_path}/locations.csv",
            f"run/{base_path}/velocities.csv"
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset swarm
        self.swarm.reset_all()
        
        # Reset fields
        self.v = self.sim_engine.create_velocity_field()
        self.p = None
        
        # Reset tracking
        self.episode_time = 0.0
        self.current_timestep = 0
        self.episode_cum_reward = 0.0
        self.episode_cum_objectives = {'progress': 0.0, 'energy_efficiency': 0.0, 'smoothness': 0.0}
        
        # Initialize CSV files with headers if needed
        locations_csv_path, velocities_csv_path = self._get_csv_paths()
        base_path = self.run_dir if self.run_dir else self.folder
        os.makedirs(f'run/{base_path}', exist_ok=True)
        
        if not self.locations_csv_initialized:
            with open(locations_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['timestep'] + [f'location_{i}_x' for i in range(len(self.swarm.members))] + \
                         [f'location_{i}_y' for i in range(len(self.swarm.members))]
                writer.writerow(header)
            self.locations_csv_initialized = True
        
        if not self.velocities_csv_initialized:
            with open(velocities_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['timestep'] + [f'velocity_{i}_x' for i in range(len(self.swarm.members))] + \
                         [f'velocity_{i}_y' for i in range(len(self.swarm.members))]
                writer.writerow(header)
            self.velocities_csv_initialized = True
        
        # Clear field lists (they should already be empty if save_fields is True and were saved)
        # But clear anyway to ensure no accumulation if save_fields is False
        self.fields_vx_list.clear()
        self.fields_vy_list.clear()
        self.fields_p_list.clear()
        self.field_timesteps.clear()
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Advance simulation by one step."""
        # Clip actions
        action = np.clip(action, -1.0, 1.0)
        
        # Advance simulation
        v_new, p_new, swarm_new = self.sim_step.step(
            self.v, self.p, self.swarm, self.episode_time, action
        )
        
        if v_new is None or p_new is None:
            # Simulation diverged
            obs = self._get_observation()
            return obs, 0.0, False, True, {'diverged': True}
        
        self.v = v_new
        self.p = p_new
        self.swarm = swarm_new
        
        # Update time
        self.current_time += self.sim_params.dt
        self.episode_time += self.sim_params.dt
        self.current_timestep += 1
        
        # Compute reward
        reward = self._compute_reward()
        self.rewards.append(reward)
        self.episode_cum_reward += float(reward)
        self.episode_cum_objectives['progress'] += float(self.last_objectives.get('progress', 0.0))
        self.episode_cum_objectives['energy_efficiency'] += float(self.last_objectives.get('energy_efficiency', 0.0))
        self.episode_cum_objectives['smoothness'] += float(self.last_objectives.get('smoothness', 0.0))
        
        # Save locations and velocities to CSV incrementally
        locations_csv_path, velocities_csv_path = self._get_csv_paths()
        with open(locations_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [self.episode_time] + \
                  [member.location['x'] for member in self.swarm.members] + \
                  [member.location['y'] for member in self.swarm.members]
            writer.writerow(row)
        
        with open(velocities_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [self.episode_time] + \
                  [member.velocity['x'] for member in self.swarm.members] + \
                  [member.velocity['y'] for member in self.swarm.members]
            writer.writerow(row)
        
        # Save fields if requested
        if self.save_fields and self.v is not None and self.p is not None:
            v_data = self.v.staggered_tensor()
            vx_np = v_data[0].numpy('x,y').astype(np.float64)
            vy_np = v_data[1].numpy('x,y').astype(np.float64)
            p_np = self.p.values.numpy('x,y').astype(np.float64)
            
            self.fields_vx_list.append(vx_np)
            self.fields_vy_list.append(vy_np)
            self.fields_p_list.append(p_np)
            self.field_timesteps.append(self.episode_time)
        
        # Check termination
        terminated = self._compute_terminated()
        truncated = self._compute_truncated()
        
        if terminated or truncated:
            self._finalize_episode(terminated, truncated)
        
        info = {
            'reward_components': self.last_reward_components,
            'objectives': self.last_objectives,
            'terminated': terminated,
            'truncated': truncated,
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Generate observation array."""
        obs = []
        for member in self.swarm.members:
            if self.p is not None:
                pressure_profile = self.physics.sample_field_around_member(
                    self.p, member, n_samples=4, offset=2
                )
            else:
                pressure_profile = np.zeros(4, dtype=np.float64)
            
            obs.append([
                member.location['x'],
                member.location['y'],
                member.velocity['x'],
                member.velocity['y'],
                *pressure_profile
            ])
        return np.array(obs, dtype=np.float64)
    
    def _compute_terminated(self) -> bool:
        """Check if episode should terminate."""
        return self.episode_time > self.episode_duration
    
    def _compute_truncated(self) -> bool:
        """Check if episode should truncate."""
        return self.v is None
    
    def _compute_reward(self) -> float:
        """Compute multi-objective reward."""
        # Warmup first step
        if self.episode_time <= self.sim_params.dt:
            self.last_reward_components = {'progress': 0.0, 'energy_efficiency': 0.0, 'smoothness': 0.0}
            self.last_objectives = {'progress': 0.0, 'energy_efficiency': 0.0, 'smoothness': 0.0}
            self.reward_components_history.append(self.last_reward_components.copy())
            self.objectives_history.append(self.last_objectives.copy())
            return 0.0
        
        # 1) Progress: same sign convention as RL.SwarmEnv (u_fluid_x - v_member_x), normalized.
        # Main repo uses ring-mean fluid u_x from sample_field_around_obstacles; this env uses bulk
        # inflow amplitude as a scalar u_x proxy when not sharing the main simulation stack.
        v_ref = max(abs(float(self.inflow_params.amplitude)), 1e-9)
        ux_bulk = float(self.inflow_params.amplitude)
        vx_mem = np.array([m.velocity['x'] for m in self.swarm.members], dtype=float)
        r_per_member = ux_bulk - vx_mem
        loc_unweighted = float(np.clip(float(np.mean(r_per_member)) / v_ref, -1.0, 1.0))
        r_progress = self.w_prog * loc_unweighted
        
        # 2) Energy efficiency: mean_i (1 - ||F_i|| / ||F_max||)
        energy_terms = []
        for member in self.swarm.members:
            if len(member.previous_actions) < 1:
                energy_terms.append(1.0)
                continue
            a = member.previous_actions[-1]
            ax, ay = float(a['x']), float(a['y'])
            fmax = float(member.max_force)
            f_mag = fmax * np.sqrt(ax * ax + ay * ay)
            f_max_mag = fmax * np.sqrt(2.0)
            if f_max_mag <= 0.0:
                eff = 1.0
            else:
                eff = float(np.clip(1.0 - f_mag / f_max_mag, 0.0, 1.0))
            energy_terms.append(eff)
        r_energy_unweighted = float(np.mean(energy_terms)) if energy_terms else 0.0
        r_energy = self.w_energy * r_energy_unweighted
        
        # 3) Smoothness: cosine similarity between actions
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
                smooth_vals.append((cos_sim + 1.0) / 2.0)
            else:
                smooth_vals.append(1.0)
        smoothness = float(np.mean(smooth_vals))
        r_smooth = self.w_smooth * smoothness
        
        total_reward = r_progress + r_energy + r_smooth
        
        self.last_reward_components = {
            'progress': float(r_progress),
            'energy_efficiency': float(r_energy),
            'smoothness': float(r_smooth)
        }
        
        self.last_objectives = {
            'progress': float(loc_unweighted),
            'energy_efficiency': float(r_energy_unweighted),
            'smoothness': float(r_smooth / self.w_smooth) if self.w_smooth > 0 else 0.0
        }
        
        self.reward_components_history.append(self.last_reward_components.copy())
        self.objectives_history.append(self.last_objectives.copy())
        
        return float(total_reward)
    
    def _finalize_episode(self, terminated: bool, truncated: bool) -> None:
        """Save episode summary to CSV."""
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
    
    def save_fields_to_disk(self) -> None:
        """Save accumulated fields to compressed npz file."""
        if not self.save_fields or len(self.fields_vx_list) == 0:
            return
        
        # Use run_dir if available, otherwise folder
        base_path = self.run_dir if self.run_dir else self.folder
        output_dir = f"run/{base_path}/fields"
        os.makedirs(output_dir, exist_ok=True)
        
        vx_array = np.stack(self.fields_vx_list, axis=0)
        vy_array = np.stack(self.fields_vy_list, axis=0)
        p_array = np.stack(self.fields_p_list, axis=0)
        timesteps_array = np.array(self.field_timesteps, dtype=np.float64)
        
        # Save with episode index to avoid overwriting
        output_path = f"{output_dir}/fields_episode_{self.episode_index}.npz"
        np.savez_compressed(
            output_path,
            vx=vx_array,
            vy=vy_array,
            p=p_array,
            timesteps=timesteps_array,
            length_x=self.sim_params.length_x,
            length_y=self.sim_params.length_y,
            resolution=self.sim_params.resolution
        )
        print(f"Saved {len(self.field_timesteps)} field snapshots to {output_path}")
        print(f"File size: {os.path.getsize(output_path) / (1024**2):.2f} MB")
        
        # Clear field lists after saving
        self.fields_vx_list.clear()
        self.fields_vy_list.clear()
        self.fields_p_list.clear()
        self.field_timesteps.clear()
    
    def render(self, mode: str = 'human') -> None:
        """Render the environment."""
        pass

