# wildlife_patrol_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class WildlifePatrolEnv(gym.Env):
    """
    Custom Environment for wildlife patrol optimization
    """
    
    def __init__(self, grid_size=5):
        super(WildlifePatrolEnv, self).__init__()
        
        self.grid_size = grid_size
        self.window_size = 512
        
        # Initialize episode rewards and steps
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_steps = 0
        self.max_episode_steps = 100  # Maximum steps per episode
        
        # Define action space (4 possible moves: up, down, left, right)
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 5x5 grid with multiple channels
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(grid_size, grid_size, 4),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.ranger_pos = np.array([0, 0])
        self.episode_steps = 0
        self.current_episode_reward = 0
        
        # Initialize static elements
        self.wildlife_spots = np.zeros((self.grid_size, self.grid_size))
        self.water_sources = np.zeros((self.grid_size, self.grid_size))
        
        # Place wildlife spots (static)
        self.wildlife_spots[1, 1] = 1
        self.wildlife_spots[3, 3] = 1
        self.wildlife_spots[4, 1] = 1
        
        # Place water sources (static)
        self.water_sources[2, 2] = 1
        self.water_sources[4, 4] = 1
        
        # Initialize poacher tracks (dynamic)
        self.poacher_tracks = np.random.uniform(
            0, 0.3, 
            (self.grid_size, self.grid_size)
        )
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute action and return new state, reward, done, truncated, info"""
        self.episode_steps += 1
        prev_pos = self.ranger_pos.copy()
        
        # Move ranger based on action
        if action == 0:  # up
            self.ranger_pos[0] = max(0, self.ranger_pos[0] - 1)
        elif action == 1:  # down
            self.ranger_pos[0] = min(self.grid_size - 1, self.ranger_pos[0] + 1)
        elif action == 2:  # left
            self.ranger_pos[1] = max(0, self.ranger_pos[1] - 1)
        elif action == 3:  # right
            self.ranger_pos[1] = min(self.grid_size - 1, self.ranger_pos[1] + 1)
        
        # Calculate reward
        reward = self._calculate_reward(prev_pos)
        self.current_episode_reward += reward
        
        # Update poacher tracks
        self._update_poacher_tracks()
        
        # Check if episode should end
        terminated = self.episode_steps >= self.max_episode_steps
        truncated = False
        
        if terminated:
            self.episode_rewards.append(self.current_episode_reward)
        
        return self._get_observation(), reward, terminated, truncated, {}
    
    def _calculate_reward(self, prev_pos):
        """Calculate reward based on current state and action"""
        reward = 0
        
        # Reward for monitoring wildlife spots
        if self.wildlife_spots[tuple(self.ranger_pos)] == 1:
            reward += 2.0
            
        # Reward for checking water sources
        if self.water_sources[tuple(self.ranger_pos)] == 1:
            reward += 1.5
            
        # Reward/penalty based on poacher track intensity
        track_intensity = self.poacher_tracks[tuple(self.ranger_pos)]
        if track_intensity > 0.1:
            reward += 3.0 * track_intensity
            
        # Small penalty for movement to encourage efficient patrolling
        reward -= 0.1
        
        return reward
    
    def _update_poacher_tracks(self):
        """Update poacher tracks - decay old ones and add new ones"""
        # Decay existing tracks
        self.poacher_tracks *= 0.95
        
        # Add new tracks with small probability
        new_tracks = np.random.random(self.poacher_tracks.shape) < 0.02
        self.poacher_tracks[new_tracks] = np.random.uniform(0.5, 1.0)
    
    def _get_observation(self):
        """Return current observation state"""
        obs = np.zeros((self.grid_size, self.grid_size, 4))
        
        # Channel 1: Ranger position
        obs[self.ranger_pos[0], self.ranger_pos[1], 0] = 1
        
        # Channel 2: Wildlife spots
        obs[:, :, 1] = self.wildlife_spots
        
        # Channel 3: Water sources
        obs[:, :, 2] = self.water_sources
        
        # Channel 4: Poacher tracks
        obs[:, :, 3] = self.poacher_tracks
        
        return obs