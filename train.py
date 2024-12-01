from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from wildlife_patrol_env import WildlifePatrolEnv
import numpy as np
import pygame
import time
import os

class VisualCallback(BaseCallback):
    """
    Custom callback for visualizing training in real-time
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.pygame_initialized = False
        self.cell_size = 100
        
    def _init_pygame(self, env):
        """Initialize pygame display"""
        pygame.init()
        self.screen = pygame.display.set_mode(
            (env.grid_size * self.cell_size, env.grid_size * self.cell_size)
        )
        pygame.display.set_caption("Wildlife Patrol Training Visualization")
        self.pygame_initialized = True
        
    def render_env(self, env):
        """Render the environment state"""
        # Clear screen
        self.screen.fill((255, 255, 255))
        
        # Get current state
        obs = env._get_observation()
        
        # Draw grid
        for i in range(env.grid_size):
            for j in range(env.grid_size):
                pygame.draw.rect(
                    self.screen, 
                    (200, 200, 200),
                    (j*self.cell_size, i*self.cell_size, 
                     self.cell_size, self.cell_size), 
                    1
                )
                
                # Draw wildlife spots (green)
                if obs[i, j, 1] == 1:
                    pygame.draw.circle(
                        self.screen, 
                        (0, 255, 0),
                        (j*self.cell_size + self.cell_size//2, 
                         i*self.cell_size + self.cell_size//2), 
                        self.cell_size//4
                    )
                
                # Draw water sources (blue)
                if obs[i, j, 2] == 1:
                    pygame.draw.circle(
                        self.screen, 
                        (0, 0, 255),
                        (j*self.cell_size + self.cell_size//2, 
                         i*self.cell_size + self.cell_size//2),
                        self.cell_size//4
                    )
                
                # Draw poacher tracks (red intensity)
                if obs[i, j, 3] > 0:
                    intensity = int(obs[i, j, 3] * 255)
                    pygame.draw.rect(
                        self.screen, 
                        (intensity, 0, 0),
                        (j*self.cell_size + 10, i*self.cell_size + 10,
                         self.cell_size - 20, self.cell_size - 20), 
                        0
                    )
        
        # Draw ranger (yellow)
        ranger_pos = np.where(obs[:,:,0] == 1)
        if len(ranger_pos[0]) > 0:
            pygame.draw.circle(
                self.screen, 
                (255, 255, 0),
                (ranger_pos[1][0]*self.cell_size + self.cell_size//2,
                 ranger_pos[0][0]*self.cell_size + self.cell_size//2),
                self.cell_size//3
            )
        
        # Add training info
        font = pygame.font.Font(None, 36)
        text = font.render(f"Steps: {self.n_calls}", True, (0, 0, 0))
        self.screen.blit(text, (10, 10))
        
        mean_reward = np.mean(self.training_env.get_attr('episode_rewards'))
        reward_text = font.render(f"Mean Reward: {mean_reward:.2f}", True, (0, 0, 0))
        self.screen.blit(reward_text, (10, 50))
        
        pygame.display.flip()
        
    def _on_step(self) -> bool:
        """Called after each step during training"""
        if not self.pygame_initialized:
            self._init_pygame(self.training_env.envs[0])
        
        # Update visualization every 10 steps
        if self.n_calls % 10 == 0:
            self.render_env(self.training_env.envs[0])
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return False
            
            # Small delay to make visualization visible
            time.sleep(0.01)
        
        return True

def main():
    # Create log dir
    log_dir = "wildlife_patrol_tensorboard/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment
    env = WildlifePatrolEnv()
    env = DummyVecEnv([lambda: env])
    
    # Initialize PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=log_dir
    )
    
    # Initialize visualization callback
    viz_callback = VisualCallback()
    
    # Train the agent
    try:
        model.learn(
            total_timesteps=1000000,
            callback=viz_callback,
            progress_bar=True
        )
        
        # Save the trained model
        model.save("wildlife_patrol_ppo")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")
        model.save("wildlife_patrol_ppo_interrupted")
    
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()