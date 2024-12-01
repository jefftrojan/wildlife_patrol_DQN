import pygame
import numpy as np
import sys
from stable_baselines3 import PPO
from wildlife_patrol_env import WildlifePatrolEnv

class GameState:
    def __init__(self):
        self.total_reward = 0
        self.steps = 0
        self.episode = 1
        self.last_action = None
        self.last_reward = 0
        self.current_strategy = ""

def get_direction(dx, dy):
    """Convert coordinate changes to cardinal directions"""
    if dx == -1: return "north"
    if dx == 1: return "south"
    if dy == -1: return "west"
    if dy == 1: return "east"
    return ""

def get_action_explanation(action):
    """Convert action number to descriptive text"""
    if isinstance(action, np.ndarray):
        action = action.item()
    return {
        0: "Moving Up",
        1: "Moving Down",
        2: "Moving Left",
        3: "Moving Right"
    }.get(action, "Unknown Action")

def get_strategic_insights(obs, ranger_pos, action, reward):
    """Generate detailed strategic insights about agent's decisions"""
    wildlife_spots = np.where(obs[:,:,1] == 1)
    water_sources = np.where(obs[:,:,2] == 1)
    poacher_tracks = obs[:,:,3]
    
    # Get current position
    ranger_x = ranger_pos[0][0] if len(ranger_pos[0]) > 0 else 0
    ranger_y = ranger_pos[1][0] if len(ranger_pos[1]) > 0 else 0
    
    # Analyze surroundings
    nearby_threats = []
    opportunities = []
    
    # Check adjacent cells for threats and opportunities
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        new_x, new_y = ranger_x + dx, ranger_y + dy
        if 0 <= new_x < obs.shape[0] and 0 <= new_y < obs.shape[1]:
            # Check for poacher tracks
            if poacher_tracks[new_x, new_y] > 0.3:
                nearby_threats.append(f"High poacher activity ({poacher_tracks[new_x, new_y]:.2f}) {get_direction(dx, dy)}")
            # Check for wildlife spots
            if obs[new_x, new_y, 1] == 1:
                opportunities.append(f"Wildlife spot {get_direction(dx, dy)}")
            # Check for water sources
            if obs[new_x, new_y, 2] == 1:
                opportunities.append(f"Water source {get_direction(dx, dy)}")
    
    # Analyze current position priority
    current_priority = "Patrolling"
    if poacher_tracks[ranger_x, ranger_y] > 0.1:
        current_priority = f"Investigating tracks (intensity: {poacher_tracks[ranger_x, ranger_y]:.2f})"
    elif obs[ranger_x, ranger_y, 1] == 1:
        current_priority = "Monitoring wildlife spot"
    elif obs[ranger_x, ranger_y, 2] == 1:
        current_priority = "Checking water source"
    
    return {
        'current_priority': current_priority,
        'nearby_threats': nearby_threats,
        'opportunities': opportunities,
        'action_reasoning': get_action_reasoning(action, nearby_threats, opportunities),
        'reward_explanation': get_reward_explanation(reward, current_priority)
    }

def get_action_reasoning(action, threats, opportunities):
    """Explain the reasoning behind the chosen action"""
    action_name = get_action_explanation(action)
    
    if threats and action_name.lower().split()[-1] in [t.split()[-1].lower() for t in threats]:
        return f"{action_name} - Responding to nearby threat"
    elif opportunities and action_name.lower().split()[-1] in [o.split()[-1].lower() for o in opportunities]:
        return f"{action_name} - Moving toward point of interest"
    else:
        return f"{action_name} - Exploratory patrol"

def get_reward_explanation(reward, current_priority):
    """Explain the received reward"""
    if reward > 2.0:
        return f"High reward ({reward:.1f}): Successfully detected significant activity"
    elif reward > 1.0:
        return f"Good reward ({reward:.1f}): {current_priority}"
    elif reward > 0:
        return f"Small reward ({reward:.1f}): Minor activity detected"
    else:
        return f"Movement cost ({reward:.1f}): Continuing patrol"

def calculate_cell_size(screen_width, screen_height, grid_size, info_height):
    """Calculate optimal cell size based on screen dimensions"""
    available_height = screen_height - info_height
    cell_size_by_height = available_height // grid_size
    cell_size_by_width = screen_width // grid_size
    return min(cell_size_by_height, cell_size_by_width)

def render_env(screen, env, game_state):
    """Render the environment using Pygame with dynamic sizing and strategic insights"""
    # Get screen dimensions and calculate sizes
    screen_width, screen_height = screen.get_size()
    info_height = screen_height // 5
    cell_size = calculate_cell_size(screen_width, screen_height, env.grid_size, info_height)
    grid_size = env.grid_size * cell_size
    padding = cell_size // 10
    grid_top_margin = info_height + padding
    
    # Get current state and insights
    obs = env._get_observation()
    ranger_pos = np.where(obs[:,:,0] == 1)
    insights = get_strategic_insights(obs, ranger_pos, game_state.last_action, game_state.last_reward)
    
    # Clear screen
    screen.fill((240, 240, 240))
    
    # Create info surface
    info_surface = pygame.Surface((screen_width, info_height))
    info_surface.fill((255, 255, 255))
    
    # Calculate font size based on screen dimensions
    font_size = min(32, max(24, screen_width // 40))
    font = pygame.font.Font(None, font_size)
    
    # Prepare info items with insights
    info_items = [
        f"Episode: {game_state.episode}",
        f"Steps: {game_state.steps}",
        f"Total Reward: {game_state.total_reward:.1f}",
        f"Current Priority: {insights['current_priority']}",
        f"Action: {insights['action_reasoning']}",
        f"Reward: {insights['reward_explanation']}"
    ]
    
    # Add threats and opportunities if present
    if insights['nearby_threats']:
        info_items.append(f"Threats: {', '.join(insights['nearby_threats'])}")
    if insights['opportunities']:
        info_items.append(f"Opportunities: {', '.join(insights['opportunities'])}")
    
    # Draw info items
    line_height = font_size + 5
    for i, text in enumerate(info_items):
        y_pos = i * line_height + padding
        if y_pos + line_height < info_height:  # Only draw if it fits
            text_surface = font.render(text, True, (0, 0, 0))
            info_surface.blit(text_surface, (padding, y_pos))
    
    # Draw grid with calculated positions
    grid_surface = pygame.Surface((grid_size, grid_size))
    grid_surface.fill((255, 255, 255))
    
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            # Calculate positions
            cell_x = j * cell_size
            cell_y = i * cell_size
            
            # Draw cell borders
            pygame.draw.rect(grid_surface, (150, 150, 150),
                           (cell_x, cell_y, cell_size - 2, cell_size - 2), 2)
            
            # Draw poacher tracks
            if obs[i, j, 3] > 0:
                intensity = int(obs[i, j, 3] * 255)
                s = pygame.Surface((cell_size - 6, cell_size - 6))
                s.set_alpha(intensity)
                s.fill((255, 0, 0))
                grid_surface.blit(s, (cell_x + 3, cell_y + 3))
            
            # Draw wildlife spots and water sources
            center_x = cell_x + cell_size // 2
            center_y = cell_y + cell_size // 2
            
            if obs[i, j, 1] == 1:  # Wildlife spots
                pygame.draw.circle(grid_surface, (34, 139, 34),
                                 (center_x, center_y), cell_size // 3, 4)
            
            if obs[i, j, 2] == 1:  # Water sources
                pygame.draw.circle(grid_surface, (0, 105, 148),
                                 (center_x, center_y), cell_size // 3, 4)
    
    # Draw ranger
    if len(ranger_pos[0]) > 0:
        center_x = ranger_pos[1][0] * cell_size + cell_size // 2
        center_y = ranger_pos[0][0] * cell_size + cell_size // 2
        pygame.draw.circle(grid_surface, (0, 0, 0),
                         (center_x, center_y), cell_size // 3 + 3)
        pygame.draw.circle(grid_surface, (255, 255, 0),
                         (center_x, center_y), cell_size // 3)
    
    # Draw legend
    legend_height = min(40, info_height // 5)
    legend_surface = pygame.Surface((screen_width, legend_height))
    legend_surface.fill((255, 255, 255))
    
    legend_items = [
        ("Ranger", (255, 255, 0)),
        ("Wildlife Spot", (34, 139, 34)),
        ("Water Source", (0, 105, 148)),
        ("Poacher Tracks", (255, 0, 0))
    ]
    
    # Position legend items
    legend_font_size = min(24, font_size)
    legend_font = pygame.font.Font(None, legend_font_size)
    legend_spacing = screen_width // len(legend_items)
    
    for i, (text, color) in enumerate(legend_items):
        x_pos = i * legend_spacing + padding
        pygame.draw.circle(legend_surface, color, 
                         (x_pos + 10, legend_height // 2), 8)
        text_surface = legend_font.render(text, True, (0, 0, 0))
        legend_surface.blit(text_surface, (x_pos + 25, legend_height // 4))
    
    # Calculate grid position to center it horizontally
    grid_x = (screen_width - grid_size) // 2
    
    # Blit all surfaces
    screen.blit(info_surface, (0, 0))
    screen.blit(legend_surface, (0, info_height - legend_height))
    screen.blit(grid_surface, (grid_x, grid_top_margin))
    
    pygame.display.flip()

def main():
    pygame.init()
    
    # Get the display info
    display_info = pygame.display.Info()
    initial_width = min(display_info.current_w - 100, 1024)
    initial_height = min(display_info.current_h - 100, 900)
    
    # Create resizable window
    screen = pygame.display.set_mode((initial_width, initial_height), pygame.RESIZABLE)
    pygame.display.set_caption("Wildlife Patrol Simulation")
    
    env = WildlifePatrolEnv()
    game_state = GameState()
    
    try:
        model = PPO.load("wildlife_patrol_ppo")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have trained the model first by running train.py")
        pygame.quit()
        return
    
    obs, _ = env.reset()
    clock = pygame.time.Clock()
    
    running = True
    paused = False
    
    # Display controls
    print("\nControls:")
    print("F: Toggle fullscreen")
    print("Space: Pause/Resume simulation")
    print("R: Reset simulation")
    print("ESC: Exit\n")
    
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_f:  # Toggle fullscreen
                        pygame.display.toggle_fullscreen()
                    elif event.key == pygame.K_SPACE:  # Pause/Resume
                        paused = not paused
                        print("Simulation", "paused" if paused else "resumed")
                    elif event.key == pygame.K_r:  # Reset
                        obs, _ = env.reset()
                        game_state = GameState()
                        print("Simulation reset")
                    elif event.key == pygame.K_ESCAPE:  # Exit
                        running = False
            
            if not paused:
                try:
                    # Get action from model
                    action, _ = model.predict(obs, deterministic=True)
                    if isinstance(action, np.ndarray):
                        action = action.item()
                    
                    # Execute action
                    new_obs, reward, terminated, truncated, _ = env.step(action)
                    
                    # Update game state
                    game_state.steps += 1
                    game_state.total_reward += reward
                    game_state.last_action = action
                    game_state.last_reward = reward
                    
                    if terminated or truncated:
                        obs, _ = env.reset()
                        game_state.episode += 1
                        game_state.steps = 0
                    else:
                        obs = new_obs
                    
                except Exception as e:
                    print(f"Error during simulation step: {e}")
                    running = False
            
            # Render
            render_env(screen, env, game_state)
            
            # Control simulation speed
            clock.tick(2)
    
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()