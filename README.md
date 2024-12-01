# Wildlife Patrol Reinforcement Learning Environment

## Project Description
This project implements a custom reinforcement learning environment using OpenAI Gym to optimize wildlife ranger patrol routes. The environment simulates a wildlife reserve section where a ranger (agent) must efficiently patrol to protect wildlife, monitor water sources, and detect potential poaching activities.

### Key Features
- 5x5 grid environment representing a wildlife reserve section
- Dynamic poacher track generation and decay
- Multiple observation channels (ranger position, wildlife spots, water sources, poacher tracks)
- Reward system based on conservation priorities
- Real-time visualization of the patrol simulation

## Project Structure
```
wildlife-patrol-rl/
├── wildlife_patrol_env.py   # Custom Gym environment
├── train.py                 # Training script
├── play.py                  # Visualization/simulation script
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Requirements
- Python 3.7 or newer
- PyTorch
- Stable-Baselines3
- Pygame
- Gymnasium (OpenAI Gym)
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jefftrojan/wildlife_patrol_DQN.git
cd wildlife_patrol_DQN
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Project

### Training the Agent
To train the reinforcement learning agent:
```bash
python train.py
```
This will:
- Create the custom environment
- Initialize a PPO/DQN agent
- Train for 1 million timesteps
- Save the trained model as "wildlife_patrol"
- Generate tensorboard logs in "./wildlife_patrol_tensorboard/"

### Running the Simulation
To visualize the trained agent in action:
```bash
python play.py
```
This will:
- Load the trained model
- Launch a Pygame window showing the simulation
- Display the agent's patrol behavior in real-time

## Environment Details

### State Space
The environment uses a 4-channel observation space:
- Channel 1: Ranger position (binary)
- Channel 2: Wildlife spots (binary)
- Channel 3: Water sources (binary)
- Channel 4: Poacher tracks (continuous 0-1)

### Action Space
Four discrete actions:
- 0: Move up
- 1: Move down
- 2: Move left
- 3: Move right

### Reward Structure
- +2.0 for monitoring wildlife spots
- +1.5 for checking water sources
- +3.0 * track_intensity for investigating poacher tracks
- -0.1 movement penalty to encourage efficient patrolling

## Visualization Guide
In the simulation window:
- Yellow circle: Ranger (agent)
- Green circles: Wildlife spots
- Blue circles: Water sources
- Red shading: Poacher track intensity
- Grid lines: Reserve section boundaries

## Performance Monitoring
Training progress can be monitored using Tensorboard:
```bash
tensorboard --logdir=./wildlife_patrol_tensorboard/
```

## Known Issues and Limitations
- The environment currently uses a fixed 5x5 grid size
- Poacher track generation is simplified and probabilistic
- The simulation runs at a fixed 2 FPS for visibility


### demo video link 

-   https://youtu.be/vKehofkK_Ao 

### simulation video link 

- https://drive.google.com/file/d/1veJZ0wZ4PyJlD_E7AeNu4Q4n8U6JQt0f/view?usp=sharing
