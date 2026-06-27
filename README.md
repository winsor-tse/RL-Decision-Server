# RL-Decision-Server

A bridge between a Reinforcement Learning (RL) agent and the Yugen Saga game using Deep Reinforcement Learning (DRL) algorithms. This server facilitates communication between RL algorithms (DQN and PPO) and the game engine via WebSocket and ZMQ protocols.

## Overview

The RL Decision Server implements:

- **DQN (Deep Q-Network)**: Selected for discrete action space and stationary environment
- **PPO (Proximal Policy Optimization)**: Alternative policy gradient approach
- **Custom Gymnasium Environment**: Parses game state data and defines reward functions
- **POMDP-Aware Features**: Uses player location and nearby enemy positions (within sensing range) for attention mechanisms

### Project Architecture

- **ws_zmq_bridge.py**: Translates WebSocket messages from Yugen Saga to ZMQ backend requests
- **DQN_server.py / PPO_server.py**: RL agent servers that process observations and select actions
- **Custom_env/**: Game state parsing, reward functions, and environment wrapper
- **buffers.py**: Experience replay buffer for DQN training

## Prerequisites

- Python 3.8+
- Yugen Saga local server running
- Chrome extension installed from [Yugen-Battler-Custom](https://github.com/winsor-tse/Yugen-Battler-Custom)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "RL Server"
```

### 2. Create Virtual Environment

```bash
python -m venv RL_venv
```

Activate the virtual environment:

- **Windows (PowerShell)**: `.\RL_venv\Scripts\Activate.ps1`
- **Windows (CMD)**: `.\RL_venv\Scripts\activate.bat`
- **Linux/Mac**: `source RL_venv/bin/activate`

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the RL Server with Yugen Saga

### Step 1: Start Yugen Saga Local Server

Launch your Yugen Saga local server instance.

### Step 2: Install Chrome Extension

Install the custom Chrome extension from [Yugen-Battler-Custom](https://github.com/winsor-tse/Yugen-Battler-Custom) to enable communication between the game and the RL server.

### Step 3: Start WebSocket-ZMQ Bridge

```bash
python ws_zmq_bridge.py
```

This bridge:

- Listens for WebSocket connections from Yugen Saga
- Translates game state messages to ZMQ requests
- Routes RL agent decisions back to the game

### Step 4: Start the RL Agent Server

Choose **either** DQN or PPO:

**For DQN (recommended for discrete actions):**

```bash
python DQN_server.py
```

**For PPO:**

```bash
python PPO_server.py
```

The agent will begin training on game interactions.

## Monitoring Training Progress

### TensorBoard Visualization

View real-time training metrics:

```bash
./RunTensorboard.bat
```

Or manually:

```bash
tensorboard --logdir=runs
```

Then open http://localhost:6006 in your browser.

Training logs are saved in the `runs/` directory with timestamps.

## Configuration

Hyperparameters can be modified by editing the source files:

**DQN_server.py:**

- `total_timesteps`: Total training steps (Scale this to time it takes for external sim (yugen saga to respond with an action)
- `learning_rate`: Learning rate for optimizer (default: 2.5e-4)
- `buffer_size`: Replay buffer size (default: 10000)
- Additional parameters in the `Args` dataclass

**PPO_server.py (Not implemented on this game):**

- `total_timesteps`: Total training steps (default: 500000)
- `learning_rate`: Learning rate for networks (default: 2.5e-4)
- `num_envs`: Parallel environments (default: 4)
- `num_steps`: Steps per policy rollout (default: 128)
- Additional parameters in the `Args` dataclass

Pass arguments from command line or modify defaults in the code.

## Custom Environment

The ability to create more enviornments will depend on classes different abilities (TBD), obs space will usually stay the same. Reward design per class can also be varied (TBD).

### Action Space

The agent has 15 discrete actions:

- Movement: `up`, `down`, `left`, `right`
- Directional: `direction:up`, `direction:down`, `direction:left`, `direction:right`
- Combat: `attack`
- Spells: `castSpell:1` through `castSpell:7`

### Observation Space

- **Player state**: 5 features (position, health, etc.)
- **Enemy state**: 4 features × 2 enemies maximum (position, health, etc.)
- **Total**: 13-dimensional observation vector
- Enemies outside sensing range are zero-padded

### Reward Design

Rewards are based on:

- Player performance metrics
- Enemy proximity and state
- Episode progression
- Generalization through POMDP-aware features

See `Custom_env/` for implementation details.


## Training and Inference

### Training Loop Overview

The training process follows a standard DRL workflow:

1. **Exploration Phase** (Steps 0 to learning_starts): Agent explores randomly to populate replay buffer
2. **Learning Phase**: Agent learns from experiences while continuing to explore
3. **Convergence Phase**: Epsilon decays, exploration decreases, agent exploits learned policy

#### DQN Training Flow

```
For each timestep:
  1. Select action using epsilon-greedy policy
  2. Execute action in Yugen Saga environment
  3. Receive reward and next observation
  4. Store transition in replay buffer
  5. Sample batch from replay buffer
  6. Compute Bellman TD-error loss
  7. Update Q-network with backpropagation
  8. Periodically update target network
  9. Log metrics to TensorBoard
```

### Epsilon-Greedy Exploration

The agent balances exploration and exploitation:

- **Start Epsilon**: 1.0 (100% random actions)
- **End Epsilon**: 0.05 (95% greedy actions)
- **Exploration Fraction**: 0.5 (epsilon decays over first 50% of training)

The epsilon decay schedule is linear:
```
epsilon = start_e + (end_e - start_e) * (global_step / exploration_steps)
```

**Important**: At inference time, epsilon should be fixed at a low value (typically 0.0 or 0.05) to run the learned policy without random exploration.

### Experience Replay

DQN maintains an experience replay buffer that stores transitions:

- **Buffer Size**: 10,000 (configurable)
- **Batch Size**: 64 samples per training step
- **Sampling**: Uniformly random from buffer

The replay buffer provides:
- **Decorrelated Samples**: Breaks temporal correlations in the data
- **Improved Stability**: Smooths training by using diverse past experiences
- **Off-Policy Learning**: Can reuse old experiences

### Network Architecture

Both Q-network and target network follow:

```
Input Layer (13 units) → ReLU
Hidden Layer 1 (120 units) → ReLU
Hidden Layer 2 (84 units) → ReLU
Output Layer (15 units) → Q-values per action
```

### Training Parameters

**DQN_server.py defaults:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `total_timesteps` | 5000 | Total training interactions (scaled by Yugen Saga action time ~0.5s) |
| `learning_rate` | 2.5e-4 | Adam optimizer learning rate |
| `learning_starts` | 100 | Timesteps before learning begins (populate buffer) |
| `train_frequency` | 10 | Perform training every N timesteps |
| `target_network_frequency` | 50 | Update target network every N timesteps |
| `batch_size` | 64 | Samples per training batch |
| `gamma` | 0.99 | Discount factor for future rewards |
| `tau` | 1.0 | Target network soft-update rate |

**TD-Loss (Bellman Loss):**

```
TD-Target = reward + gamma * max_Q(next_obs) * (1 - done)
Loss = MSE(Q(obs, action) - TD-Target)
```

### Model Checkpointing

Models are saved automatically during training:

- **Save Frequency**: Every 100 global steps
- **Save Location**: `runs/{exp_name}__{timestamp}/{exp_name}.pt`
- **Format**: PyTorch state dict

Models are overwritten at each checkpoint, keeping only the latest snapshot.

**Checkpoint Selection Tips:**

1. ✅ **Good indicators for deployment:**
   - High episodic return (cumulative reward)
   - Low epsilon (< 0.1) - agent stopped exploring
   - Stable Q-values - loss not spiking
   - Converged TD-loss - no longer decreasing

2. ❌ **Avoid deploying models where:**
   - High episodic return BUT epsilon still high - may not have converged
   - TD-loss very low but episodic return low - overfitting to buffer
   - Q-values unstable - agent still learning unpredictably

### Inference / Model Evaluation

Inference is performed using the saved Q-network without exploration:

**Loading a Trained Model:**

```python
from DQN_server import QNetwork
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QNetwork(env).to(device)
model.load_state_dict(torch.load("runs/DQN_server__1234567890/DQN_server.pt"))
model.eval()  # Set to evaluation mode (disables dropout, batch norm updates)
```

**Inference Step:**

```python
with torch.no_grad():  # Disable gradient computation
    q_values = model(torch.Tensor(observation).to(device))
    action = torch.argmax(q_values).item()
```

**DQN_eval.py** (evaluation script):

- Loads a trained model from checkpoint
- Runs the model for N evaluation episodes
- **Epsilon = 0.0** (no exploration, pure greedy policy)
- Records episodic returns
- Reports average performance

Usage:
```bash
python DQN_eval.py --model_path runs/DQN_server__1234567890/DQN_server.pt --eval_episodes 10
```

### Training Metrics in TensorBoard

Monitor these metrics in TensorBoard:

| Metric | Location | Interpretation |
|--------|----------|-----------------|
| `charts/episodic_return` | Scalars | Cumulative reward per episode (higher is better) |
| `losses/td_loss` | Scalars | Bellman TD-error (should decrease over time) |
| `losses/q_values` | Scalars | Mean Q-values (useful for debugging) |
| `charts/epsilon` | Scalars | Exploration rate (should decay to end_e) |
| `charts/SPS` | Scalars | Steps per second (throughput) |

### Practical Training Workflow

1. **Start Training**
   ```bash
   python DQN_server.py
   ```
   
2. **Monitor Real-Time Progress**
   ```bash
   tensorboard --logdir=runs
   ```
   Open http://localhost:6006

3. **Evaluate Training**
   - Watch `episodic_return` and `td_loss` curves
   - Ensure `epsilon` is decaying smoothly
   - Stop training when metrics plateau

4. **Select Best Checkpoint**
   - Identify timestamp of best model from TensorBoard
   - Note the checkpoint path: `runs/DQN_server__<timestamp>/DQN_server.pt`

5. **Deploy for Inference**
   ```bash
   python DQN_eval.py --model_path runs/DQN_server__<timestamp>/DQN_server.pt
   ```

### Performance Expectations

For Yugen Saga training:

- **Convergence Time**: 30-60 minutes (5000 timesteps at 0.5s per action)
- **Expected Return**: Domain-dependent (see your TensorBoard logs)
- **Exploration Duration**: ~2500 steps (50% of 5000)
- **Stable Learning**: Typically after 3000+ steps

## Project Structure

```
.
├── DQN_server.py          # DQN training and inference server
├── PPO_server.py          # PPO training and inference server
├── ws_zmq_bridge.py       # WebSocket ↔ ZMQ bridge
├── buffers.py             # Experience replay buffer
├── Custom_env/
│   ├── Test_env.py        # Custom Gymnasium environment
│   └── Parse_data.py      # Game state parsing utilities
├── runs/                  # TensorBoard event logs
├── Sample_Data/           # Sample training data
├── Tests/                 # Test utilities
├── RL_venv/              # Python virtual environment
└── README.md             # This file
```

## Notes

- The environment is stationary (game state doesn't change without player action)
- Discrete action space makes DQN more suitable for this use case
- Agent learns to maximize cumulative reward through interactions with Yugen Saga
