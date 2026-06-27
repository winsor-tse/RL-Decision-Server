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
- `total_timesteps`: Total training steps (default: 126000)
- `learning_rate`: Learning rate for optimizer (default: 2.5e-4)
- `buffer_size`: Replay buffer size (default: 10000)
- Additional parameters in the `Args` dataclass

**PPO_server.py:**
- `total_timesteps`: Total training steps (default: 500000)
- `learning_rate`: Learning rate for networks (default: 2.5e-4)
- `num_envs`: Parallel environments (default: 4)
- `num_steps`: Steps per policy rollout (default: 128)
- Additional parameters in the `Args` dataclass

Pass arguments from command line or modify defaults in the code.

## Custom Environment

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

- Both DQN and PPO run as single instances; only one should be active at a time
- The environment is stationary (game state doesn't change without player action)
- Discrete action space makes DQN more suitable for this use case
- Agent learns to maximize cumulative reward through interactions with Yugen Saga
