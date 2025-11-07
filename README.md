# RL-Decision-Server

This project is a bridge between a Reinforcement Learning (RL) agent and an external application (e.g., game engine or simulation) via a REST API. It enables communication of actions and observations between the agent and the environment, allowing you to train or run RL agents in a decoupled system.

## Features

- Custom Gymnasium environment (`CustomBlankEnv`). Note Enviornments can be customized, Action and Observation spaces need to be matched in order for Agent to be trained.
- Communication via REST (`FastAPI`).
- Action + Observation exchange via `/last-action` and `/observation` endpoints.
- Dummy API server to simulate or test the system.
- Plug-and-play with any RL loop or game engine that supports HTTP.

## Quick Start

### 1. Clone the Repository

### 2. Install Dependencies

### 3. Run the Server (TBD)

For now, Run main function inside Dummy_env. PPO server is yet to be tested and integrated.

## FAST API Endpoints


| Method | Endpoint       | Description                    |
| -------- | ---------------- | -------------------------------- |
| POST   | `/observation` | Send observation to the agent  |
| GET    | `/observation` | Get latest observation         |
| POST   | `/last-action` | Send agent's action            |
| GET    | `/last-action` | Get last action from the agent |
| POST   | `/reset`       | Trigger environment/game reset |
