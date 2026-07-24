# RL Decision Server

A bridge between a reinforcement learning agent and the Yugen Saga game. The project connects the browser game to Python RL training code through a WebSocket-to-ZMQ bridge, then trains or evaluates agents with custom Gymnasium environments.

## Overview

- `Automation/Bridge/ws_zmq_bridge.py`: translates WebSocket messages from Yugen Saga into ZMQ backend requests.
- `Training/DQN_server.py`: DQN training loop for the current custom environment.
- `Inference/dqn_eval.py`: loads a saved DQN checkpoint and runs evaluation.
- `RunRL.ps1`: starts TensorBoard, the bridge, and the configured RL algorithm together.
- `Utils/buffers.py`: replay buffer used by DQN.
- `Custom_enviornments/`: shared env config, base env, and class-specific environments.

## DQN Learning Showcase

The clips below show the DQN agent interacting with the live game while it learns to select movement, targeting, and combat actions from the environment state and reward signal.

<p align="center">
  <img src="Sample_Runs/SampleGif.gif" alt="DQN agent gameplay sample 1" width="32%">
  <img src="Sample_Runs/SampleGif2.gif" alt="DQN agent gameplay sample 2" width="32%">
  <img src="Sample_Runs/SampleGif3.gif" alt="DQN agent gameplay sample 3" width="32%">
</p>

The training dashboard captures the learning signals produced during a demo run, including episodic return, epsilon decay, Q-values, and temporal-difference loss.

![TensorBoard metrics from a DQN demo run](Sample_Runs/Demo_Run.png)

## Project Structure

```text
.
|-- requirements.txt
|-- pyproject.toml
|-- InstallationReadMe.md
|-- RunRL.ps1
|-- RunInference.ps1
|-- Automation/
|   |-- automation_config.yaml
|   |-- processes.py
|   |-- train.py
|   |-- infer.py
|   |-- RunTensorboard.bat
|   `-- Bridge/
|       `-- ws_zmq_bridge.py
|-- Utils/
|   `-- buffers.py
|-- Training/
|   |-- DQN_server.py
|   `-- PPO_server.py
|-- Inference/
|   `-- dqn_eval.py
|-- Custom_enviornments/
|   |-- BaseEnv.py
|   |-- Config.yaml
|   |-- Load_env_config.py
|   `-- Test_Env/
|       |-- Env_16.py
|       `-- Env_conditions.py
|-- Tests/
|   |-- test_automation.py
|   |-- test_dqn_metrics.py
|   `-- test_environment.py
`-- README.md
```

## Custom Environments

Environment code is split by responsibility:

- `Custom_enviornments/Config.yaml`: shared state-space and runtime constants used by all envs.
- `Custom_enviornments/Load_env_config.py`: the only config parser used by environments.
- `Custom_enviornments/BaseEnv.py`: bare Gymnasium/ZMQ base environment.
- `Custom_enviornments/Test_Env/Env_16.py`: current class-specific discrete-action environment.
- `Custom_enviornments/Test_Env/Env_conditions.py`: example class-specific observation parsing, rewards, termination, and truncation.

Each new game/class environment should have its own folder, action-space file, and `Env_conditions.py`. Shared state-space config stays in `Config.yaml`; class-specific reward and end-condition logic stays beside that class env.

## Shared Env Config

`Custom_enviornments/Config.yaml` defines the shared state space:

- `ZMQ_BIND_URL`: ZMQ endpoint used by the browser/game bridge.
- `OBS_PLAYER_SIZE`: number of player features.
- `OBS_ENEMY_SIZE`: number of features per tracked enemy.
- `MAX_ENEMIES`: number of enemies included in the observation state.
- `OBS_SIZE`: total observation size, validated as `OBS_PLAYER_SIZE + OBS_ENEMY_SIZE * MAX_ENEMIES`.
- `MAX_EPISODE_STEPS`: shared runtime default that env-specific conditions can use.
- `STATE_DTYPE`: dtype metadata for state tensors.
- `ACTION_SPACE_TYPE`: action-space metadata.

The current observation space contains 26 normalized/numeric values:

- 6 player features at indices `0-5`: map X, map Y, direction, HP percentage,
  MP percentage, and map ID.
- 5 enemy blocks with 4 features each: distance, direction, HP percentage,
  and MP percentage.
- Enemies are sorted by distance, so the first block represents the nearest
  enemy.
- Missing enemy blocks are zero-padded.

For enemy index `n`, its block starts at:

```text
OBS_PLAYER_SIZE + n * OBS_ENEMY_SIZE
```

The nearest enemy therefore starts at index `6`, and its HP percentage is at
index `8` (`OBS_PLAYER_SIZE + 2`). Player and enemy HP metrics range from
`0.0` to `1.0`.

## Current Env

`Custom_enviornments/Test_Env/Env_16.py` is the active example env used by `Training/DQN_server.py` and `Inference/dqn_eval.py`.

It exposes 15 discrete actions:

- Movement: `up`, `down`, `left`, `right`
- Direction: `direction:up`, `direction:down`, `direction:left`, `direction:right`
- Combat: `attack`
- Spells: `castSpell:1`, `castSpell:2`, `castSpell:3`, `castSpell:5`, `castSpell:6`, and `castSpell:7`

## Running Training

Recommended one-command startup:

```powershell
.\RunRL.ps1
```

Or directly from any shell:

```bash
python -m Automation.train
```

The launcher reads `Automation/automation_config.yaml`, starts TensorBoard when enabled, starts the bridge, waits for its configured readiness signal, and then starts the selected RL algorithm. Configured commands are YAML argument lists and use the active Python interpreter.

Manual startup is still supported.

Start the WebSocket/ZMQ bridge first:

```bash
python -m Automation.Bridge.ws_zmq_bridge
```

Then start DQN training:

```bash
python -m Training.DQN_server
```

To change how often regular step metrics are written:

```bash
python -m Training.DQN_server --metrics-frequency 25
```

The game must be running and sending `ai_tick` messages through the browser extension before the env can step.

## Monitoring

Training logs are written under `runs/`. The automation launcher starts
TensorBoard by default; when training manually, start it with:

```bash
tensorboard --logdir=runs
```

Open http://localhost:6006.

Useful metrics:

- `charts/step_reward`
- `charts/episodic_return`
- `charts/episode_length`
- `charts/win_rate`
- `charts/epsilon`
- `losses/td_loss`
- `losses/mean_q_value`
- `training/replay_buffer_size`
- `environment/player_hp`
- `environment/enemy_hp`
- `environment/action_frequency/*`
- `rewards/damage_dealt`
- `rewards/damage_taken`
- `rewards/terminal`
- `rewards/health_state`
- `rewards/positioning`
- `rewards/episode_*`

Step and environment metrics are sampled every 10 steps by default and always
written for non-zero reward events and episode endings. Change the regular
sampling interval with `--metrics-frequency`.
Episode return, length, win rate, and episode reward-component totals are
written when an episode ends.

### Metric meanings

| Metric | Meaning |
|---|---|
| `charts/step_reward` | Total reward returned for the sampled environment step. |
| `charts/episodic_return` | Sum of rewards over the completed episode. |
| `charts/episode_length` | Number of environment steps in the completed episode. |
| `charts/win_rate` | Cumulative fraction of completed episodes classified as wins. |
| `charts/epsilon` | Current epsilon-greedy exploration probability. |
| `losses/td_loss` | DQN temporal-difference mean-squared error. |
| `losses/mean_q_value` | Mean selected-action Q-value in the training batch. |
| `training/replay_buffer_size` | Number of transitions currently available in replay memory. |
| `environment/player_hp` | Current player HP percentage from observation index `3`. |
| `environment/enemy_hp` | Nearest-enemy HP percentage from observation index `8`. |

`environment/action_frequency` is a histogram of actions taken since the last
metric write. Tags below `environment/action_frequency/*` show each action's
cumulative fraction, such as `environment/action_frequency/attack` and
`environment/action_frequency/castSpell_1`.

### Reward components

The environment records signed reward components in `info["reward_components"]`.
Their sum is exactly the reward returned to DQN:

| Component | Meaning |
|---|---|
| `damage_dealt` | Positive reward for reducing the nearest enemy's HP. |
| `damage_taken` | Negative reward when the player's HP decreases. |
| `terminal` | Kill bonus or loss penalty. |
| `health_state` | Penalty for remaining at low health. |
| `positioning` | Position-based penalty from the current environment rules. |

The corresponding `rewards/episode_*` metrics accumulate each component across
the complete episode. Console output is intentionally limited to episode
summaries and checkpoint-save messages; raw world-state details use debug
logging instead of per-step printing.

## Evaluation

After training creates a checkpoint:

```bash
python -m Inference.dqn_eval --model-path runs/DQN_server__<timestamp>/DQN_server.pt
```

Optional examples:

```bash
python -m Inference.dqn_eval --model-path runs/DQN_server__<timestamp>/DQN_server.pt --eval-episodes 20
python -m Inference.dqn_eval --model-path runs/DQN_server__<timestamp>/DQN_server.pt --epsilon 0.05
python -m Inference.dqn_eval --model-path runs/DQN_server__<timestamp>/DQN_server.pt --cuda false
```

The inference launcher uses the newest `.pt` checkpoint under `runs/` by
default. Set `inference_model_path` in `Automation/automation_config.yaml` to
select a specific checkpoint, or pass a complete command:

```powershell
.\RunInference.ps1 -Command "python -m Inference.dqn_eval --model-path runs/DQN_server__<timestamp>/DQN_server.pt"
```

## Notes

- DQN is the active training path for the current discrete action space.
- Run Python entry points with `python -m ...`; package imports no longer depend on script-relative `sys.path` changes.
- `Load_env_config.py` should remain the single place that parses shared env config.
- `Automation/automation_config.yaml` controls TensorBoard startup, bridge startup, training command, and inference command.
