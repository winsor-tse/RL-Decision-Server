# Installation

Setup guide for running the RL Decision Server with the current custom environment layout.

## Prerequisites

- Python 3.8+
- Yugen Saga local server running
- Chrome extension installed from `Yugen-Battler-Custom`
- PowerShell, Windows Terminal, or another shell from the project root

## 1. Clone And Enter The Repo

```bash
git clone <repository-url>
cd "RL Server"
```

## 2. Create And Activate A Virtual Environment

PowerShell:

```powershell
python -m venv RL_venv
.\RL_venv\Scripts\Activate.ps1
```

Command Prompt:

```bat
python -m venv RL_venv
RL_venv\Scripts\activate.bat
```

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If TensorBoard is missing when running `Training/DQN_server.py`, install it into the same active environment:

```bash
python -m pip install tensorboard
```

## 4. Verify Imports

Run these from the repo root:

```bash
python -m compileall Custom_enviornments Utils Training Inference Automation Tests
python -c "from Custom_enviornments.Test_Env import Env_16; print(Env_16.Env16.__name__)"
python -m unittest discover -s Tests -p "test_*.py"
```

Expected output includes:

```text
Env16
```

### Check NVIDIA/PyTorch compatibility

Run the diagnostic with the same virtual environment used for training:

```powershell
.\RL_venv\Scripts\python.exe -m Tests.pytorch_gpu_check
```

It compares the NVIDIA driver's maximum CUDA version, GPU compute capability,
installed PyTorch CUDA runtime, compiled architectures, and a real CUDA matrix
operation. If the environment has a CPU-only or incompatible PyTorch build, it
prints a suitable project-version install command without executing it.

## 5. Check Shared Env Config

Shared state-space config lives here:

```text
Custom_enviornments/Config.yaml
```

The config is parsed only by:

```text
Custom_enviornments/Load_env_config.py
```

All environments use this config for shared state-space values such as `OBS_PLAYER_SIZE`, `OBS_ENEMY_SIZE`, `MAX_ENEMIES`, and `OBS_SIZE`. Each class-specific env keeps its own action space and `Env_conditions.py`.

## 6. Start Yugen Saga

Start the Yugen Saga local server and make sure the browser extension is connected.

## 7. Start Bridge And Training Together

The cleaner startup path is the automation script:

```powershell
.\RunRL.ps1
```

The PowerShell wrappers automatically use `RL_venv\Scripts\python.exe` when
that environment exists; otherwise they use `python` from `PATH`.

Or directly from any shell:

```bash
python -m Automation.train
```

These scripts read:

```text
Automation/automation_config.yaml
```

The launcher starts TensorBoard when enabled, starts `Automation/Bridge/ws_zmq_bridge.py`, waits until the bridge prints the configured ready signal, then starts the configured RL algorithm command.

## 8. Manual Startup: Start The WebSocket-ZMQ Bridge

In one terminal:

```bash
python -m Automation.Bridge.ws_zmq_bridge
```

The bridge listens for browser messages and forwards AI ticks to the Python env through ZMQ.

## 9. Manual Startup: Start Training

In another terminal with the same virtual environment active:

```bash
python -m Training.DQN_server
```

PPO and recurrent PPO use the same live environment:

```bash
python -m Training.PPO_server
python -m Training.PPO_lstm_server
```

The active env is:

```text
Custom_enviornments/Test_Env/Env_16.py
```

The active env conditions are:

```text
Custom_enviornments/Test_Env/Env_conditions.py
```

## 10. Monitor Training

```bash
python -m Automation.tensorboard_server --logdir runs
```

Open http://localhost:6006.

## 11. Evaluate A Checkpoint

Evaluate recurrent PPO on the custom `Env16` environment:

```bash
python -m Inference.ppo_lstm_eval
python -m Inference.ppo_lstm_eval --no-deterministic
```

Evaluate feed-forward PPO:

```bash
python -m Inference.ppo_eval
python -m Inference.ppo_eval --no-deterministic
```

Both PPO evaluators automatically load the newest matching checkpoint from
`runs/<run_name>/`. Pass `--model-path` only when selecting a specific run.

Evaluate DQN:

```bash
python -m Inference.dqn_eval --model-path runs/DQN_server__<timestamp>/DQN_server.pt
```

Useful options:

```bash
python -m Inference.dqn_eval --model-path runs/DQN_server__<timestamp>/DQN_server.pt --eval-episodes 20
python -m Inference.dqn_eval --model-path runs/DQN_server__<timestamp>/DQN_server.pt --epsilon 0.05
python -m Inference.dqn_eval --model-path runs/DQN_server__<timestamp>/DQN_server.pt --cuda false
```

To run inference with bridge automation:

```powershell
.\RunInference.ps1
```

The launcher uses `inference_algorithm` plus the matching
`dqn_inference_command`, `ppo_inference_command`, or
`ppo_lstm_inference_command` from `Automation/automation_config.yaml`. PPO
automation commands use explicit run-local checkpoint paths, matching DQN;
replace `XXXX` with the desired run directory. Direct PPO evaluator commands
select the newest checkpoint when `--model-path` is omitted. Pass `-Command` to
override the configured evaluator.

## Troubleshooting

If Python cannot import a project package, confirm that the editable install completed or run module commands from the repo root:

```text
C:\Users\winds\Desktop\YugenSaga\RL Server
```

If ZMQ binding fails, check whether another process is already using:

```text
tcp://127.0.0.1:5555
```

If the agent waits forever on reset or step, make sure the bridge, browser extension, and Yugen Saga local server are all running.
