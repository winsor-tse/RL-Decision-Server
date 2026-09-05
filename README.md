# RL Decision Server

A bridge between a reinforcement learning agent and the Yugen Saga game. The project connects the browser game to Python RL training code through a WebSocket-to-ZMQ bridge, then trains or evaluates agents with custom Gymnasium environments.

## Overview

- `Automation/Bridge/ws_zmq_bridge.py`: translates WebSocket messages from Yugen Saga into ZMQ backend requests.
- `Training/PPO_server.py`: active PPO training loop and checkpoint writer.
- `Training/PPO_lstm_server.py`: recurrent PPO training loop for the same live environment.
- `Training/DQN_server.py`: legacy DQN training loop.
- `Inference/ppo_lstm_eval.py`: evaluates recurrent PPO checkpoints on `Env16`.
- `Inference/ppo_eval.py`: evaluates PPO checkpoints deterministically or by policy sampling.
- `Inference/dqn_eval.py`: evaluates DQN checkpoints with optional epsilon exploration.
- `Inference/any_percent_bc_eval.py`: evaluates BC checkpoints on Minari data or `Env16`.
- `Offline/awac.py`: trains discrete AWAC from local BC-Minari demonstrations, with optional live `Env16` fine-tuning and TensorBoard logging.
- `Offline/record_player.py`: records full world-state payloads and global player input to SQLite.
- `Offline/record_minari.py`: records mapped human actions as a Minari behavior-cloning dataset.
- `Custom_enviornments/Test_Env/Env_16_BC.py`: no-op, valid-action-only recording environment.
- `Automation/automation_config.yaml`: selects the training and inference entry points.
- `Automation/offline_rl.py`: routes BC training, dataset analysis, and live evaluation.
- `RunRL.ps1`: starts TensorBoard, the bridge, and the configured RL algorithm together.
- `RunInference.ps1`: starts the bridge and the configured evaluator.
- `RunOfflineRL.ps1`: trains or evaluates an any-percent BC model.
- `RunRecorder.ps1`: starts the bridge and offline recorder together.
- [`PS1_COMMANDS.md`](PS1_COMMANDS.md): complete PowerShell launcher CLI and examples.
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
|   |-- buffers.py
|   `-- model_paths.py
|-- Training/
|   |-- DQN_server.py
|   |-- PPO_server.py
|   |-- PPO_lstm_server.py
|   `-- ppo_metrics.py
|-- Inference/
|   |-- dqn_eval.py
|   |-- ppo_eval.py
|   `-- ppo_lstm_eval.py
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
|   |-- test_environment.py
|   |-- test_model_paths.py
|   |-- test_ppo_lstm_eval.py
|   |-- test_ppo_lstm_server.py
|   |-- test_ppo_server.py
|   `-- test_ppo_eval.py
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

`Custom_enviornments/Test_Env/Env_16.py` is the active environment used by all
three training servers and both evaluators.

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

The launcher reads `Automation/automation_config.yaml`, starts TensorBoard when
enabled, starts the bridge, waits for its readiness signal, and then starts the
command selected by `rl_algorithm`. All three trainers remain available:

```yaml
rl_algorithm: "ppo_lstm" # use "ppo" for feed-forward PPO or "dqn" for DQN

# Optional: initialize PPO/PPO-LSTM from an existing checkpoint.
restore_model_path: null
# restore_model_path: runs/Env16__PPO_lstm_server__1__<timestamp>/PPO_lstm_server.pt

dqn_command:
  - python
  - -m
  - Training.DQN_server

ppo_command:
  - python
  - -m
  - Training.PPO_server

ppo_lstm_command:
  - python
  - -m
  - Training.PPO_lstm_server
```

Configured commands are YAML argument lists and use the active Python
interpreter. Training and inference selection are independent:
`rl_algorithm` controls `RunRL.ps1`, while `inference_algorithm` controls
`RunInference.ps1`.

Both PowerShell launchers stream output to the terminal and save combined
standard output and errors as readable UTF-8 text files:

```text
logs/training_YYYYMMDD_HHMMSS_mmm.txt
logs/inference_YYYYMMDD_HHMMSS_mmm.txt
```

Each log includes the command, config path, start and finish times, complete
child-process output, final exit code, and its own absolute path. Failed runs
raise a PowerShell error that points to the saved log. Use
`-LogDirectory <path>` to store logs somewhere else.

Manual startup is still supported.

Start the WebSocket/ZMQ bridge first:

```bash
python -m Automation.Bridge.ws_zmq_bridge
```

Then start PPO training:

```bash
python -m Training.PPO_server
```

Or start recurrent PPO training:

```bash
python -m Training.PPO_lstm_server
```

The game must be running and sending `ai_tick` messages through the browser extension before the env can step.

## Recording Player Demonstrations

The offline recorder replaces the training/evaluation backend on the same ZMQ
endpoint. The recommended one-command startup is:

```powershell
.\RunRecorder.ps1
```

Or run the Python automation entry point directly:

```bash
python -m pip install -r Offline/requirements.txt
python -m Automation.record
```

Both commands start `Automation.Bridge.ws_zmq_bridge`, wait for it to become
ready, then start `Offline.record_player`. Exiting the recorder also stops the
bridge. Recording output is saved under `logs/recording_*.txt` when the
PowerShell wrapper is used. Manual two-terminal startup remains available for
debugging.

Enter a recording name at the prompt (or press Enter for the generated name).
Recording starts immediately and continues until `Ctrl+C`. There is no command
shell or background server: the main loop directly performs the same sequence
as `Env16.step`:

```text
socket.recv_json()
world_state = message.get("worldState", {})
capture current player input
socket.send_json({"move": "NoOp", ...})
insert the state/action row into SQLite
```

Every saved tick prints its SQLite frame ID, captured input, complete
`worldState`, and exact response. The default database is
`Offline/recordings.sqlite3` and can be changed with `--database PATH`.

Each `frames` row contains the untouched payload, extracted `worldState`, exact
reply, and a global Windows input snapshot. `input_json` stores the keys/buttons
held at that tick plus down/up events since the preceding recorded tick. This
captures held actions and transitions observed between state snapshots.
Virtual-key numbers are retained alongside readable names so they can be mapped
to environment actions later.

Use `--no-keyboard` to collect state-only data for manual annotation, or
`--no-op-action VALUE` if the JS no-op value changes. SQLite support comes from
Python's standard-library `sqlite3` module, so it requires no separate pip
package. Only one backend can bind port 5555, so do not run a trainer/evaluator
at the same time as the recorder.

To skip the name prompt, configure or override the recorder command with
`--session-name`, for example:

```powershell
.\RunRecorder.ps1 -Command "python -m Offline.record_player --session-name demo-1"
```

### AWAC training

Train AWAC on every episode of the local BC-Minari dataset (the same
`env16/BC-v0` ID used by behavior cloning):

```powershell
.\RunOfflineRL.ps1 -Algorithm AWAC -DatasetId "env16/BC-v0"
.\RL_venv\Scripts\python.exe -m tensorboard.main --logdir runs
```

The default performs 1,000,000 offline updates without connecting to the game.
Use `-UpdateSteps` to change this count and `-Device cpu` to select CPU.
Each run writes TensorBoard losses, `config.yaml`, and `AWAC_model.pt` under
`runs/AWAC-BC-v0-<id>/`. The checkpoint includes both critics, the categorical
actor, optimizers, observation normalization, and the BC action ordering.
AWAC checkpoints require the AWAC actor; the BC evaluator cannot load them.

For live fine-tuning, add `-OnlineIterations 100000` to the training command
and connect the game. The launcher supervises the WebSocket bridge and starts
TensorBoard when enabled in the automation config. After offline training,
AWAC creates one `Env16` using the recorded BC action order
(`up, left, right, down, ...`) and logs online episode returns and lengths.
Only terminations stop critic bootstrapping; time-limit truncations reset the
game episode while retaining the next-state value target.

### Minari behavior-cloning recorder

`Offline.record_minari` records the same live player interaction directly as a
Minari dataset using `Env16BC`. Start it together with the WebSocket bridge:

```powershell
.\RunRecorder.ps1 -Command "python -m Offline.record_minari --dataset-id env16/BC-v0 --max-steps 500"
```

The dataset ID must include a Minari version suffix. Dataset IDs cannot be
overwritten, so use `env16/BC-v1`, `env16/BC-v2`, and so on for later runs.
Minari saves under `~/.minari/datasets` by default; pass `--datasets-path PATH`
to select another root.

The BC recorder sends `NoOp` to the game on every tick. It records only one
unambiguous mapped action: `W=up(0)`, `A=left(1)`, `D=right(2)`, `S=down(3)`,
`SPACE=attack(4)`, and spells `1,2,3,5,6,7` as indices `5..10`. The matching
numpad keys (`NUMPAD1`, `NUMPAD2`, `NUMPAD3`, `NUMPAD5`, `NUMPAD6`, and
`NUMPAD7`) use the same spell indices. Unmapped ticks and ambiguous
multi-action ticks are acknowledged but never passed to Minari.
The recorder buffers one labeled frame so each stored action is aligned with
the observation from which the player chose it.

Player HP reaching zero or leaving combat map `53` terminates the current
episode as a loss. The existing maximum-step and map-position truncation rules
still apply. A terminal/truncating world-state tick completes the preceding
valid action even if no new mapped key is held, so a released keyboard cannot
hide a death or boundary state.

Every terminated or truncated episode is written to the local Minari dataset
before the environment resets. Later episodes are appended to that dataset.
`Ctrl+C` also flushes a partial episode with at least one complete transition;
Minari marks its last transition truncated. The launcher gives this final save
up to 30 seconds before forcing process cleanup.

Recorder output is limited to the captured key, parsed state, termination and
truncation flags, plus dataset-save confirmation:

```text
key=W state=[...] terminated=False truncated=False
dataset_saved=True reason=truncated id=env16/BC-v0 episodes=1 transitions=256 path=...
```

Inspect a completed dataset and its episode fields with:

```powershell
python -c "import minari; d=minari.load_dataset('env16/BC-v0'); e=next(d.iterate_episodes()); print(d.total_episodes, d.total_steps); print(e.observations.shape, e.actions.shape, e.rewards.shape, e.terminations.shape, e.truncations.shape, e.infos.keys())"
```

This Minari dataset uses Minari's HDF5 storage. It is separate from the raw
SQLite capture produced by `Offline.record_player`.

### Any-percent behavior-cloning workflow

`RunOfflineRL.ps1` replaces the old hard-coded
`Offline/run_inference_from_checkpoint.py` and
`Offline/run_live_evaluation.py` scripts. Training and both evaluation paths
now use the same automation entry point:

| Mode | Bridge | Result |
| --- | --- | --- |
| `Train` | No | Trains from Minari and writes one final `BC_model.pt`. |
| `Dataset` | No | Reports MSE/accuracy and writes `predicted_actions.csv`. |
| `Live` | Yes | Runs the BC model against the live `Env16` game stream. |

Live evaluation translates BC action indices into the different `Env16`
movement ordering before sending actions to the game.

Evaluate the checkpoint against recorded transitions:

```powershell
.\RunOfflineRL.ps1 -Mode Dataset `
  -CheckpointPath "runs\bc-BC-v0-example\BC_model.pt" `
  -DatasetId "env16/BC-v0"
```

Run five live episodes:

```powershell
.\RunOfflineRL.ps1 -Mode Live `
  -CheckpointPath "runs\bc-BC-v0-example\BC_model.pt" `
  -DatasetId "env16/BC-v0" `
  -EvalEpisodes 5
```

Use the same `-TopFraction`, `-Gamma`, and normalization setting used during
training. Pass `-NoNormalizeState` when the checkpoint was trained without
state normalization. `-Device` accepts `auto`, `cpu`, or `cuda`. Logs are
written under `logs` by default.

### Any-percent behavior-cloning training

Train the BC model from every episode in the local Minari dataset:

```powershell
.\RunOfflineRL.ps1 -Mode Train `
  -DatasetId "env16/BC-v0" `
  -TopFraction 1.0 `
  -CheckpointsPath "runs"
```

For a shorter first run:

```powershell
.\RunOfflineRL.ps1 -Mode Train `
  -DatasetId "env16/BC-v0" `
  -UpdateSteps 10000 `
  -BatchSize 256 `
  -TopFraction 1.0 `
  -CheckpointsPath "runs"
```

Training creates a uniquely named `runs/bc-BC-v0-<id>` directory containing
TensorBoard events, `config.yaml`, and one final `BC_model.pt`. It does not
write intermediate PyTorch models at evaluation intervals. `RunOfflineRL.ps1`
defaults to `-TopFraction 1.0` so all recorded episodes are used. Offline
training does not start the game bridge. Use `-Mode Live` afterward for live
evaluation. See the complete parameter and command reference in
[`PS1_COMMANDS.md`](PS1_COMMANDS.md).

Both PPO trainers use one live `Env16` instance directly because the external
simulator owns a single ZMQ request stream. They do not use `SyncVectorEnv` or
`RecordEpisodeStatistics`; each trainer handles episode resets and batching.
Their shared TensorBoard environment dashboard is implemented in
`Training/ppo_metrics.py`.

## PPO Training

PPO collects `num_steps` sequential transitions from the external simulator
before each optimization cycle. With the defaults, 128 game ticks form one
rollout batch, which is divided into four minibatches of 32. `num_steps` is the
rollout horizon; it does not create additional environments.

| Parameter | Default |
|---|---:|
| `total_timesteps` | 20,000 |
| `learning_rate` | 0.00025 |
| `num_envs` | 1 |
| `num_steps` | 128 |
| `num_minibatches` | 4 |
| `update_epochs` | 4 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_coef` | 0.2 |
| `ent_coef` | 0.01 |
| `vf_coef` | 0.5 |
| `metrics_frequency` | 1 |
| `save_model` | `true` |
| `model_path` | `None` (the current `runs/<run_name>/PPO_server.pt`) |
| `restore_model_path` | `None` (start with newly initialized weights) |

Example:

```bash
python -m Training.PPO_server \
  --total-timesteps 20000 \
  --num-steps 128 \
  --num-minibatches 4 \
  --metrics-frequency 10
```

After every PPO rollout/update cycle, the current actor and critic state is
written to `runs/<run_name>/PPO_server.pt`, beside that run's TensorBoard event
files. Passing `--model-path` overrides this location.
Passing `--restore-model-path runs/<run_name>/PPO_server.pt` initializes the
new run from an existing checkpoint without changing where the new checkpoint
is saved.

## Recurrent PPO (LSTM) Training

The recurrent trainer collects one sequential `Env16` rollout and divides it
into contiguous sequences for optimization. Each sequence starts with the
hidden and cell state recorded at that point in the rollout, so LSTM context is
preserved even though `Env16` supports only one live environment. Episode
boundaries reset the recurrent state through the rollout's done mask.

| Parameter | Default |
|---|---:|
| `total_timesteps` | 20,000 |
| `learning_rate` | 0.00025 |
| `num_envs` | 1 |
| `num_steps` | 128 |
| `num_minibatches` | 4 |
| `update_epochs` | 4 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_coef` | 0.2 |
| `metrics_frequency` | 1 |
| `save_model` | `true` |
| `restore_model_path` | `None` (start with newly initialized weights) |
| `model_path` | `None` (the current `runs/<run_name>/PPO_lstm_server.pt`) |

`num_steps` must be divisible by `num_minibatches`. For example:

```bash
python -m Training.PPO_lstm_server \
  --total-timesteps 20000 \
  --num-steps 128 \
  --num-minibatches 4 \
  --metrics-frequency 10
```

Use `--restore-model-path runs/<run_name>/PPO_lstm_server.pt` to initialize a
new recurrent run from an existing checkpoint. As with feed-forward PPO, the
restored run writes its updates to a new run-local checkpoint by default.

The LSTM trainer emits the same step, episode, reward-component, environment,
loss, and throughput charts as `PPO_server.py`, plus
`lstm/hidden_state_norm` and `lstm/cell_state_norm`. Like `PPO_server.py`, it
overwrites `runs/<run_name>/PPO_lstm_server.pt` after every rollout/update
cycle unless `--model-path` provides an explicit override.

## Legacy DQN Training

Hyperparameters are concrete `Args` values and are not automatically changed
when `total_timesteps` changes. The defaults are currently configured for a
20,000-step run using the proportions from the supplied CleanRL CartPole
configuration:

| Parameter | Default |
|---|---:|
| `total_timesteps` | 20,000 |
| `buffer_size` | 400 |
| `target_network_frequency` | 20 |
| `batch_size` | 5 |
| `learning_starts` | 400 |
| `train_frequency` | 1 |
| `metrics_frequency` | 1 |

Every value can be configured directly for training or fine-tuning:

```bash
python -m Training.DQN_server \
  --total-timesteps 20000 \
  --buffer-size 2000 \
  --batch-size 32 \
  --learning-starts 0 \
  --train-frequency 2 \
  --target-network-frequency 50 \
  --metrics-frequency 25
```

Configuration checks only issue warnings. They never rewrite a supplied value
or stop the run. A warning is emitted when a count parameter is below half or
above twice the proportional CleanRL recommendation for the selected
`total_timesteps`, when batch size exceeds replay-buffer size, or when learning
would start after the run ends.

Configured values are printed once at startup and stored in TensorBoard's
`hyperparameters` text entry.

## Monitoring

Training logs are written under `runs/`. The automation launcher starts
TensorBoard by default; when training manually, start it with:

```bash
python -m Automation.tensorboard_server --logdir runs
```

Open http://localhost:6006.

The project launcher suppresses only TensorBoard's known `pkg_resources`
deprecation and optional-TensorFlow notices. TensorFlow is not required for
the scalar, histogram, text, and hyperparameter dashboards used here. Other
warnings and errors still appear in the terminal and PowerShell text logs.

Useful metrics:

- `charts/step_reward`
- `charts/episodic_return`
- `charts/episode_length`
- `charts/win_rate`
- `charts/learning_rate`
- `charts/SPS`
- `losses/policy_loss`
- `losses/value_loss`
- `losses/entropy`
- `losses/old_approx_kl`
- `losses/approx_kl`
- `losses/clipfrac`
- `losses/explained_variance`
- `lstm/hidden_state_norm` (recurrent PPO only)
- `lstm/cell_state_norm` (recurrent PPO only)
- `environment/player_hp`
- `environment/enemy_hp`
- `environment/action_frequency/*`
- `rewards/damage_dealt`
- `rewards/damage_taken`
- `rewards/terminal`
- `rewards/health_state`
- `rewards/positioning`
- `rewards/episode_*`

Step and environment metrics are sampled according to the configured
`metrics_frequency` (1 step by default) and always written for
non-zero reward events and episode endings. Set `--metrics-frequency`
explicitly when a different logging interval is preferred.
Episode return, length, win rate, and episode reward-component totals are
written when an episode ends.

### Metric meanings

| Metric | Meaning |
|---|---|
| `charts/step_reward` | Total reward returned for the sampled environment step. |
| `charts/episodic_return` | Sum of rewards over the completed episode. |
| `charts/episode_length` | Number of environment steps in the completed episode. |
| `charts/win_rate` | Cumulative percentage (0–100) of completed episodes classified as wins. |
| `charts/learning_rate` | Current PPO optimizer learning rate after optional annealing. |
| `charts/SPS` | External simulator steps processed per second. |
| `losses/policy_loss` | PPO clipped policy-objective loss. |
| `losses/value_loss` | PPO critic/value-function loss. |
| `losses/entropy` | Entropy of the policy action distribution. |
| `losses/approx_kl` | Approximate KL divergence between the old and updated policy. |
| `losses/clipfrac` | Fraction of minibatch samples whose policy ratio was clipped. |
| `losses/explained_variance` | How much variance in PPO returns is explained by the critic. |
| `environment/player_hp` | Current player HP percentage from observation index `3`. |
| `environment/enemy_hp` | Nearest-enemy HP percentage from observation index `8`. |

`environment/action_frequency` is a histogram of actions taken since the last
metric write. Tags below `environment/action_frequency/*` show each action's
cumulative fraction, such as `environment/action_frequency/attack` and
`environment/action_frequency/castSpell_1`.

### Reward components

The environment records signed reward components in `info["reward_components"]`.
Their sum is exactly the reward returned to PPO or DQN:

| Component | Meaning |
|---|---|
| `damage_dealt` | Positive reward for reducing matched enemy HP by entity ID, including nearby enemies that disappear from the next world state. |
| `damage_taken` | Negative reward when the player's HP decreases. |
| `terminal` | Kill bonus or loss penalty. |
| `health_state` | Penalty for remaining at low health. |
| `positioning` | Position-based penalty from the current environment rules. |

The corresponding `rewards/episode_*` metrics accumulate each component across
the complete episode. Console output is intentionally limited to episode
summaries and checkpoint-save messages; raw world-state details use debug
logging instead of per-step printing.

## Evaluation

Recurrent PPO training saves each model under its matching TensorBoard run:

```text
runs/Env16__PPO_lstm_server__<seed>__<timestamp>/PPO_lstm_server.pt
```

The evaluator automatically selects the newest matching run checkpoint, so the
direct `Env16` command is:

```bash
python -m Inference.ppo_lstm_eval
```

It resets the LSTM hidden and cell state at every `Env16` episode boundary. To
sample actions or change the run length:

```bash
python -m Inference.ppo_lstm_eval --no-deterministic
python -m Inference.ppo_lstm_eval --eval-episodes 20
python -m Inference.ppo_lstm_eval --no-cuda
```

Feed-forward PPO uses the equivalent run-local path:

```text
runs/Env16__PPO_server__<seed>__<timestamp>/PPO_server.pt
```

Evaluate the newest checkpoint deterministically with:

```bash
python -m Inference.ppo_eval
```

To sample from the learned policy instead of selecting its highest-probability
action:

```bash
python -m Inference.ppo_eval --no-deterministic
```

Other feed-forward PPO examples:

```bash
python -m Inference.ppo_eval --eval-episodes 20
python -m Inference.ppo_eval --no-cuda
```

DQN checkpoints are evaluated separately:

```bash
python -m Inference.dqn_eval --model-path runs/DQN_server__<timestamp>/DQN_server.pt
```

Optional examples:

```bash
python -m Inference.dqn_eval --model-path runs/DQN_server__<timestamp>/DQN_server.pt --eval-episodes 20
python -m Inference.dqn_eval --model-path runs/DQN_server__<timestamp>/DQN_server.pt --epsilon 0.05
python -m Inference.dqn_eval --model-path runs/DQN_server__<timestamp>/DQN_server.pt --cuda false
```

All three evaluators use the direct external `Env16` flow and report episode
returns, outcomes, wins, win rate, and elapsed time. The bridge and game must
be running and producing `ai_tick` messages.

### Automated inference

The default Automation configuration evaluates recurrent PPO:

```yaml
inference_algorithm: "ppo_lstm" # "ppo" and "dqn" are also supported

dqn_inference_command:
  - python
  - -m
  - Inference.dqn_eval
  - --model-path
  - runs/DQN_server__1783138095/DQN_server.pt

ppo_inference_command:
  - python
  - -m
  - Inference.ppo_eval
  - --model-path
  - runs/XXXX/PPO_server.pt

ppo_lstm_inference_command:
  - python
  - -m
  - Inference.ppo_lstm_eval
  - --model-path
  - runs/XXXX/PPO_lstm_server.pt
```

Automation uses the explicit checkpoint paths above, so replace `XXXX` with the
run directory you want to evaluate. Direct PPO evaluator commands still select
the newest matching checkpoint when `--model-path` is omitted. Start the
configured evaluator with:

```powershell
.\RunInference.ps1
```

You can still override the configured evaluator for one run:

```powershell
.\RunInference.ps1 -Command "python -m Inference.ppo_lstm_eval --model-path runs/Env16__PPO_lstm_server__1__<timestamp>/PPO_lstm_server.pt"
```

## Notes

- Recurrent PPO is the default training path for the current discrete action space.
- Run Python entry points with `python -m ...`; package imports no longer depend on script-relative `sys.path` changes.
- `Load_env_config.py` should remain the single place that parses shared env config.
- `Automation/automation_config.yaml` controls TensorBoard startup, bridge startup, training command, and inference command.
