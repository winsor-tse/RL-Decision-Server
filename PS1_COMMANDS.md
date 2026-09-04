# PowerShell CLI Reference

Run these commands from the repository root in PowerShell. Each launcher uses
`RL_venv\Scripts\python.exe` when it exists and otherwise falls back to the
`python` command available on `PATH`.

## Optional environment activation

The launchers do not require manual activation, but the environment can be
activated for direct Python commands:

```powershell
.\RL_venv\Scripts\Activate.ps1
```

## `RunRL.ps1`

Starts the configured online RL training algorithm and WebSocket bridge, plus
TensorBoard when it is enabled in the YAML configuration. The algorithm is
selected by `rl_algorithm`.

### Syntax

```powershell
.\RunRL.ps1 [-Config <string>] [-LogDirectory <string>]
```

| Parameter | Default | Description |
| --- | --- | --- |
| `-Config` | `Automation\automation_config.yaml` | Automation YAML file. |
| `-LogDirectory` | `logs` | Directory for timestamped training logs. |

### Examples

Use the default configuration:

```powershell
.\RunRL.ps1
```

Use another automation configuration and log directory:

```powershell
.\RunRL.ps1 `
  -Config "Automation\my_training_config.yaml" `
  -LogDirectory "logs\training"
```

The configuration selects one of these command keys:

```yaml
rl_algorithm: ppo
# dqn_command, ppo_command, or ppo_lstm_command
```

## `RunInference.ps1`

Starts the WebSocket bridge and a live DQN, PPO, or PPO-LSTM evaluator. Without
an override, `inference_algorithm` and its corresponding command are read from
the automation YAML file.

### Syntax

```powershell
.\RunInference.ps1 `
  [-Config <string>] `
  [-Command <string>] `
  [-LogDirectory <string>]
```

| Parameter | Default | Description |
| --- | --- | --- |
| `-Config` | `Automation\automation_config.yaml` | Automation YAML file. |
| `-Command` | Empty | Optional evaluator command override. |
| `-LogDirectory` | `logs` | Directory for timestamped inference logs. |

### Examples

Use the configured inference algorithm:

```powershell
.\RunInference.ps1
```

Evaluate a specific PPO checkpoint:

```powershell
.\RunInference.ps1 `
  -Command "python -m Inference.ppo_eval --model-path runs/my_ppo/PPO_server.pt --eval-episodes 10"
```

Evaluate a specific PPO-LSTM checkpoint:

```powershell
.\RunInference.ps1 `
  -Command "python -m Inference.ppo_lstm_eval --model-path runs/my_lstm/PPO_lstm_server.pt --eval-episodes 10"
```

Evaluate a DQN checkpoint with epsilon exploration:

```powershell
.\RunInference.ps1 `
  -Command "python -m Inference.dqn_eval --model-path runs/my_dqn/DQN_server.pt --eval-episodes 10 --epsilon 0.05"
```

## `RunRecorder.ps1`

Starts the WebSocket bridge and an offline recorder. The default recorder saves
raw world-state and keyboard data to SQLite. `-Command` can select the Minari
behavior-cloning recorder instead.

### Syntax

```powershell
.\RunRecorder.ps1 `
  [-Config <string>] `
  [-Command <string>] `
  [-LogDirectory <string>]
```

| Parameter | Default | Description |
| --- | --- | --- |
| `-Config` | `Automation\automation_config.yaml` | Automation YAML file. |
| `-Command` | Empty | Optional recorder command override. |
| `-LogDirectory` | `logs` | Directory for timestamped recording logs. |

### Examples

Start the configured raw SQLite recorder:

```powershell
.\RunRecorder.ps1
```

Record a named raw SQLite session without prompting:

```powershell
.\RunRecorder.ps1 `
  -Command "python -m Offline.record_player --session-name demo-1"
```

Record valid player actions into a local Minari dataset:

```powershell
.\RunRecorder.ps1 `
  -Command "python -m Offline.record_minari --dataset-id env16/BC-v0 --max-steps 500"
```

Minari dataset IDs are versioned and cannot be overwritten. Use a new suffix,
such as `env16/BC-v1`, for another dataset.

## `RunOfflineRL.ps1`

Provides one interface for behavior-cloning training, dataset evaluation, and
live evaluation. `Train` and `Dataset` modes do not start the bridge. `Live`
mode starts and supervises the bridge.

### Syntax

```powershell
.\RunOfflineRL.ps1 `
  [-Mode <Train|Dataset|Live>] `
  [-CheckpointPath <string>] `
  [-DatasetId <string>] `
  [-EvalEpisodes <int>] `
  [-UpdateSteps <int>] `
  [-BufferSize <int>] `
  [-BatchSize <int>] `
  [-EvalEvery <int>] `
  [-TopFraction <double>] `
  [-Gamma <double>] `
  [-Device <auto|cpu|cuda>] `
  [-NoNormalizeState] `
  [-CheckpointsPath <string>] `
  [-OutputCsv <string>] `
  [-Config <string>] `
  [-LogDirectory <string>]
```

| Parameter | Default | Modes | Description |
| --- | --- | --- | --- |
| `-Mode` | `Train` | All | Selects training, dataset evaluation, or live evaluation. |
| `-CheckpointPath` | Empty | Dataset, Live | Required path to `BC_model.pt` for evaluation. |
| `-DatasetId` | `env16/BC-v0` | All | Local Minari dataset ID. |
| `-EvalEpisodes` | `5` | Live | Number of live evaluation episodes. |
| `-UpdateSteps` | `1000000` | Train | Number of gradient updates. |
| `-BufferSize` | `2000000` | Train | Maximum replay-buffer transitions. |
| `-BatchSize` | `256` | Train | Training batch size. |
| `-EvalEvery` | `5000` | Train | Reserved evaluation interval; does not save models. |
| `-TopFraction` | `1.0` | All | Highest-return fraction of episodes to use. |
| `-Gamma` | `0.99` | All | Discount used to rank episode returns. |
| `-Device` | `auto` | Dataset, Live | Evaluation device. |
| `-NoNormalizeState` | Off | All | Disables observation normalization. |
| `-CheckpointsPath` | `runs` | Train | Root directory for generated run folders. |
| `-OutputCsv` | Empty | Dataset | Prediction CSV path; defaults beside the model. |
| `-Config` | `Automation\automation_config.yaml` | Live | Bridge configuration. |
| `-LogDirectory` | `logs` | All | Directory for timestamped offline-RL logs. |

### Train a BC model

Train on every episode in the dataset:

```powershell
.\RunOfflineRL.ps1 -Mode Train `
  -DatasetId "env16/BC-v0" `
  -TopFraction 1.0 `
  -UpdateSteps 1000000 `
  -BatchSize 256 `
  -EvalEvery 5000 `
  -CheckpointsPath "runs"
```

Short training run:

```powershell
.\RunOfflineRL.ps1 -Mode Train `
  -DatasetId "env16/BC-v0" `
  -UpdateSteps 10000 `
  -EvalEvery 1000
```

Training creates a unique directory such as `runs/bc-BC-v0-a1b2c3d4` with:

```text
BC_model.pt
config.yaml
events.out.tfevents...
```

Only `BC_model.pt` is written after training completes. BC training does not
save intermediate `.pt` models at evaluation intervals.

### Evaluate against the dataset

```powershell
.\RunOfflineRL.ps1 -Mode Dataset `
  -DatasetId "env16/BC-v0" `
  -CheckpointPath "runs\bc-BC-v0-a1b2c3d4\BC_model.pt"
```

Choose the prediction CSV path:

```powershell
.\RunOfflineRL.ps1 -Mode Dataset `
  -DatasetId "env16/BC-v0" `
  -CheckpointPath "runs\bc-BC-v0-a1b2c3d4\BC_model.pt" `
  -OutputCsv "reports\bc_predictions.csv"
```

### Evaluate against the live game

```powershell
.\RunOfflineRL.ps1 -Mode Live `
  -DatasetId "env16/BC-v0" `
  -CheckpointPath "runs\bc-BC-v0-a1b2c3d4\BC_model.pt" `
  -EvalEpisodes 5 `
  -Device auto
```

Use the same `-TopFraction`, `-Gamma`, and normalization setting for evaluation
that were used for training. If training used `-NoNormalizeState`, pass it
again during dataset or live evaluation.

## Logs and stopping

- All launchers create timestamped logs below `logs` unless `-LogDirectory` is
  changed.
- Use `Ctrl+C` to stop a running launcher.
- Recorder shutdown allows time to flush SQLite or Minari data before child
  processes are forcibly cleaned up.
