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

Provides one interface for BC or AWAC training and BC dataset/live evaluation.
BC is the default algorithm. Offline training and dataset evaluation run without
the bridge. BC live evaluation and AWAC training with positive
`-OnlineIterations` start and supervise the bridge.

### Syntax

```powershell
.\RunOfflineRL.ps1 `
  [-Mode <Train|Dataset|Live>] `
  [-Algorithm <BC|AWAC>] `
  [-CheckpointPath <string>] `
  [-DatasetId <string>] `
  [-EvalEpisodes <int>] `
  [-UpdateSteps <int>] `
  [-OnlineIterations <int>] `
  [-HiddenDim <int>] `
  [-LearningRate <double>] `
  [-Tau <double>] `
  [-AwacLambda <double>] `
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
| `-Algorithm` | `BC` | All | `BC` or `AWAC`; AWAC supports Train mode only. |
| `-CheckpointPath` | Empty | Dataset, Live | Required path to `BC_model.pt` for evaluation. |
| `-DatasetId` | `env16/BC-v0` | All | Local Minari dataset ID. |
| `-EvalEpisodes` | `5` | Live | Number of live evaluation episodes. |
| `-UpdateSteps` | `1000000` | Train | Offline gradient updates; maps to AWAC's `--offline-iterations`. |
| `-OnlineIterations` | `0` | AWAC Train | Live fine-tuning updates after offline training; positive values enable the bridge. |
| `-HiddenDim` | `256` | AWAC Train | Actor and critic hidden-layer size. |
| `-LearningRate` | `0.0003` | AWAC Train | Actor and critic optimizer learning rate. |
| `-Tau` | `0.005` | AWAC Train | Target-critic soft-update coefficient. |
| `-AwacLambda` | `1.0` | AWAC Train | Advantage-weight temperature. |
| `-BufferSize` | `2000000` | Train | Maximum replay-buffer transitions. |
| `-BatchSize` | `256` | Train | Training batch size. |
| `-EvalEvery` | `5000` | BC Train | Reserved evaluation interval; does not save models. |
| `-TopFraction` | `1.0` | BC | Highest-return fraction of episodes; AWAC requires `1.0` and uses all episodes. |
| `-Gamma` | `0.99` | All | BC episode-ranking discount or AWAC critic discount. |
| `-Device` | `auto` | AWAC Train, Dataset, Live | Training/evaluation device; auto selects CUDA when available. BC training selects its own device. |
| `-NoNormalizeState` | Off | All | Disables observation normalization. |
| `-CheckpointsPath` | `runs` | Train | Root directory for generated run folders. |
| `-OutputCsv` | Empty | Dataset | Prediction CSV path; defaults beside the model. |
| `-Config` | `Automation\automation_config.yaml` | BC Live, AWAC online training | Bridge and optional TensorBoard server configuration. |
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

### Train an AWAC model

Train on the same local BC-Minari dataset:

```powershell
.\RunOfflineRL.ps1 -Algorithm AWAC -Mode Train `
  -DatasetId "env16/BC-v0" `
  -UpdateSteps 1000000 `
  -BatchSize 256 `
  -Device auto `
  -CheckpointsPath "runs"
```

Short CPU run:

```powershell
.\RunOfflineRL.ps1 -Algorithm AWAC -UpdateSteps 100 -Device cpu
```

Train offline, then fine-tune against the live game:

```powershell
.\RunOfflineRL.ps1 -Algorithm AWAC -Mode Train `
  -UpdateSteps 100000 -OnlineIterations 100000 `
  -AwacLambda 1.0 -LearningRate 0.0003
```

With positive `-OnlineIterations`, automation starts the bridge before the
trainer, starts TensorBoard if enabled in `-Config`, and supervises their
shutdown. The game must be connected for the online phase to progress.

AWAC saves `AWAC_model.pt`, `config.yaml`, and TensorBoard events under a unique
`runs/AWAC-BC-v0-<id>/` directory. Offline-only runs write TensorBoard events but
do not start its server. To view them, run:

```powershell
.\RL_venv\Scripts\python.exe -m Automation.tensorboard_server --logdir runs
```

`Dataset` and `Live` modes evaluate BC checkpoints only. AWAC checkpoints
include a categorical actor and critics and cannot use the BC evaluator.

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
