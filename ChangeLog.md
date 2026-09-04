# Change Log

## BC-IL branch

### 2026-09-03 - Numpad behavior-cloning actions

- Added stable `NUMPAD0` through `NUMPAD9` names to Windows input capture.
- Mapped `NUMPAD1`, `NUMPAD2`, `NUMPAD3`, `NUMPAD5`, `NUMPAD6`, and `NUMPAD7`
  to BC spell action indices `5` through `10`, matching the top-row digits.
- Treats top-row and numpad aliases pressed together as one action when both
  map to the same spell, while different simultaneous actions remain invalid.

### 2026-09-03 - Behavior-cloning training entry point

- Replaced the unavailable `pyrallis` command-line dependency in
  `Offline/any_percent_bc.py` with the project's installed `tyro` dependency.
- BC training now defaults to a unique run directory below `runs`, writes its
  effective configuration and TensorBoard events there, and does not attempt
  to open a live game environment during offline training.
- Extended `Automation/offline_rl.py` and `RunOfflineRL.ps1` with `Train`,
  `Dataset`, and `Live` modes. Training and dataset evaluation run without the
  bridge; live evaluation starts the supervised bridge.
- Renamed the final PyTorch artifact from `final_checkpoint.pt` to
  `BC_model.pt`. Periodic checkpoints retain the `checkpoint_<step>.pt` name.
- Added documented full-dataset and shorter training commands.

### 2026-09-03 - Streamlined offline checkpoint evaluation

- Replaced the hard-coded `Offline/run_inference_from_checkpoint.py` and
  `Offline/run_live_evaluation.py` scripts with
  `Inference/any_percent_bc_eval.py`.
- Added configurable dataset and live evaluation modes. Dataset mode reports
  action MSE and discrete-action accuracy and exports a prediction CSV. Live
  mode evaluates directly against `Env16`.
- Corrected live behavior-cloning action translation: the BC dataset orders
  movement as up, left, right, down, while `Env16` orders it as up, down, left,
  right.
- Added `Automation/offline_rl.py` to run dataset evaluation without the bridge
  and live evaluation with the supervised WebSocket bridge.
- Added `RunOfflineRL.ps1` as the shared PowerShell entry point with explicit
  checkpoint, dataset, episode, trajectory-selection, normalization, device,
  output, config, and log options.
- Added synthetic checkpoint, dataset preparation, action conversion, CSV, and
  automation-routing tests.

### 2026-08-09 through 2026-08-29 - Minari and offline-learning foundations

This branch adds the first end-to-end path for capturing human Yugen Saga
demonstrations and storing them as an offline reinforcement-learning dataset.
It also adds initial behavior-cloning, AWAC, and Rainbow DQN experiments.

#### Human demonstration recording

- Added `Offline/record_player.py`, which captures Windows keyboard input and
  game world-state messages while returning `NoOp` to the game. Raw frames can
  be stored separately in SQLite for diagnostics and manual inspection.
- Added `RunRecorder.ps1` and `Automation/record.py` so the WebSocket-to-ZeroMQ
  bridge and a selected recorder can be started and stopped together.
- Added recorder lifecycle handling, bridge diagnostics, and end-to-end tests.
  After `Ctrl+C`, the launcher now gives the recorder up to 30 seconds to flush
  pending SQLite or Minari data before forcing cleanup.

Relevant history: `9a03252`, `21f18f5`, `55a9872`, `1712072`, `b209c42`, and
`f264eb2`.

#### Minari behavior-cloning dataset

- Added Minari `0.5.3` to the project dependencies.
- Added `Env16BC`, a demonstration-specific version of `Env16` that preserves
  the existing observation, reward, termination, truncation, and info logic.
- The game always receives `NoOp`; captured keys become offline action labels:

  | Key | Action | Index |
  | --- | --- | ---: |
  | `W` | up | 0 |
  | `A` | left | 1 |
  | `D` | right | 2 |
  | `S` | down | 3 |
  | `Space` | attack | 4 |
  | `1`, `2`, `3` | cast spell 1, 2, 3 | 5, 6, 7 |
  | `5`, `6`, `7` | cast spell 5, 6, 7 | 8, 9, 10 |

- Invalid, released, or ambiguous multi-key input is acknowledged but is not
  added to the dataset and does not advance the Gym environment.
- Player HP reaching zero or leaving combat map `53` terminates an episode as a
  loss. The existing maximum-step and map-position truncation rules still
  apply. A keyless death or truncation tick completes the preceding valid
  action so the episode boundary is not lost after the player releases a key.
- Added `Offline/record_minari.py`, which records all Minari episode fields:
  observations, actions, rewards, terminations, truncations, and infos.
- A completed episode is saved before reset. The first episode creates the
  local dataset and later episodes are appended. `Ctrl+C` or the configured
  step limit saves a partial episode containing at least one complete
  transition, with its final transition marked as truncated.
- Recorder output is limited to the captured key, parsed state, termination and
  truncation flags, and dataset-save confirmation.
- Added `Tests/test_minari_BC.py` for sampling local episodes and printing each
  transition's observation, reward, action, termination, and truncation values.
- Added Minari, environment-boundary, action-mapping, persistence, and graceful
  interruption tests.

Record a dataset with:

```powershell
.\RunRecorder.ps1 `
  -Command "python -m Offline.record_minari --dataset-id env16/BC-v0 --max-steps 500"
```

Inspect sampled transitions with:

```powershell
.\RL_venv\Scripts\python.exe Tests\test_minari_BC.py --dataset-id env16/BC-v0 --episodes 5
```

Minari datasets use local HDF5 storage under `~/.minari/datasets` by default;
they are separate from the raw SQLite recorder. Dataset IDs are versioned and
cannot be overwritten, so later recordings should use `env16/BC-v1`,
`env16/BC-v2`, and so on. A verified example contains one 100-transition
episode under `env16/BC-v0`.

Relevant history: `3231786`, `5eeaa6d`, `06103ef`, and `48e744a`.

#### Behavior cloning prototype

- Added `Offline/any_percent_bc.py`, derived from the CORL behavior-cloning
  implementation.
- Loads a local Minari dataset, ranks episodes by discounted return, selects a
  configurable top fraction, converts episodes into transition batches, and
  supports state normalization, replay-buffer sampling, evaluation, checkpoint
  writing, and Weights & Biases metrics.
- This remains a prototype. Its actor and environment setup still assume a
  continuous `Box` action space (`shape`, `high`, and `tanh` output), while
  `Env16BC` uses discrete action indices. Evaluation also attempts to recover a
  live environment from recorded data. These parts must be adapted before the
  trainer can be used reliably with the Yugen Saga dataset.

Relevant history: `a41afb9` and `665415d`.

#### Offline RL and advanced RL prototypes

- Added `Offline/awac.py`, an Advantage-Weighted Actor-Critic reference with an
  offline replay-buffer phase, optional online fine-tuning, twin critics,
  advantage-weighted actor updates, normalization, evaluation, checkpoints,
  and Weights & Biases logging.
- AWAC currently targets Gym/D4RL continuous-control datasets and is not yet
  connected to `Env16BC` or the local Minari dataset. It imports `d4rl`, which
  is not currently pinned in `requirements.txt`.
- Added `Training/Rainbow_DQN.py` with a noisy dueling distributional network,
  prioritized replay, and target-network training. Rainbow DQN is an advanced
  online RL experiment rather than part of the Minari behavior-cloning path.
- Moved imitation/offline-learning files from `Imatation_Learning` into the
  `Offline` directory and added initial hierarchical-RL design notes.

Relevant history: `a913774`, `d953f2d`, `acfc78e`, and `573674d`.

Minari reference: <https://minari.farama.org/main/content/basic_usage/>
