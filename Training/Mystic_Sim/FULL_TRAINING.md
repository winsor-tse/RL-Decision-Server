# Standalone 1v80 LSTM PPO training

This entry point starts fresh or continues a checkpoint on the original map53
full scenario without a JSON config. One player faces 80 Innies (two per
original spawn area). The goal remains five kills without dying, with the
existing 1,024-step cap, Y-boundary rules, rewards, observations and combat.

## Continue a checkpoint for additional transitions

```powershell
.\Train_Mystic_Full.bat --continue-from "runs/path/to/checkpoints/checkpoint_08.pt" --total-timesteps 1000000 --device cuda
```

`--total-timesteps` is the **additional** budget when using `--continue-from`,
rounded up to a complete rollout. This restores weights, optimizer moments and
training RNG using the existing continuation implementation. Simulator episodes
and LSTM episode memory start fresh. PPO settings, including learning rate,
seed and number of environments, default to the checkpoint's saved values.
To explicitly change the learning rate, add `--learning-rate 0.0001`.

Each continuation creates a new timestamped directory under
`runs/mystic_sim/full`; optionally specify `--run-dir runs/mystic_sim/full/my_run`.
It saves ten checkpoints, evaluates both modes, and records GIFs from CP5 by
default. Add `--validate-only` to check the checkpoint, settings and additional
budget without training or writing run files.

`--resume` instead finishes the original saved budget and cannot be combined
with `--continue-from`. The continuation budget must be explicitly supplied.
Existing compatibility rules still apply: LR, budget, seeds and evaluation
count may change, but architecture and PPO batching settings must match.
Annealed runs remain subject to their saved LR horizon; this full launcher
defaults to a constant learning rate for fresh runs. Use the curriculum runner
for curriculum checkpoints and scenario changes.

## Setup for a new checkout

From the repository root, use a Python 3.11+ environment supported by PyTorch:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Install a CPU or CUDA build of PyTorch compatible with your machine using the
[official PyTorch installer](https://pytorch.org/get-started/locally/), then:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-mystic-sim.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The focused requirements file avoids the live-server/cloud/BC dependency stack
and the existing full requirements file's machine-specific CUDA build pin. Its
version ranges are compatibility bounds, not a lockfile. Record your environment
with `python -m pip freeze` when sharing experiment results. An existing working
RL_venv is also supported; no package reinstall is required for it.

## Run

With your environment activated, from the repository root:

```powershell
python -m Training.Mystic_Sim.train_full --device cuda
```

Windows shortcut, which also works when invoked from another directory:

```powershell
.\Train_Mystic_Full.bat --device cuda
```

The batch file uses the active virtual environment, then `.venv`, then `RL_venv`.
It changes to its own repository directory and forwards all CLI flags. On Linux
or macOS use the module command from the repository root with your environment's
Python. Use `--device cpu` without CUDA, or omit it for automatic selection.

Preflight only, without training, recording or creating output files:

```powershell
python -m Training.Mystic_Sim.train_full --validate-only --device cuda
```

Default settings:

- Fresh LSTM PPO with linear actor head; seed 42.
- Three million requested transitions, combined across four environments.
- 256 rollout steps per environment; four minibatches; four PPO epochs.
- Constant learning rate 0.00025, gamma 0.999, learning-only reward scale 0.01.
- Exactly ten checkpoints, each with sampled and greedy evaluation on seeds
  20000-20029. Full-scenario reports reuse these evaluations rather than running
  the same episodes twice.
- GIFs for the first evaluation seed begin at checkpoint 5, in both modes.
- TensorBoard and JSON diagnostics use the existing training pipeline.

With the default rollout size, complete PPO batches round the budget to
3,000,320 transitions. Episode truncation remains 1,024 decisions; rollout length
is not episode length. Independent worlds are stepped synchronously on CPU;
CUDA accelerates the neural network, not the simulator. More environments do not
guarantee higher throughput.

All settings can be overridden without a config:

```powershell
python -m Training.Mystic_Sim.train_full --device cuda --seed 123 --num-envs 8 --total-timesteps 3000000 --eval-episodes 100 --eval-seed 41000 --run-dir runs/mystic_sim/full/seed123
python -m Training.Mystic_Sim.train_full --help
```

Choose disjoint training/evaluation seed banks. Default run directories are
unique under `runs/mystic_sim/full/` and printed before training. Relative
`--run-dir` and `--resume` paths resolve against this checkout, not a developer's
home or the caller's directory; absolute paths are accepted. Existing nonempty
directories are not overwritten. Map data is bundled in the repository. There
is no dependency on the C# source directory, live ZMQ, previous runs, or JSON
curriculum configs. `run.json` saves the resolved hyperparameters and contract.

## Throughput optimization and benchmarking

Recurrent training now executes whole contiguous time spans in a single LSTM
call, splitting at episode starts to preserve hidden-state resets and gradient
barriers. It still shuffles only contiguous time blocks, never independent
transitions. Final-state bootstrap evaluates only the critic; the unused actor
lookahead has been removed. Weights, optimizer structure, observations, rewards,
simulator mechanics, and checkpoint contracts are unchanged. Existing checkpoints
remain loadable. Different GPU kernel accumulation can change numerical rounding
and subsequent sampled trajectories; bit-identical training replay is not promised.

Benchmark on your machine without optimizer updates, checkpoints, or training:

```powershell
python -m Training.Mystic_Sim.benchmark --device cuda --compare-reference --steps 512 --iterations 20
python -m Training.Mystic_Sim.benchmark --device cpu --compare-reference
python -m Training.Mystic_Sim.benchmark --device cuda --num-envs 8
```

The reference is the previous one-timestep-at-a-time recurrent implementation.
Both paths execute the same scripted actions and reset seed. Environment SPS
excludes policy inference; rollout SPS includes inference and value lookahead.
The recurrent forward/backward microbenchmark uses 64 steps, four environments
by default, and two reset boundaries. It computes gradients without optimizer
updates. Measurements exclude startup/evaluation/GIFs and are not end-to-end PPO
training SPS.

One local CUDA comparison at four environments measured recurrent forward/backward
71.6 ms -> 6.64 ms and rollout 964 -> 1,037 SPS. Shorter measurements varied; use
TensorBoard `charts/SPS_training` and `charts/SPS_wall` for your actual experiment.
New `charts/SPS_rollout` and `charts/optimization_seconds` separate collection from
update time. Existing wall SPS also includes evaluation/recording overhead.

Defaults retain four environments. Increasing `--num-envs` can amortize inference
cost but also increases rollout batch size at fixed `--num-steps`, changing PPO's
update frequency. Benchmark first and treat this as an experiment setting, not
a free speed improvement. Worlds remain synchronous; no subprocess backend or
changes to combat timing were introduced. `--no-record-gameplay` and a smaller
`--eval-episodes` reduce wall-clock overhead but also reduce saved diagnostics.

## Resume and monitoring commands

```powershell
python -m Training.Mystic_Sim.train_full --resume runs/mystic_sim/full/YOUR_RUN/checkpoints/checkpoint_05.pt --device cuda
python -m tensorboard.main --logdir runs/mystic_sim/full --port 6006
```

Open http://localhost:6006. Resume restores saved training hyperparameters,
optimizer, model and counters, and finishes the original budget. It starts fresh
episodes, not exact world snapshots. Use the latest checkpoint; later existing
checkpoints are protected. Curriculum checkpoints cannot be resumed through this
full-scenario launcher. Runtime options such as device and recording remain
selectable. `--no-record-gameplay` disables GIFs; `--record-from-checkpoint 1`
enables early recordings; `--stop-after-checkpoint 5` provides a clean stopping
point. Training remains simulation-only.

Local validation covered the full-map reset, CLI overrides, repository-relative
paths, checkpoint selection guards and a two-step GIF smoke test. No three-million
transition experiment or clean-machine dependency installation was run as part of
adding this launcher.
