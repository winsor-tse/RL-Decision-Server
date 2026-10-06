# Standalone Mystic Sim PPO

This trainer uses the current `MysticSimEnv` and its full training rules:
eight actions, 26 observation values, map53 terrain, 1,024 decisions, five-kill
win, player-death loss, and the configured damage/kill/collision/cooldown/Y
rewards. It does not connect to ZMQ or import a live PPO server/automation.

Run commands from the repository root with the project's Python environment.
The existing requirements already include PyTorch, Gymnasium, TensorBoard,
NumPy, pygame-ce and Pillow; recording needs the last two packages. No FFmpeg
or additional video package is needed.

## Train

```powershell
# Default: one environment, auto-select CUDA when available, ten checkpoints/GIFs.
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.train

# Explicit CUDA, four independent worlds, and a chosen output directory.
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.train --device cuda --num-envs 4 --total-timesteps 262144 --run-dir runs/mystic_sim/my_run

# CPU also works; disable Pygame recording for a fully headless dependency path.
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.train --device cpu --num-envs 1 --no-record-gameplay
```

`--device auto` is the default. Explicit `--device cuda` fails with a clear error
if CUDA is unavailable; it does not silently switch devices. Simulator math
always runs on CPU; CUDA handles policy inference and PPO updates. A small MLP
can be slower on CUDA than CPU, especially with one environment.

`--num-envs N` batches N separate environments, each with its own RNG, occupancy,
enemies and scheduler. This first implementation steps them synchronously in
one process, then batches network inference. It supports multiple worlds but
does **not** claim parallel CPU simulation or Phase 7 throughput optimization.
Initial seeds are `seed + environment_index`; subsequent episode resets continue
each world's independent RNG stream.

Defaults follow the reference PPO server: separate two-layer 128-unit Tanh
actor/critic networks, Adam, clipped policy loss, GAE, normalized
advantages, entropy bonus, and annealed learning rate. Critic repair defaults
are reward scale 0.01, gamma 0.999, no value clipping, and independent actor/critic
gradient clipping at 0.5 each.
An optional KL threshold stops update epochs early. `--help` lists settings.
The policy divides each observation feature by its declared Box upper bound;
this fixed transform is checkpointed. Raw rewards remain unchanged in the
environment, evaluation, episode returns and dashboard components. PPO multiplies
rewards by `--reward-scale 0.01` only before calculating advantages and value
targets. Critic predictions/bootstrap values therefore use scaled units; they
are not multiplied a second time. No action mask hides illegal casts.

Value clipping is disabled by default. Set `--value-clip-coef 0.2` to test a
separate clip in **scaled value units**, independently of `--clip-coef 0.2`, which
controls the dimensionless policy ratio. `--actor-max-grad-norm 0.5` and
`--critic-max-grad-norm 0.5` replace the old joint `--max-grad-norm` setting.
The simulator cooldown-attempt penalty is now -1 (previously -5); damage, kill,
win/loss, collision and Y rewards are otherwise unchanged. Nothing rewards a
spell-button press without damage. The 26-value observation and feed-forward
architecture remain unchanged; LSTM work is deferred.

**Start a fresh run after these changes.** Checkpoints now use
`mystic_sim_ppo_v2` and record the scale, gamma, optional value clip, and separate
gradient limits. v1 checkpoints have unscaled critics and incompatible simulator
metadata and cannot resume/evaluate through this version. Existing artifacts
remain intact; use their original code revision for historical evaluation.

## Ten checkpoints, evaluation, and gameplay

Each completed run saves exactly **ten** distinct checkpoints at approximately
10%, 20%, ... 100% of its PPO updates. A rollout is
`num_steps * num_envs` transitions. The budget rounds up to a full rollout; the
manifest records requested and actual totals. At least ten updates are required
so checkpoints represent distinct optimizer states. For example, with 128 steps
and four environments, request at least 5,120 timesteps.

At each checkpoint the trainer:

1. Atomically saves model, optimizer, progress, RNG state, and compatibility metadata.
2. Runs greedy evaluation on fixed seeds (`--eval-seed 10000 --eval-episodes 3`).
3. Records the first evaluation episode through the existing Pygame viewer as a
   660x430 animated GIF, with HP/MP, cooldowns, steps, kills, rewards and end reason.
4. Saves evaluation JSON and TensorBoard metrics, including comparison with the
   random-action baseline evaluated on those same seeds at the start of the run.

Recording uses Pygame's hidden SDL renderer; it does not open an interactive
window or slow training to real-time playback. The default `--record-stride 5`
captures one frame per simulated second plus the initial/final frames. Use
`--record-stride 1` for every 200ms decision, at greater recording cost and memory.
GIF timing follows simulation time, with a one-second hold on the final frame.
An episode may end early due to death, five kills, or a Y boundary. Recording
failures are printed and stored in evaluation JSON; the model is already saved
and training can continue. `--no-record-gameplay` disables recording explicitly.

Evaluation runs in separate worlds with fixed seeds and a local action RNG. It
does not advance training worlds or consume training RNG. Greedy evaluation may
look different from the stochastic actions used for training; standalone
evaluation also supports `--stochastic`.

Artifacts are under `runs/mystic_sim/<timestamp_seed_id>/` unless `--run-dir` is given:

```text
run.json                         environment/configuration/contract/version metadata
invocation_after_update_*.json    per-invocation settings and resume origin
random_baseline.json              fixed-seed random-policy comparison
tensorboard/                     TensorBoard events
checkpoints/checkpoint_01.pt      through checkpoint_10.pt; complete training state
evaluation/checkpoint_01.json    through checkpoint_10.json
gameplay/checkpoint_01.gif        through checkpoint_10.gif (when recording succeeds)
progress.json                    last evaluated checkpoint, losses, SPS, counters
```

## TensorBoard

```powershell
.\RL_venv\Scripts\python.exe -m tensorboard.main --logdir runs/mystic_sim
```

Open the address TensorBoard prints. Charts include PPO losses, entropy, KL,
clipping, value explained variance, parameter changes, learning rate, train-only
and wall-clock SPS, returns, episode lengths, kills/win rate, survival, HP/MP,
damage dealt/taken, action frequencies, invalid casts, collisions, cooldown
rejections, and evaluation versus random. The six reward component names remain
`health_state`, `positioning`, `damage_taken`, `damage_dealt`, `terminal`, `killed`.
Training SPS excludes checkpoint evaluation/recording; wall SPS includes them.
`charts/mean_step_reward` stays raw; `charts/mean_scaled_step_reward` shows the
learning scale. `losses/actor_grad_norm` and `losses/critic_grad_norm` report norms
before independent clipping. Critic diagnostics include `critic_value_std`,
`critic_target_std`, `critic_hidden_saturation` (fraction of second-layer Tanh
activations with magnitude above 0.99), and `post_update_explained_variance`.
Value loss and standard deviations are in scaled learning units, so compare
against old raw-unit losses carefully.

## Resume or evaluate

```powershell
# Optional clean stop at a scheduled checkpoint.
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.train --run-dir runs/mystic_sim/resumable --stop-after-checkpoint 3

# Continue the original budget and remaining checkpoints. CPU/CUDA may be selected.
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.train --resume runs/mystic_sim/resumable/checkpoints/checkpoint_03.pt --device cuda

# Evaluate on held-out seeds; optionally record another episode.
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.evaluate runs/mystic_sim/my_run/checkpoints/checkpoint_10.pt --episodes 5 --seed 20000 --record runs/mystic_sim/my_run/heldout.gif --output runs/mystic_sim/my_run/heldout.json
```

Resume restores the saved PPO hyperparameters (including total budget and
environment count), weights, optimizer, update/step/episode counters, learning-rate
schedule position, and Python/NumPy/Torch/CUDA RNG states. CLI device and recording
options remain invocation controls. Resume starts **fresh simulator episodes**
using `seed + completed_updates * num_envs + environment_index`; unfinished episode
statistics are discarded. It does not restore world/scheduler state or claim
identical next transitions. Cross-device floating-point results may differ.

Resume from the latest checkpoint to finish a stopped run. If later checkpoints
already exist, choose an empty `--run-dir` to branch without overwriting them.
A finished checkpoint can be evaluated but cannot resume beyond its saved budget.
Run new experiments in new directories. Full uninterrupted runs produce ten
checkpoints; stopped runs and branches contain only checkpoints reached there.

Checkpoints reject mismatched action/observation schemas, preprocessing,
environment settings, map/mechanics hashes, and simulator source hashes.
Live/legacy model files are intentionally not accepted. No exact world resume,
recurrent PPO, subprocess vector optimization, BC/Minari, or sim-to-live adapter
is implemented by this phase.

## Validation and interpretation

```powershell
.\RL_venv\Scripts\python.exe -m unittest Tests.test_mystic_sim_ppo -v
```

Regression tests cover terminal versus truncated GAE, pre-reset observation
bootstrapping, independent worlds, real parameter updates, ten checkpoint files,
optimizer/schedule/RNG resume, metadata rejection, TensorBoard tags, evaluation
RNG isolation, and actual GIF creation. See [VALIDATION.md](VALIDATION.md) for
the measured short CPU/CUDA runs and the bounded two-seed comparison.

The current observation omits cooldowns, relative terrain and collision history.
This feed-forward policy can overfit map53 and cannot directly observe all timing
state. Successful optimizer updates do not prove combat learning or live parity.
Use held-out evaluation, damage/kills/survival and invalid-action metrics to assess
learning; reward magnitude alone is insufficient. Recurrent inputs or additional
observations require their own contract change and validation.
