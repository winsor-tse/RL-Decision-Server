# Standalone Mystic Sim PPO

## Recurrent PPO (LSTM)

Use `--recurrent` for the simulator adaptation of `Training/PPO_lstm_server.py`.
New recurrent runs default to `--actor-head linear`: the actor's 128-unit LSTM
feeds seven logits directly. Use `--actor-head tanh` for the previous two-layer
Tanh head. The critic, ordered batching, 50 inputs, rewards and environment rules
are unchanged. Head type is checkpointed; start a fresh run to change it.

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.train --recurrent --actor-head linear --device cuda --seed 42 --num-envs 4 --num-steps 256 --num-minibatches 4 --total-timesteps 200000 --run-dir runs/mystic_sim/lstm-linear-200k
```

Older seven-action recurrent checkpoints lacking `actor_head` load as **tanh**,
including on resume. Resume restores their original head, not the new default.
For fresh matched-seed comparisons, critic and actor embedding/LSTM initialization
are identical between head choices. Initialization order was adjusted for this;
fresh Tanh runs are not bit-for-bit reproductions of the older initialization.
Feed-forward PPO remains the default. Old eight-action checkpoints are incompatible. Recurrent and MLP weights are not interchangeable; start a fresh run.

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.train --recurrent --device cuda --num-envs 4 --num-steps 256 --num-minibatches 4 --total-timesteps 2000000 --run-dir runs/mystic_sim/lstm_first

# Optional early stop at the second of ten checkpoints (~400k transitions).
# Add --stop-after-checkpoint 2 to the command above, then continue with:
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.train --resume runs/mystic_sim/lstm_first/checkpoints/checkpoint_02.pt --device cuda

.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.evaluate runs/mystic_sim/lstm_first/checkpoints/checkpoint_10.pt --episodes 30 --seed 20000 --stochastic --record runs/mystic_sim/lstm_first/evaluation.gif
```

CPU, TensorBoard, ten checkpoints, per-checkpoint evaluation/GIF recording, and
resume work through the same commands as MLP PPO. Checkpoints identify the
architecture and history schema; evaluation automatically selects the model.
Training still truncates episodes at 1,024 decisions. A 256-step rollout is
not an episode limit. The final timestep budget rounds up to a full rollout.

**Sequence batching:** buffers remain `[time, environment, feature]`. As in the
reference server, shuffle contiguous time blocks, not individual transitions.
Every minibatch includes all environments over one contiguous time block, with
the saved hidden/cell state from before its first observation. `num_steps` must
divide evenly by `num_minibatches`, with at least two timesteps per block.
The command above uses four 64-step sequences, each containing four worlds
(256 transitions per minibatch). Sequence order can be shuffled; time order
inside each sequence and world is preserved. Gradients are truncated at block
boundaries, but inference memory persists across blocks and rollout updates.
Saved boundary states are detached rollout-policy states, as in the server;
there is no burn-in or full-episode recurrent-state recomputation after updates.

**Observable history:** the Gym environment retains its 26-value contract.
The recurrent trainer/evaluator append 24 features, giving a separately versioned
50-value policy input (`mystic-local-terrain-50-v3`), implemented in `history.py`.
Start a fresh recurrent run: older 46/51-input recurrent and eight-action checkpoints are rejected.

| Appended indices | Meaning |
|---|---|
| 26–32 | Previous chosen action, one-hot (all zero on reset) |
| 33–35 | Previous action applied, movement blocked, cooldown rejected |
| 36–38 | Time since each confirmed spell cast, capped at 10 seconds and divided by 10 seconds |
| 39–41 | Whether each spell has been successfully cast this episode |
| 42–43 | Up: normalized free-tile distance, detected flag |
| 44–45 | Down: normalized free-tile distance, detected flag |
| 46–47 | Left: normalized free-tile distance, detected flag |
| 48–49 | Right: normalized free-tile distance, detected flag |

History uses past action outcomes; terrain sensors use the known static collision
map. No exact ready timestamps, full collision grid, or hidden monster IDs enter
the policy. Cast ages
are measured at the resulting observation (a successful cast is already 200 ms
old after one simulator decision). Rejected casts do not restart these ages.
The remembered absolute blocked-origin fields have been removed. Unknown
cast ages must be interpreted using the corresponding seen flag. All history
and recurrent state reset per episode, independently for each world. Sensors are
recomputed at the current observation's position, including reset observations
and final observations used for truncation bootstrapping.

Sensors scan cardinal offsets 1 through 4 and stop at the first static blocked
tile or map boundary. Distance is **free tiles before the blocker / 4**: an
adjacent wall gives `(0, 1)`, a blocker four tiles away gives `(0.75, 1)`, and
no blocker within range gives `(1, 0)`. The second value is the detection flag.
Rays use world directions (up = negative Y), not player facing, and do not see
around corners or off the ray. Moving entities are excluded; previous-action
and blocked-movement feedback still report failed attempts into either terrain
or entities. The range and semantics are included in checkpoint metadata.

These relative sensors transfer across translated local geometry. The original
26 inputs still include absolute player X/Y and map ID, and training solely on
map53 can still overfit to it; this change does not claim map generalization.

**Deliberate differences/fixes relative to the server reference:**

- Generalized contiguous sequence batching to multiple independent worlds.
- Retained its ReLU embedding → 512 features → 128-unit LSTM structure with
  separate actor/critic embeddings and LSTMs. The critic keeps two 64-unit
  Tanh layers; the default actor now projects the LSTM directly to seven logits.
  This costs more compute and preserves independent gradient clipping; a shared
  LSTM would couple actor and critic gradients again.
- Truncations bootstrap from the final observation with pre-reset memory;
  deaths do not bootstrap. GAE stops at either kind of episode boundary.
  Bootstrap lookahead does not advance the stored policy memory a second time.
- Retained Mystic's reward scale 0.01, gamma 0.999, separate optional value
  clipping, entropy coefficient, value coefficient, and independent gradient
  norm limits. These are not silently replaced by live-server defaults.
- Evaluation maintains memory for the whole episode and uses local sampling
  RNG. Resume restores weights/optimizer/RNG but starts fresh worlds and zero
  history/memory; it does not restore a mismatched old hidden state into a reset
  world or claim exact interrupted-trajectory replay.

**Required future live-server work (Phase 8; not implemented here):** port this
50-value history/sensor contract and model architecture to live training/inference.
Load the live map's authoritative static collision layer and its dimensions,
then compute the same four rays at the player's current tile on each observation.
Static terrain stays fixed in map coordinates; distances relative to the player
change as the player moves. Use the same collision semantics, four-tile cutoff,
world direction order, free-tile normalization and boundary handling. Exclude
NPC/player occupancy from terrain rays. Recompute after movement, teleport and
respawn; reload the collision map on map change. Test translated layouts, adjacent
walls, range-edge blockers, boundaries and no-hit rays against simulator fixtures.
Map seven simulator outputs to live labels explicitly: up/down/left/right and spells 1/2/3. Do not reuse old eight-output indices.
Maintain hidden/cell and history separately per player/session; reset on death,
respawn/new episode, reconnect, or map/task reset. Map live action order explicitly.
Use authoritative action acknowledgements for successful casts, collision and
cooldown rejection; sending a command is not confirmation. Measure elapsed time
from server timestamps and align outcomes with the command that caused them.
Until these payload fields are available and validated, this policy is not live
compatible. Test simulator/live replay for feature values, action timing,
episode resets, hidden states and logits. Do not fall back to treating missing
acknowledgements as successful casts or all-zero history. Neither existing
`PPO_lstm_server.py` nor its live inference script has been modified.

This trainer uses the current `MysticSimEnv` and its full training rules:
seven actions, 26 observation values, map53 terrain, 1,024 decisions, five-kill
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
2. Runs both greedy and sampled evaluation on the same fixed seeds
   (`--eval-seed 10000 --eval-episodes 3`).
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

The base 26-value MLP observation omits cooldowns, relative terrain and collision history.
This feed-forward policy can overfit map53 and cannot directly observe all timing
state. Successful optimizer updates do not prove combat learning or live parity.
Use held-out evaluation, damage/kills/survival and invalid-action metrics to assess
learning; reward magnitude alone is insufficient. The recurrent path above adds
bounded terrain sensing and action history through its separate 50-value contract.


## Seven-action change and next actor experiment

The policy action IDs are 0 up, 1 down, 2 left, 3 right, 4 Arcane Blast,
5 Acid Cloud, 6 Tempest Inferno. Attack was removed, not renamed to wait.
`advance_idle()` is only for the manual viewer and isolated simulator timing
tests; PPO cannot select it. The recurrent input loses the attack-history bit,
so it is now 50 values. Start a fresh run with the new action/observation schemas.

Step 1 (linear head and diagnostics) is implemented. Curriculum and reward
changes remain deferred. The comparison plan is:

1. Keep the 128-unit LSTM and the healthy critic initially. Saturated Tanh heads
   indicate poor activation/optimization behavior, not evidence of insufficient
   memory capacity. A wider head can saturate too.
2. Inspect the new activation/sensitivity metrics, per-action failure counts,
   and both greedy/sampled evaluations; a healthy critic or improved raw return
   is not an actor exit gate.
3. Compare the default `LSTM(128) -> Linear(128, 7)` actor head against the optional
   two 64-unit Tanh head, using fresh seeds and the same seven-action environment.
   Keep the embedding/LSTM/critic and rewards fixed to isolate the head change.
4. If needed, separately test lower actor learning rate (e.g. 1e-4 versus 2.5e-4),
   while preserving critic learning rate and separate gradient limits. Do not
   bundle wider layers, reward changes and learning-rate changes into one test.
5. Use 50k/100k/200k checks, multiple seeds and fixed held-out evaluation worlds.
   Advance only when actor probabilities respond to observations and combat
   improves (kills/wins, damage, survival, legal casts), not merely movement.
   Consider 256 LSTM units only if a stable actor still shows a memory limitation.
   Easier encounters/curriculum remain a separate subsequent experiment.

Removing attack does not repair already saturated weights or guarantee useful
behavior: illegal casts can still become ineffective stationary actions. No
arbitrary movement reward or unconditional inactivity penalty was added.

## Actor diagnostics and paired evaluation

Each checkpoint now saves `checkpoint_NN.json` (greedy) and
`checkpoint_NN_sampled.json`, plus matching greedy and `_sampled.gif` gameplay
recordings when recording is enabled. Ten model checkpoints produce twenty
evaluation reports and up to twenty GIFs. Evaluation takes more wall time but
does not alter training worlds or training RNG. Existing `evaluation/*` scalar
tags retain greedy results; explicit `evaluation/greedy/*` and
`evaluation/sampled/*` tags identify the two modes.

Recurrent TensorBoard metrics under `losses/` include:

- `actor_embedding_grad_norm`, `actor_lstm_grad_norm`, `actor_head_grad_norm`:
  gradient norms before independent actor/critic clipping.
- `actor_probability_mean/<action>` and `actor_probability_std/<action>`:
  preferences and variation across the rollout replayed in temporal order with
  saved initial memory and episode masks, after optimization.
- `actor_logit_std_mean`, `actor_feature_std_mean` and
  `actor_probability_std_mean`: aggregate variation across those same states.
- `actor_lstm_output_saturation` for the linear head, or
  `actor_head_tanh_saturation` for Tanh: fraction with absolute activation >0.99.
  They measure different layers; the linear head has no Tanh-head saturation.

Training also reports `environment/successful_move_rate` and per-action
`environment/failure_rate/<action>/<reason>` when failures occur. Each evaluation
JSON's `diagnostics` contains action fractions, probability mean/std, successful
moves, per-action failure counts, and spell probabilities grouped by cooldown
ready/blocked states, with observation counts. These cooldown labels come from
the engine **only for measurement**; they do not enter policy inputs or mask
actions. Ready here means slot/family/global timers allow casting, not that MP,
HP or targeting requirements are satisfied.

Compare conditional ready/blocked probabilities together with cast failures and
combat outcomes. Probability variation is descriptive, not a causal test that
the policy understands cooldowns. Different histories/encounters can affect the
statistics; neither high variation nor movement alone proves a better policy.
