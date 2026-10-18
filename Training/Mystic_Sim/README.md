# Standalone Mystic Sim PPO

Curriculum Phases 0–3 are documented in
[CURRICULUM_PLAN.md](CURRICULUM_PLAN.md). Use the
[manual run config](configs/curriculum_manual.json) to train one selected encounter,
save it, and explicitly continue to another encounter with weights and optimizer
retained. Automatic stage progression is deferred to Phase 7 or later.

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.curriculum_run --config Training/Mystic_Sim/configs/curriculum_manual.json --validate-only
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.curriculum_run --config Training/Mystic_Sim/configs/curriculum_manual.json
```

## Recurrent PPO (LSTM)

For the compact-spawn experiment, use `configs/three_innies_tight.json` with
`curriculum_run`. It changes initial path distances to 3–8 tiles and retains
the same checkpoint-8 start. New training and evaluation runs log range access,
damage gaps, revisits and movement reversals. See the curriculum plan for
definitions and the evaluation-only `compare_encounters` command.

To continue the completed one-Innie run into three Innies using checkpoint 8:

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.curriculum_run --config Training/Mystic_Sim/configs/three_innies_from_one.json
```

This config inherits shared PPO settings and preserves both LSTMs and the optimizer.
Add `--validate-only` to check without training, or `--list-stages` to see available
encounters. See the curriculum plan for a fresh comparison and later-stage commands.

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

**Base observation: 56-value contract (`mystic-ten-enemies-relative-five-56-v2`).**
Both `reset()` and `step()` return a `float32` array with shape `(56,)`.
The existing first 26 values retain their meaning. Five more legacy-format
slots extend coverage to ten enemies; signed offsets describe the same nearest
five enemies without duplicating their HP/MP fields. This is one combined
observation, not two selectable experiment modes.

| Indices (zero-based) | Value | Raw bounds on map53 |
|---|---|---|
| 0-1 | Absolute player X, Y | 0-99 |
| 2 | Player facing | 0-4 |
| 3-4 | Player HP, MP fractions | 0-1 |
| 5 | Map ID | 0-53; emits 53 |
| 6-45 | Ten enemies, four values each: Manhattan distance, direction, HP fraction, MP fraction | 0-198, 0-4, 0-1, 0-1 |
| 46-47 | Nearest enemy: signed dx, dy | -99 to 99 each |
| 48-49 | Second enemy: signed dx, dy | -99 to 99 each |
| 50-51 | Third enemy: signed dx, dy | -99 to 99 each |
| 52-53 | Fourth enemy: signed dx, dy | -99 to 99 each |
| 54-55 | Fifth enemy: signed dx, dy | -99 to 99 each |

`dx = enemy.x - player.x`, `dy = enemy.y - player.y`: positive X is right,
positive Y is down. The policy divides offsets by 99, preserving their sign
and mapping them to [-1, 1]. Other features retain division by their Box upper
bound. Both feed-forward PPO and recurrent PPO consume this combined contract.

Direction codes: **up=0, down=1, left=2, right=3, idle=4**. Enemy direction
selects the dominant offset axis (Y wins ties), not the enemy's facing.
Living monsters are sorted by Manhattan distance, then entity ID. Slots rerank
on every observation; they are not persistent identities. There is no visibility
or casting-range filter. Spell targeting remains separate and unchanged.
Each missing four-value slot and corresponding dx/dy pair is zero-padded;
there is no separate presence flag. Read its HP slot to distinguish padding.
HP/MP fractions are clamped to [0, 1]; zero maximum gives zero fraction.

**Observable history:** the recurrent trainer/evaluator append 24 features,
giving an 80-value policy input
(`mystic-ten-enemies-relative-five-local-terrain-80-v4`).

| Appended indices | Meaning |
|---|---|
| 56-62 | Previous chosen action, one-hot (all zero on reset) |
| 63-65 | Previous action applied, movement blocked, cooldown rejected |
| 66-68 | Time since confirmed spell cast, capped at 10 seconds / 10 seconds |
| 69-71 | Whether each spell has been successfully cast this episode |
| 72-73 | Up: normalized free-tile distance, detected flag |
| 74-75 | Down: normalized free-tile distance, detected flag |
| 76-77 | Left: normalized free-tile distance, detected flag |
| 78-79 | Right: normalized free-tile distance, detected flag |

**Checkpoint compatibility:** start a fresh run. Previous 26-input feed-forward
and 50-input recurrent checkpoints cannot load into this contract. No automatic
weight/optimizer migration is provided; preserve their original code revision
for playback. Increasing observation coverage does not guarantee better results.

**Live ZMQ follow-up:** the live parser has not been changed. Before deploying
these policies, encode the same ten sorted living entities and five signed
coordinate pairs, with the same padding, normalization, and history. Confirm
the payload exposes those entities and coordinates; do not invent unseen data.
The legacy live 26-value parser is not compatible with these new policies.

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
56 inputs still include absolute player X/Y and map ID, and training solely on
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
80-value history/sensor contract and model architecture to live training/inference.
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
seven actions, 56 observation values, map53 terrain, 1,024 decisions, five-kill
win, player-death loss, and the configured damage/kill/collision/cooldown/Y
rewards. It does not connect to ZMQ or import a live PPO server/automation.

Run commands from the repository root with the project's Python environment.
The existing requirements already include PyTorch, Gymnasium, TensorBoard,
NumPy, pygame-ce and Pillow; recording needs the last two packages. No FFmpeg
or additional video package is needed.

## Benchmark CUDA SPS

Run from the repository root to benchmark the recurrent policy and simulator:

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.benchmark --device cuda --num-envs 4 --steps 256 --iterations 10 --torch-threads 1
```

This uses scripted actions and performs **no optimizer updates**, saves no
checkpoints, and does not start a training run. CUDA must be available in your
PyTorch installation; the simulator itself still runs on CPU.

The JSON output reports:

- `environment_sps`: simulator transitions per second without policy inference.
- `optimized.rollout_sps`: transitions per second including recurrent policy
  inference and bootstrap-value computation.
- `optimized.recurrent_forward_backward_ms`: average recurrent forward/backward
  time for a 64-step sequence batch; lower is better.

These are component benchmarks, not end-to-end PPO training SPS. `--steps 256`
with `--num-envs 4` measures 1,024 transitions per rollout benchmark. Change
`--num-envs` to compare batch sizes, use `--device cpu` for a CPU comparison,
or add `--compare-reference` to compare the optimized and reference recurrent
implementations. Avoid running training simultaneously when measuring speed.

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
spell-button press without damage. The combined 56-value observation is shared by feed-forward and recurrent PPO.

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

The base 56-value MLP observation omits cooldowns, relative terrain and collision history.
This feed-forward policy can overfit map53 and cannot directly observe all timing
state. Successful optimizer updates do not prove combat learning or live parity.
Use held-out evaluation, damage/kills/survival and invalid-action metrics to assess
learning; reward magnitude alone is insufficient. The recurrent path above adds
bounded terrain sensing and action history through its separate 80-value contract.


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

## Compare frozen policies across stages

Run from the repository root (evaluation only; no training):

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.compare --config Training/Mystic_Sim/configs/compare_4v1_5v1.json
```

The example compares 4v1 checkpoint 10 and 5v1 checkpoint 10 on
`ten_innies_goal5` (10 enemies, five total kills, including respawn kills) and
`full` (the original scenario). Both sampled (temperature 1) and greedy modes
use the same episode seeds. LSTM state resets per episode; weights stay frozen.
Full episode metrics and diagnostics are saved to the configured JSON output.

Edit `policies` to add named checkpoint paths, and `stages` to choose stage names
from `comparison_stages.json`. That file inherits the existing manual curriculum
stages and adds tight 4/5/10-enemy encounters. You may instead point `stage_config`
to any existing curriculum run config. The stage-config path is relative to the
comparison config; checkpoint and output paths are relative to your working directory.
Training initialization and budget settings in stage configs are ignored.

Use `--validate-only` to check all checkpoint/stage contracts and reset geometry
without episode rollouts. Override `--episodes 100`, `--seed 23000`, or `--output`
for another evaluation bank. Output is updated after each pairing, so interrupted
comparisons may contain only some results. Reusing an output path replaces it.
Only encounter settings and kill goals may differ; incompatible engine/reward/
observation contracts are rejected. No best checkpoint is automatically promoted.

The training checkpoint CLI now prints explicitly labeled **stage sampled** and
**stage greedy** return, win rate, kills, and episode length. Curriculum runs also
print both **full** scenario modes. Sampled is shown first; saved TensorBoard tags
and report names remain compatible.

### Smaller-update continuation from the 10v1 checkpoint 7

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.curriculum_run --config Training/Mystic_Sim/configs/timeout25_10v1_goal5_from_cp07_lr1e4.json
```

This experiment starts from checkpoint 7 of `timeout25_10v1_goal5_from_5v1_cp08_seed42`.
It retains all learned weights and Adam moments, reducing the constant learning
rate from 0.00025 to 0.0001. The engine, rewards, ten enemies, five-kill goal,
seed 42, and other PPO settings stay unchanged. It adds 200,704 transitions and
writes a separate run directory named `timeout25_10v1_goal5_from_cp07_lr1e4_seed42`.
This tests a smaller optimizer step size; it does not guarantee smaller actual
policy changes. Compare sampled success and survival with the source checkpoint.

Explicit `continue_from` runs now permit a learning-rate change; the training
loop overrides the restored optimizer rate before the first update. Resume
still requires the exact saved PPO configuration. Other PPO hyperparameter
changes remain rejected by continuation validation. Add `--validate-only` to
check compatibility without training.

### Evaluate original spawn areas across the map

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.compare --config Training/Mystic_Sim/configs/compare_20v1_mapwide.json
```

Compares the source low-rate 10v1 checkpoint 9 and the resulting 20v1
checkpoints 3 and 10, on map-wide 10/20-enemy scenarios and the original full
scenario. Both modes use 100 episodes starting at seed 25000.

The map-wide stages reuse the original spawn boxes and quantities, sampling
NPC slots without replacement. `[1, 10000]` permits all reachable spawn cells
except the player cell, removing the local-distance cap on this 100x100 map.
This samples from original spawn areas; it does not force even spacing, populate
every region, or reproduce the full scenario's exact reset RNG sequence.
Respawns remain in each selected NPC's original box. The five-kill objective,
map collisions, observations, rewards and Y-boundary rules remain unchanged.
This evaluates placement transfer separately from increasing population to 80.
No training is launched by this command.

### Twenty Innies in the ten nearest original spawn areas

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.curriculum_run --config Training/Mystic_Sim/configs/timeout25_20v1_nearest10_areas.json
```

Continues from lower-LR 10v1 checkpoint 9 at learning rate 0.0001, retaining
weights and Adam state, for 200,704 additional transitions (CUDA, four envs,
seed 42). The five-kill goal, normal damage/HP, movement, cooldowns, Y rules,
and 50-second original-area respawns remain unchanged.

`placement: nearest_spawn_areas_v1` ranks original areas by shortest terrain
path from the sampled player start to a legal non-player cell, breaking ties
by object ID. It populates the nearest ten areas with their original quantity
of two Innies. Spawn positions use normal legal-cell sampling throughout each
selected rectangle. `[1, 10000]` is required as a neutral inherited field;
there is no individual-enemy distance cap. Unselected areas stay unpopulated.
Placement runs in a training env subclass and does not change the shared combat
engine or invalidate existing checkpoint engine hashes. All configuration
changes during reset are instance-local and restored before stepping.

Optional frozen-policy baseline before training:

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.compare --config Training/Mystic_Sim/configs/compare_nearest10_areas.json
```

This evaluates the source on the new placement and full scenario in both modes,
100 episodes per pairing, starting at seed 26000. Add `--validate-only` to either
command for compatibility/reset checks without training or evaluation rollouts.

### One Innie per original spawn area: 20, 30, 40

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.compare --config Training/Mystic_Sim/configs/compare_single_innie_areas.json
```

Evaluates lower-LR 10v1 checkpoint 9 on the nearest 20/30/40 original areas,
one Innie per area, plus the unchanged full scenario. There are 100 episodes
per stage/mode (sampled and greedy), seeds 26000�26099, matching the earlier
nearest-ten-area evaluation bank. Results go to
`runs/mystic_sim/comparisons/single_innie_areas.json`.

`nearest_spawn_areas_single_v1` preserves terrain-path ranking and legal-cell
sampling inside original rectangles, with one NPC assigned to each selected
area. An area is a spawn rectangle, not a single fixed tile. At 40 areas all
original spawn areas are represented, with half the full population. Each NPC
respawns in its assigned area after 50 seconds; a second NPC is not added.
Normal combat stats, training boundaries and the five-kill goal remain intact.
The existing two-per-area mode remains unchanged. These are initial placement
rules: NPC movement and deaths can leave areas temporarily empty.

Stage definitions are in `single_innie_area_stages.json`; evaluate first before
choosing whether and which stage to train. This comparison never trains.

### Direct map-wide transfer: twenty random distinct areas

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.curriculum_run --config Training/Mystic_Sim/configs/timeout25_twenty_random_areas_single_goal5_400k.json
```

Starts from the original lower-LR 10v1 checkpoint 9, retaining learned weights
and Adam state. Uses learning rate 0.0001, seed 42, four environments, CUDA,
and 400,384 actual transitions (400,000 requested, rounded to rollout batches).
Ten checkpoints include sampled/greedy evaluations and original full-scenario
evaluations. Rewards, five-kill goal, Y limits and normal combat stats are unchanged.

`random_spawn_areas_single_v1` uniformly samples 20 distinct reachable original
spawn areas without replacement each reset, without a nearest-area preference.
One Innie is placed normally within each chosen rectangle and respawns in that
same area after 50 seconds. Areas are not reselected on an NPC death. This
allows map-wide coverage but does not guarantee even geographical spacing.
Selection uses the environment's seeded RNG; the nearest-area modes are unchanged.

Optional 100-episode source baseline on the random-area task and full scenario:

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.compare --config Training/Mystic_Sim/configs/compare_twenty_random_areas_baseline.json
```

## Mixed-scenario training without gates

See [MIXED_SCENARIOS.md](MIXED_SCENARIOS.md) for the fresh LSTM PPO config, command, episode probabilities, per-scenario evaluations, and resume behavior. Manual stage configs remain supported.


## Standalone full-scenario training (no curriculum config)

Use python -m Training.Mystic_Sim.train_full or Train_Mystic_Full.bat for fresh 1v80 LSTM PPO with three million requested transitions. See [FULL_TRAINING.md](FULL_TRAINING.md) for portable setup, CLI overrides, validation, resume, and TensorBoard commands. Curriculum JSON workflows remain available separately.



Watch a frozen checkpoint on a chosen seed in Pygame: [Policy viewer commands and controls](WATCH_POLICY.md).


Task Y-boundary exits now terminate as failure without bootstrapping. Existing penalties are retained. See [checkpoint migration and continuation commands](Y_BOUNDARY_TERMINATION.md) before loading older checkpoints.
