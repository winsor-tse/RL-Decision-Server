# Mystic curriculum foundation

## Phase 0 — Freeze the comparison contract (implemented)

The baseline remains the original map53: 80 Innies, five kills to win,
1024-step maximum, existing terrain, spawn boxes, player stats, NPC stats,
combat, respawn, rewards, seven actions and recurrent observation features.
`make_full_evaluation_env` reuses the existing training `make_env` factory.
It is independent of the current encounter factory. The original `train` CLI
defaults to the full scenario. The new `curriculum_run` CLI selects one stage;
neither command automatically changes stages.

`curriculum_baseline.json` preserves summaries, exact seeds, checkpoint references
and audit file hashes from the existing `lstm-linear-200k` 30-episode greedy and
sampled audits. Both had zero wins. These are historical results, not a rerun
after this change. Keep this comparison even when easier encounters succeed.

`EncounterConfig` is immutable. Its contract records stage ID, enemy count,
kill goal and initial shortest-path distance band. `scenario_for_encounter`
changes only that configuration and the victory kill count. Reward coefficients,
episode length and mechanics remain identical. Existing checkpoint metadata
already serializes scenario configuration and fingerprints engine source, so
encounter settings are captured automatically. Older strict-resume checkpoints
will detect these source/configuration changes. Do not bypass that check;
explicit compatible cross-stage continuation is now available in Phase 3.

**Keep all learned actor and critic parameters, including both LSTMs, across
curriculum stages.** Continue optimizer state and the overall learning-rate
schedule. Reset per-episode hidden/cell states and observation history on reset;
those are not learned weights. The network architecture and sequence minibatching
are unchanged. Phase 3 provides checkpoint continuation with fresh episode memory.

## Phase 1 — Reproducible encounter builder (implemented)

Use the original map loader, player start rectangle, NPC spawn boxes, NPC
construction, scheduler and combat engine. Select only the requested number of
Innies before scheduling. There are no hidden or inactive surplus NPCs.

Initial NPC cells must be unoccupied, walkable, reachable from the player by
cardinal movement, and inside an original NPC spawn box. Distances are shortest
walkable path lengths in tiles, inclusive at both ends; they are not spell
eligibility distances. Box population limits remain intact. The builder tries
each legal player start at most once in seeded shuffled order and reports a
clear error if the encounter cannot be placed. The supported map has disjoint
spawn boxes; overlapping boxes are explicitly rejected for custom encounters.

After placement, normal movement, aggro, damage and 50-second respawns apply.
Respawns use the selected NPC's original box; the initial distance constraint
does not constrain subsequent movement or respawns. Terrain never changes.
No stage ID, path search result or extra hidden state enters the observation.
Detached `info['encounter']` reports the encounter configuration.

```python
from Custom_enviornments.Mystic_Sim.config import EncounterConfig
from Training.Mystic_Sim.curriculum import EncounterFactory, make_full_evaluation_env
from Training.Mystic_Sim.core import SimBatch
from Training.Mystic_Sim.evaluate import evaluate

factory = EncounterFactory(EncounterConfig(
    stage_id='one_innie', enemy_count=1, kill_goal=1, path_distance=(2, 12)))
env = factory()
observation, info = env.reset(seed=42)

# Reuse existing independent-world batching and episode history handling.
batch = SimBatch(4, seed=42, factory=factory, recurrent=True)

# Once an agent is available, reuse the existing evaluator for both tasks:
# stage_result = evaluate(agent, range(20000, 20030), factory=factory)
# full_result = evaluate(agent, range(20000, 20030), factory=make_full_evaluation_env)
# Repeat with stochastic=True for sampled evaluation.
```

Construct a fresh environment/factory for each encounter. Do not mutate a running
world to switch stages. Both single-world and batched callers use the same
builder. Seeds reproduce placements, NPC movement intervals and subsequent
action traces; separate worlds share no mutable episode state.

## Phase 2 — Configurable manual stages (implemented)

`configs/curriculum_manual.json` defines a selected `stage`, a dictionary of
available encounters, `ppo` settings, `runtime` settings, and an optional shared
`schedule_timesteps`. The supplied stages contain 1, 2, 3, 5, 20 and 40 Innies,
plus `full: null` for the exact original reset. Stage names are encounter IDs.
These are provisional choices, not automatically advancing difficulty levels.
Add or change entries to train easier runs for as long as you want.

The example selects `one_innie`, recurrent PPO with the linear actor, CUDA,
four environments, 256 rollout steps and a 200,000-transition local budget
(rounded to 200,704 to complete PPO updates). Episodes still cap at 1024 steps.
Other optimization/reward settings reuse the existing PPO implementation.

**The example uses `anneal_lr: false`.** A constant learning rate allows you to
extend the curriculum later without resetting an exhausted decay schedule. To
use annealing instead, set `anneal_lr: true` before the first run and choose a
shared `schedule_timesteps` covering all planned stages. Continuations preserve
that horizon and cumulative step offset; runs exceeding it fail rather than
silently restarting decay. You cannot change annealing policy during continuation.
The original trainer's default hyperparameters are unchanged.

Paths are relative to the working directory (run commands from the repository
root). `runtime.run_dir` supports `{stage}` and `{seed}` placeholders. Use a new
directory for each new run, including another run of the same stage. Existing
run output is never silently overwritten.

Validation only, without training, optimizer updates or run files:

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.curriculum_run --config Training/Mystic_Sim/configs/curriculum_manual.json --validate-only
```

Validation checks schema/types, PPO batch constraints, selected encounter
placement, checkpoint compatibility when supplied, schedule budget, and explicit
initial training seeds against evaluation/reserved validation seeds. Episode
autoresets keep each environment's independent RNG stream; they do not cycle
through held-out seed numbers. Validation does not require a CUDA device because
it never constructs a training agent. Actual CUDA availability is checked at launch.

## Phase 3 — Train, save and manually continue a selected stage (implemented)

Start the configured one-Innie run yourself:

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.curriculum_run --config Training/Mystic_Sim/configs/curriculum_manual.json
```

Override `--stage`, `--run-dir` or `--device` on the command line; edit remaining
settings in JSON. One invocation runs one encounter for its local budget and
stops. It saves ten checkpoints, TensorBoard outputs, progress JSON and optional
greedy/sampled gameplay GIFs through the existing trainer/recorder. Set
`runtime.stop_after_checkpoint` to 1–10 to stop at a safe checkpoint boundary.

At every checkpoint, evaluate both greedy and sampled policies on the selected
stage and the original full scenario. The example uses 30 fixed held-out episodes
per mode/task: 120 evaluation episodes per checkpoint for an easier stage.
This adds evaluation time beyond the requested training steps. Only selected-stage
gameplay is recorded automatically. Stage metrics remain under `evaluation/`;
full-scenario TensorBoard metrics use `evaluation_full/`. Full JSON reports end
in `_full_greedy.json` and `_full_sampled.json`. Stage/evaluation rewards stay in
original units; training-only reward scaling remains unchanged.

Resume an interrupted run with the same config/stage and original total budget
(replace the checkpoint number as needed):

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.curriculum_run --config Training/Mystic_Sim/configs/curriculum_manual.json --resume runs/mystic_sim/curriculum/one_innie_seed42/checkpoints/checkpoint_05.pt
```

Remove/increase `stop_after_checkpoint` before resuming beyond it. Resume restores
saved weights, optimizer, RNG and local progress, but starts fresh simulator
episodes. It is not an exact mid-episode replay. A completed budget cannot be
resumed; use continuation to add another run.

Continue into two Innies, retaining all actor/critic/LSTM parameters and Adam
moments. This starts a new local budget and preserves cumulative schedule progress:

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.curriculum_run --config Training/Mystic_Sim/configs/curriculum_manual.json --stage two_innies --continue-from runs/mystic_sim/curriculum/one_innie_seed42/checkpoints/checkpoint_10.pt
```

The same command can select `one_innie` again with a new `--run-dir`, or select
`full` later. You decide when to change the goal. Without `--continue-from`, a
new run initializes fresh weights. No stage switch happens mid-episode.

Continuation permits changing only the encounter/victory goal in the environment.
It requires identical map, engine, rewards, action/observation schema and network
architecture. PPO settings must also match except the local timestep budget,
training seed, evaluation seed and evaluation episode count. Checkpoints from
older incompatible engine/schema revisions remain rejected. Run-level episode
and win counters start afresh; checkpoint lineage, cumulative timesteps and the
shared learning-rate horizon are saved separately under `curriculum`.

Evaluate a saved easier policy on either task without training:

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.evaluate runs/mystic_sim/curriculum/one_innie_seed42/checkpoints/checkpoint_10.pt --episodes 30 --output runs/mystic_sim/stage_eval.json
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.evaluate runs/mystic_sim/curriculum/one_innie_seed42/checkpoints/checkpoint_10.pt --episodes 30 --full-scenario --output runs/mystic_sim/full_eval.json
```

Add `--stochastic` for sampled actions or `--record <path.gif>` for gameplay.
Evaluation reconstructs and validates the checkpoint's source encounter first;
`--full-scenario` then chooses the unchanged full environment for the comparison.

## Running additional stages (implemented)

### Ready-to-run continuation after the one-Innie experiment

Phases 2/3 already support all configured stages. The additional
`configs/three_innies_from_one.json` selects three Innies and your one-Innie
`checkpoint_08.pt`. It inherits the shared settings using `extends`, so PPO
parameters do not need to be duplicated. The source checkpoint is a candidate
from the earlier analysis, not an assertion that the advancement gate passed.

```powershell
# Continue checkpoint 8 into three Innies (new 200,704-transition budget).
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.curriculum_run --config Training/Mystic_Sim/configs/three_innies_from_one.json

# Matched fresh three-Innie comparison; explicitly use a different output folder.
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.curriculum_run --config Training/Mystic_Sim/configs/three_innies_from_one.json --fresh --run-dir runs/mystic_sim/curriculum/three_innies_fresh_seed42

# When you decide to proceed, continue the three-Innie policy into five Innies.
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.curriculum_run --config Training/Mystic_Sim/configs/curriculum_manual.json --stage five_innies --continue-from runs/mystic_sim/curriculum/three_innies_from_one_cp08_seed42/checkpoints/checkpoint_10.pt

# List all available encounters without training, resets or checkpoint loading.
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.curriculum_run --config Training/Mystic_Sim/configs/curriculum_manual.json --list-stages
```

The continuation run writes to
`runs/mystic_sim/curriculum/three_innies_from_one_cp08_seed42`.
Add `--validate-only` to any training command to check it without starting.
Later use `--stage twenty_innies`, `forty_innies` or `full` with the checkpoint
you explicitly choose. No stage is selected or promoted automatically.

Config inheritance recursively merges objects; child values replace other
values. `extends` is resolved relative to the config file. Checkpoint and run
paths remain relative to the repository working directory. Set `continue_from`
in a config to retain a selected model, or set it to `null` for a fresh run.
`--continue-from` overrides that configured checkpoint. `--fresh` ignores it,
and `--resume` restores the supplied checkpoint instead. Those three CLI options
are mutually exclusive. Keep separate run folders for separate experiments.
Change a child config's `ppo.total_timesteps` to choose an additional local
budget; all existing checkpoint compatibility and schedule checks still apply.

The actual checkpoint-8-to-three-Innie preflight passed. Eight curriculum unit
tests passed, including inheritance, stage listing, configured continuation and
fresh-start overrides. No training was run for this extension.

## Later phases — automatic progression deferred to Phase 7 or later

Add a stage controller on top of the manual training and continuation tools,
with checkpointed gate history and stage changes at episode boundaries.
A provisional ladder is 1/2/3/5/10–20/40/80
Innies, with kill goals 1/2/3/5/5/5/5. Validate placement difficulty before fixing
distance bands. Keep the original map and reward coefficients throughout.

Use the provisional gate of at least 24 successes in 30 held-out episodes in
two consecutive evaluations (80%). The two seed banks in `curriculum.py` reserve
20000–20029 and 21000–21029 for validation; exclude them from training resets.
Measure greedy and sampled policies separately, and continue evaluating the
full scenario throughout. Reserve a separate final test seed bank and repeat
experiments across training seeds. The constants currently document the gate;
they do not implement advancement. The manual runner excludes reserved validation
seeds from explicit initial training seeds, including resume seed offsets.

No PPO training was run to implement Phases 2/3. Validation uses configuration,
reset, synthetic-checkpoint and CLI-wiring tests without optimizer steps. Actual
learning performance and gameplay from these stages await your training runs.
Validation passed: 139 non-training Mystic Sim tests and the example CLI's
`--validate-only` preflight. The two existing training smoke tests were excluded;
Adam's update method was guarded to raise if any test attempted training.

Validation: full-reset pre-change fingerprint, immutable/validated configuration,
multi-seed legal placements, deterministic action replay, isolated worlds,
impossible-placement rejection, stage victory and original-box respawn tests.
