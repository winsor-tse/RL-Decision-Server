# Mixed-scenario training without gates

## MEDIUM CP9 continuation: MEDIUM-HARD then revised HARD

The dedicated `configs/medium_cp09_to_hard_pipeline.json` starts from
`mixed_medium_random_counts_seed42/checkpoints/checkpoint_09.pt` automatically.
It trains MEDIUM-HARD (40/50/60 enemies with 40%/40%/20% episode probabilities),
then transfers checkpoint 10 into revised HARD (60/70/80 with 40%/30%/30%).
Each stage requests one million transitions (1,000,448 after batch rounding).
All areas are populated; second occupants use randomly selected original areas,
and spawn cells within areas are randomized. There is no nearest-area preference.

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.pipeline --config Training/Mystic_Sim/configs/medium_cp09_to_hard_pipeline.json
```

Add `--validate-only` for geometry and source-checkpoint checks without training.
Actor, critic, learned LSTM parameters and optimizer state are preserved, with
constant LR 0.00025, original rewards and five kills to win. GIFs begin at
checkpoint 5; full-scenario evaluation runs in revised HARD only. There are no
success gates. Outputs use new directories:

- `runs/mystic_sim/curriculum/mixed_medium_hard_from_medium_cp09_seed42`
- `runs/mystic_sim/curriculum/mixed_hard_60_70_80_from_medium_hard_seed42`

For manual MEDIUM-HARD only:

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.curriculum_run --config Training/Mystic_Sim/configs/mixed_medium_hard_from_medium_cp09.json --continue-from runs/mystic_sim/curriculum/mixed_medium_random_counts_seed42/checkpoints/checkpoint_09.pt
```

For manual revised HARD, use `mixed_hard_60_70_80.json` and `--continue-from`
pointing at the MEDIUM-HARD checkpoint of your choice. Pipeline continuation uses
the final checkpoint automatically. To resume a partial pipeline stage, specify
`--start-stage medium_hard` or `--start-stage hard_60_70_80` together with
`--resume-checkpoint <latest checkpoint>`. The configured initial source is ignored
on resume; starting at a later stage requires its explicit continuation/resume
checkpoint. The previous EASY/MEDIUM/HARD configs remain available separately.

## Staged EASY / MEDIUM / HARD pipeline

This is the newer option: start fresh on EASY, then transfer checkpoint 10 into
MEDIUM and finally HARD. Each difficulty gets 1,000,000 requested transitions
(1,000,448 after batch rounding), for 3,001,344 total. No success gates are used.
The constant learning rate is 0.00025, with four environments, CUDA, and seed 42.
Actor, critic, learned LSTM weights and Adam state carry forward. Hidden/cell
memory resets per episode. The source is the final checkpoint, not an automatically
selected best checkpoint. Ten checkpoints and sampled/greedy evaluations are
retained per difficulty, including individual mixture members. Dedicated full
scenario evaluations run only in HARD. GIF recording starts at checkpoint 5
(inclusive) in every difficulty; checkpoints 1-4 still evaluate but do not render
or buffer GIF frames. Both greedy and sampled GIFs use the first evaluation seed,
currently `ppo.eval_seed=20000`. The mixture selection and initial world repeat
for that seed within a difficulty; sampled actions use a separate seeded RNG.
The agent's behavior can change as its weights improve. A different difficulty
can select a different scenario from that same seed.

These settings are configurable per stage under `runtime.record_from_checkpoint`
and `runtime.evaluate_full_scenario`. Evaluation results are still collected at
every checkpoint for the active mixture and its individual members. Skipping full
evaluation saves simulation work; skipping early GIFs saves rendering, frame
buffer memory, and disk space. Later GIFs retain their existing peak memory use.
Disabling dedicated full evaluation does not remove full episodes from a mixture
that explicitly contains them (EASY and MEDIUM do not).

Validate the complete pipeline without training:

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.pipeline --config Training/Mystic_Sim/configs/easy_medium_hard_pipeline.json --validate-only
```

Run all three difficulties automatically:

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.pipeline --config Training/Mystic_Sim/configs/easy_medium_hard_pipeline.json
```

| Difficulty | Probability per episode reset |
| --- | --- |
| EASY | 20% each: 1, 2, 3, 5, 10 enemies |
| MEDIUM | 30% each: 20, 30, 40 enemies; 10%: 50 enemies in original spawn areas |
| HARD | Equal probabilities: 50, 60, 70, 80 enemies in original spawn areas |

MEDIUM/HARD now vary population without nearest-area episodes or easy/medium
anchor episodes. EASY is unchanged. Revised run directories include
`medium_random_counts` and `hard_random_counts` to preserve the earlier runs.
These are new continuation runs, not resumes of the old mixture. Learning rate,
budgets, rewards, GIF schedule, and full-evaluation settings remain unchanged.

EASY uses original legal NPC spawn cells randomized within bounded shortest
walkable path distance of the sampled player start. It does not always choose
the exact nearest cell or require navigation across the map:

| Population | Goal | Initial path distance in tiles |
| --- | --- | --- |
| 1 | 1 | 2-6 |
| 2 | 2 | Half the episodes 2-6, half 5-9 |
| 3 | 3 | Half 2-6, half 5-9 |
| 5 | 5 | Half 2-8, half 6-12 |
| 10 | 5 | Half 2-10, half 6-14 |

These are distances from the player, not pairwise separation guarantees. Enemies
can occupy different directions/areas, and every 1v1-3v1 enemy starts within the
16-by-10 casting-coordinate limits. Terrain may still obstruct movement. Bounds
apply at initial reset only: movement, aggro, and normal original-area respawns
remain unchanged; there is no temporary fence or relocation during play.

Twenty/thirty use one per uniformly selected distinct original area. Forty uses
one in every original area. Fifty/sixty/seventy use one in every area plus a
second in 10/20/30 randomly selected distinct areas. Eighty uses the original two
per area. Individual spawn cells are randomized on reset in all cases. With only
40 areas, 40/80 necessarily cover every area; "random" does not mean arbitrary
positions outside the game's spawn rectangles. Respawns retain their assigned
area. All stages keep the original map, normal combat stats and rewards.
Only the introductory kill goals change. Compare successes per scenario rather
than interpreting the mixed success average as five-kill success.

Shared PPO settings, budgets and mixture weights are in
`configs/difficulty_stages.json`. `mixed_easy.json`, `mixed_medium.json`, and
`mixed_hard.json` select a difficulty and can override shared settings. Use a new
run directory for a new experiment. Changing `ppo.seed` in the shared config
changes all stage run directory suffixes. Later-stage transfers are checked for
compatible observations, mechanics and PPO settings by the existing runner.

Manual training remains available:

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.curriculum_run --config Training/Mystic_Sim/configs/mixed_easy.json --fresh
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.curriculum_run --config Training/Mystic_Sim/configs/mixed_medium.json --continue-from runs/mystic_sim/curriculum/mixed_easy_seed42/checkpoints/checkpoint_10.pt
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.curriculum_run --config Training/Mystic_Sim/configs/mixed_hard.json --continue-from runs/mystic_sim/curriculum/mixed_medium_random_counts_seed42/checkpoints/checkpoint_10.pt
```

To finish EASY manually and automate the remaining two stages:

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.pipeline --config Training/Mystic_Sim/configs/easy_medium_hard_pipeline.json --start-stage medium --continue-from runs/mystic_sim/curriculum/mixed_easy_seed42/checkpoints/checkpoint_10.pt
```

To resume an interrupted MEDIUM stage at checkpoint 4 and then run HARD:

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.pipeline --config Training/Mystic_Sim/configs/easy_medium_hard_pipeline.json --start-stage medium --resume-checkpoint runs/mystic_sim/curriculum/mixed_medium_random_counts_seed42/checkpoints/checkpoint_04.pt
```

Use the latest checkpoint; strict resume rejects overwriting later checkpoints.
As before, resume creates fresh episodes rather than restoring unfinished worlds.
Existing nonempty stage directories are rejected unless explicitly resuming.
If a stage process fails or stops before completing its budget, the pipeline stops
before transferring. It does not silently restart, skip stages, or select another
checkpoint. No training is launched by `--validate-only`.

## Original single mixed-run example

Start fresh LSTM PPO with automatic episode-level scenario sampling:

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.curriculum_run --config Training/Mystic_Sim/configs/mixed_easy_medium_hard.json
```

Add `--validate-only` to check every member's reset and the config without training.
The example explicitly uses fresh weights, seed 42, CUDA, four independent worlds,
256 rollout steps per world, a constant learning rate of **0.00025**, and 400,000
requested transitions (400,384 after rounding to complete PPO batches). Edit these
under `ppo` and `runtime`; this is an initial experiment budget, not a convergence
claim. Existing reward scaling, rewards, combat stats, and 1,024-step limit remain.

| Scenario | Probability per reset | Placement |
| --- | ---: | --- |
| easy | 20% | Five nearest original spawn areas, one Innie each |
| medium | 50% | Twenty randomly selected distinct original areas, one each |
| hard | 20% | All forty original areas, one each |
| full | 10% | Original eighty Innies: two per original area |

All members use map53 and the five-kill goal. Each vector slot independently draws
a member on reset; the member remains fixed until that episode ends. Weights are
normalized probabilities, not quotas or transition proportions: longer episodes
contribute more transitions. The mix is fixed throughout training. There are no
success gates, step-based promotions, enemy-stat changes, or scenario-ID policy
features. These proportions are configurable starting choices, not a validated optimum.

The existing ordered recurrent PPO batches remain intact. Learned LSTM, actor,
and critic parameters persist across scenarios; hidden/cell state and observable
history reset at episode boundaries. Sampling uses a separate seeded RNG per
world, saved in checkpoints. Resume restores its future sampling sequence but
starts fresh episodes, as before; it does not replay unfinished worlds exactly.

Ten checkpoints retain mixed greedy/sampled gameplay recordings. Each checkpoint
also evaluates **every member separately**, in both greedy and sampled modes,
using the held-out evaluation seeds. The full scenario is always evaluated.
CLI, JSON files (`evaluation/*_scenario_*`), and TensorBoard
(`evaluation_scenarios/`) report the separate results. `training_scenarios/` logs
episode results, completed-episode counts per update, and transition fractions.
The added evaluations increase checkpoint wall time; `ppo.eval_episodes` controls
their size. No result automatically changes the mixture or stops training.

Manual stages and checkpoint continuation remain supported via `--stage`,
`--continue-from`, and `--resume`. Mixture checkpoints work with the existing
comparison runner; changing mixture settings requires a new continuation run,
not strict resume. For experiments across seeds, change `ppo.seed`; the example
run directory includes the seed. Config files are currently ignored by the
repository's existing `.gitignore`; explicitly include this JSON when sharing.

After collecting runs, compare individual scenario results across seeds and
measure actual transition exposure before changing the configured mix. The older
curriculum plan's provisional advancement gates do not apply to this pipeline.
Manual curriculum remains an alternative, not a prerequisite for this run.
