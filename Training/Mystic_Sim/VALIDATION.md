# Phase 6 validation — 2026-09-30

## Critic repair follow-up — 2026-10-05

The original results below describe v1. New runs use v2 checkpoints, training-only
reward scale 0.01, gamma 0.999, disabled value clipping by default, separate
actor/critic gradient limits of 0.5, and simulator cooldown penalty -1.

- All 114 simulator/trainer tests passed, including numeric optional value-loss
  clipping, actor gradient isolation from a large critic gradient, raw versus
  scaled reward logging, configuration persistence on resume, and v1 rejection.
- A CPU four-world, 16,384-transition probe completed ten checkpoints. Final
  rollout explained variance was 0.791 (post-update 0.792), critic prediction
  standard deviation 0.211 versus target standard deviation 0.228, and second
  hidden-layer saturation fraction 0.0. Value loss was 0.00551 in scaled units;
  actor/critic pre-clip norms were 0.405/0.0524. Unlike the old constant critic,
  this probe produced state-dependent predictions. It is not a controlled
  attribution of improvement to any one of the simultaneously requested changes.
- Final greedy evaluation on three seeds averaged 0.67 kills and -282.82 return;
  the run still had zero training wins. This short probe does not establish
  combat mastery or performance across training seeds.
- A CUDA four-world, 160-transition smoke run completed ten checkpoints and ten
  gameplay GIFs. Local output directories are `runs/mystic_sim/critic_repair_seed1`
  and `runs/mystic_sim/critic_repair_cuda_smoke`.

## Original v1 validation

Environment: Windows, PyTorch 2.9.0+cu130, Gymnasium 1.2.1, local CUDA available.
The tested environment uses default map53/combat rewards and 1,024-step episodes.

- CPU smoke: two environments, 160 transitions, ten optimizer updates. Stopped
  after checkpoint 1 and resumed through checkpoint 10; all ten files are present.
- CUDA smoke: four environments, 160 transitions, ten optimizer updates. All ten
  checkpoints and ten multi-frame Pygame GIFs were created. A final GIF frame was
  inspected: it includes player/enemies, HP/MP, spell state, steps and end reason.
- CUDA-trained checkpoint reloaded on CPU for two held-out episodes (seeds
  20000–20001), with another GIF and a random-policy comparison.
- Two bounded CPU learning runs: seeds 11 and 12, one environment each, 8,192
  transitions, rollout 128, four update epochs, four minibatches, other defaults.
  Each produced ten checkpoints and evaluated greedy actions on seeds 10000–10001.
  Recording was disabled for these learning probes. All final losses were finite,
  and final parameter-change norms were 0.00222 and 0.00281, respectively.

| Mean evaluation metric | Random baseline | PPO seed 11 | PPO seed 12 |
|---|---:|---:|---:|
| Return | -345.91 | -591.23 | -933.05 |
| Kills | 0 | 0 | 0 |
| Win fraction | 0 | 0 | 0 |
| Simulated survival seconds | 16.2 | 7.0 | 13.7 |
| Enemy HP removed | 392,510.5 | 0 | 0 |
| Player HP removed | 10,706 | 9,689 | 10,028 |
| Invalid casts / cast attempts | 0.658 | 0 | 0 |
| Training transitions/sec | N/A | 876 | 889 |

Zero invalid casts in these greedy PPO evaluations means the policies were not
casting, not that they mastered cooldowns. Player damage can exceed max HP over
an episode because regeneration restores HP. These short policies performed
worse than random; there is **no demonstrated combat-learning improvement**.
They establish executable training, nonzero updates, and repeatable evaluation.
Longer training and further evaluation are still needed before selecting a policy.

SPS excludes checkpoint evaluation/recording, is hardware-specific, and was
measured with the two CPU runs overlapping. It is not a Phase 7 vector scaling
benchmark. Raw local artifacts are in `runs/mystic_sim/phase6_cpu_smoke`,
`phase6_cuda_smoke`, `phase6_seed11`, and `phase6_seed12` (gitignored).
# Recurrent PPO validation — 2026-10-08

`python -m unittest discover -s Tests -p "test_mystic_sim*.py"`: **121 passed**.
New recurrent gates cover causal cast/collision history, independent autoreset,
contiguous block coverage, collection/replay equivalence across multiple worlds,
gradient flow through time with no leakage across episode resets, evaluation RNG
isolation, actual recurrent weight updates, checkpoint compatibility and resume.
Existing feed-forward PPO and simulator tests continue to pass.

CUDA smoke: `runs/mystic_sim/lstm_cuda_smoke_20261008`, 640 transitions,
two worlds, 32-step rollouts, two 16-step sequence minibatches, one update epoch.
All ten checkpoints, evaluations and Pygame GIF recordings completed. The final
CUDA checkpoint also loaded and ran stochastic evaluation on CPU through the
normal evaluation CLI. This small run validates execution, not learning quality.

No two-million-transition recurrent training experiment or live ZMQ validation
has been run. This smoke used the original 46-feature recurrent contract,
superseded by the sensor contract below; it is not a sensor-model CUDA benchmark.

## Bounded terrain sensors — 2026-10-08

The recurrent contract is now `mystic-local-terrain-51-v2`. Four static-terrain
rays replace the three remembered blocked-origin values. The full Mystic Sim
suite passes **125 tests**, including adjacent/farthest-visible/beyond-range
blockers, map edges, nearest-hit behavior, translation invariance, recomputation
after movement/reset, exclusion of moving entities, and old-contract rejection.
Recurrent CPU training, resume and evaluation tests also pass with 51 inputs.
Live adaptation remains documentation only; no live payload parity is claimed.

## Attack removal

The policy now uses `mystic-seven-v2` and the recurrent input is
`mystic-local-terrain-50-v3`. Attack is removed; IDs 4/5/6 are the three spells.
Old eight-action checkpoints are incompatible. Viewer idle uses an explicit
non-policy clock method. The full existing 125-test Mystic suite passed after
migration; targeted contract/recurrent tests then passed with an additional test
for seven-output heads, previous-action encoding and rejected idle/old indices.
Legacy demonstrations containing attack are rejected without dropping rows.
The actor-head architecture is unchanged; its repair plan is in the README.

## Linear actor head and diagnostics — 2026-10-10

Step 1 is implemented. New recurrent runs default to `--actor-head linear`;
the optional `tanh` head retains the prior architecture. The critic, reward
configuration, environment and recurrent sequence semantics are unchanged.

All **129 Mystic Sim tests passed**. Added checks cover identical matched-seed
critic/trunk initialization between heads, explicit old-checkpoint Tanh fallback,
architecture mismatch rejection, constant-policy responsiveness diagnostics,
both evaluation modes, and new TensorBoard actor metrics. Existing ordered
sequence/BPTT, reset, optimizer/resume and RNG-isolation tests remain green.

CUDA smoke: `runs/mystic_sim/linear_head_smoke_20261010`, 640 transitions across
two environments. All ten model checkpoints, twenty evaluation JSON reports,
and twenty multi-frame Pygame GIFs completed with no recording errors. The new
checkpoint was also evaluated on CPU. The existing `lstm-200k` Tanh checkpoint
loads via the normal evaluation CLI with its original head.

These checks establish execution and compatibility, not improved training
performance. No full training experiment, curriculum or reward changes were
performed as part of Step 1.
