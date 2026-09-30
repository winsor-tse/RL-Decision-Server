# Phase 6 validation — 2026-09-30

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
