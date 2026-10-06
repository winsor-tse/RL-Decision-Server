# Review of the completed two-million-transition run

Run: `runs/mystic_sim/20261005_232550_seed1_99510c`.
Reviewed on 2026-10-05. No training code or reward settings changed in this audit.

The run completed 2,000,384 transitions (rounded to full rollouts), 3,907 PPO
updates, and 12,910 episodes with zero wins. It used four environments, 128
decisions per rollout per world, four minibatches, four epochs, gamma 0.99,
learning rate 0.00025 annealed nearly to zero, and unscaled environment rewards.
The environment episode limit remains 1,024; rollout boundaries do not reset it.

## Evidence

Final value loss was 6,018.57. Explained variance was effectively zero throughout
the run. A fresh 512-transition probe of checkpoint 10 (four worlds, starting
seeds 20000–20003, Torch sampling seed 123) found:

- Critic predictions ranged only from -133.678970 to -133.678955.
- Critic target standard deviation was 15.37 on that probe.
- All second hidden-layer critic Tanh activations had absolute value above 0.99.
- Before joint clipping, actor gradient norm was 0.237 and critic norm was 52.92.
  These norms include the configured loss coefficients. The combined clip limit
  is 0.5. This establishes severe imbalance in that probe, but does not measure
  the resulting Adam parameter update or prove a unique cause of failure.

This is evidence of an effectively constant, saturated critic, not merely a
large numeric loss caused by reward units. The policy still changes during
training; its entropy has not collapsed to zero.

Across training, roughly 11.2% of actions attempted spells and 21.3% used the
disabled attack/idle action. Mean damage reward per step was +0.312, compared
with -1.188 positioning penalties and -0.398 damage-taken penalties. Kill count
averaged 0.106 per completed training episode. These statistics do not establish
that all spell actions have negative expected value.

## New held-out evaluation

All three policies used the same 30 seeds, 20000–20029, and unchanged task rules.
JSON reports are `audit_stochastic_30.json` and `audit_greedy_30.json` in the run
directory; both include the random baseline. Every policy had zero wins.

| Mean metric | Random | PPO sampled | PPO greedy |
|---|---:|---:|---:|
| Return | -336.29 | -427.77 | -1216.41 |
| Kills | 0.40 | 0.27 | 0 |
| Episode duration, seconds | 14.42 | 39.27 | 27.58 |
| Enemy HP removed | 404,692 | 671,958 | 80,690 |
| Invalid casts / attempts | 0.609 | 0.299 | 0.028 |

The sampled policy lasts longer and deals more total damage, but does not
translate this into more kills. Longer duration can also mean a boundary
truncation rather than successful combat. Low greedy invalid-cast rate is not
evidence of effective casting: it deals much less damage.

## Recommended order of work

1. Repair critic training: add configurable training-only reward scaling
   (0.01 is an initial experiment), independent/optional value clipping, and
   separate actor/critic gradient logging and clipping. Keep raw environment
   rewards in evaluation/dashboard metrics. Compare critic output variance,
   activation saturation and explained variance rather than only raw loss.
   Current value clipping reuses the policy coefficient 0.2 in reward units.
2. Add observable spell state (remaining slot/family/global cooldowns and
   affordability) and nearby obstacle state. The feed-forward 26-value policy
   lacks these inputs. This requires a versioned observation contract and new
   checkpoints, not silently loading old models into a changed input schema.
3. Test cooldown penalty -1 versus current -5 without paying for button presses.
   At current damage weight, one -5 penalty offsets 20% of one enemy's HP bar.
   Preserve Y boundaries/truncation and existing task constraints in the initial
   controlled comparisons. Split collision, cooldown and Y penalty diagnostics
   so the broad positioning component does not hide their separate effects.
4. After critic repair, test gamma 0.995/0.999 and rollouts 512 against the
   existing baseline, keeping episode limit 1,024. Avoid changing all settings
   at once. A learning-rate floor or non-annealed pilot can help diagnose whether
   updates fade too early; final LR here was approximately 6.4e-8.
5. Evaluate both sampled and greedy actions over fixed held-out seeds and repeat
   across multiple training seeds. Select on wins/kills, supported by damage,
   survival and invalid-action metrics. If full combat remains undiscovered,
   validate a scripted winning policy and consider fewer-enemy curricula.

These are prioritized experiments, not proven optimal settings. The completed
run does not support spending another larger budget on the unchanged setup.
