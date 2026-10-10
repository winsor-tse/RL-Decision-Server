# LSTM run comparison

Audited `runs/mystic_sim/lstm_first` (20,480 transitions, checkpoint 10) and
`runs/mystic_sim/lstm_2_mil` (latest saved checkpoint 4, 800,768 transitions).
The latter is configured for two million transitions but its saved progress is
incomplete. Both use the same 51-value sensor contract, map, reward configuration,
seed, four environments, 256-step rollouts and four 64-step time minibatches.
Only the requested budget differs, which also changes learning-rate annealing.
The short run is not a checkpoint from the longer run.

## Findings

1. Action 4 (`attack`) is a no-op in this scenario: `gear_enabled=false`, so
   `Engine.advance` returns `gear_disabled`. It is not a working melee attack.
   Existing invalid-cast metrics count spells only and do not report this failure.
2. In the final fifth of logged long-run training, attack accounts for ~71.4%
   of actions and all four movement actions combined ~0.1%. Acid Cloud is nearly
   absent. This is a real policy change, not just a recording problem.
3. Checkpoint evaluation/GIFs use argmax, whereas training samples actions.
   Argmax picks attack on every step of both final checkpoints in a fresh
   30-seed evaluation (seeds 20000..20029). Early short-run checkpoints do move;
   their first three saved evaluations deal zero damage and collide repeatedly.
4. The actor's second Tanh layer is saturated, while the critic remains healthy.
   On the same 512-transition observation bank (four worlds, 128 steps, random
   actions), the fraction of actor second-layer activations with abs(x)>0.99 is:

   | Checkpoint | Saturated fraction |
   |---|---:|
   | Short, 20,480 steps | 0% |
   | Long, 200,704 steps | 96.74% |
   | Long, 400,384 steps | 99.93% |
   | Long, 601,088 steps | 99.95% |
   | Long, 800,768 steps | 99.86% |

   At 800k, action logits have standard deviations of only ~0.0007–0.0023
   across these varied observations. Mean attack probability is ~69.6%,
   Arcane Blast ~10.7%, Tempest ~19.6%, total movement ~0.067%, Acid ~0.0032%.
   The actor has become nearly observation-insensitive on the tested states.
   This is a concrete failure; its precise optimization cause still needs a
   controlled intervention. Critic explained variance ~0.89 and zero critic
   saturation do not establish actor health.
5. A four-world, 128-step collection/replay check using each saved model reproduced
   logits exactly when replayed in contiguous 32-step blocks with saved initial
   hidden/cell states and reset masks (max absolute logit error zero on CPU).
   This check and existing sequence tests found no evidence of a timestep shuffle
   or sequence-start state mismatch. It does not rule out all recurrent training
   issues, including effects of stale detached boundary states after updates.

## Fresh held-out evaluation

Same 30 seeds, stochastic actions with local seeded sampling:

| Metric | Short run | Long run latest |
|---|---:|---:|
| Wins | 0/30 | 0/30 |
| Kills per episode | 0.267 | 0.167 |
| Damage dealt per episode | 328,824 | 269,030 |
| Survival seconds | 11.79 | 9.18 |
| Successful moves, total | 555 | 2 |
| Collision attempts per episode | 4.33 | 0 |
| Cooldown rejections per episode | 17.37 | 8.57 |
| Raw return | -264.43 | -235.00 |

Greedy evaluation: both final checkpoints select attack for all 1,685 decisions,
with zero damage, zero kills, zero movement and zero wins. Short-run sampled
movement reflects remaining exploration, not established combat skill.

The long run's better return is consistent with avoiding penalties rather than
improving combat: positioning penalties fall from -43.03 to -8.57 per episode,
while damage reward falls from +29.94 to +24.49. Zero wins across the 15,293
training episodes recorded at checkpoint 4 reinforces the lack of successful
combat trajectories. These results do not prove a unique causal explanation.

## Recommended next work

- Log actor saturation, policy variation across observations, all action failure
  reasons (including gear_disabled), movement success and both sampled/greedy
  checkpoint evaluation. The original short smoke tests verified execution but
  missed long-run actor saturation; critic-only instrumentation was insufficient.
- Test a smaller/normalized actor head (for example a direct logits projection
  from LSTM output) and/or a lower actor learning rate in a fresh controlled run.
  Preserve the healthy critic configuration. Do not assume a new activation or
  higher entropy alone will fix this; gate on actor sensitivity and combat metrics.
- Resolve the action contract explicitly: either implement correctly configured
  melee combat, or intentionally define the disabled slot as wait. Simply removing
  waiting or penalizing all inactivity can force cooldown spam and is not a fix.
- Keep terrain sensors. They are identical in both runs; there is no evidence
  from this comparison that the relative sensor implementation caused the failure.
- Use easier combat encounters/curriculum after actor conditioning is addressed.
  Do not add arbitrary movement/button-press rewards or continue the unchanged
  run solely because raw return or critic explained variance looks good.

Raw audit outputs/scripts (gitignored): `runs/mystic_sim/lstm_run_comparison.json`,
`lstm_actor_saturation.json`, `audit_lstm_runs.py`, `audit_actor_saturation.py`.
No rewards, model code, checkpoints or running training processes were changed
by this audit. Long-run performance remains unvalidated beyond the saved budget.
