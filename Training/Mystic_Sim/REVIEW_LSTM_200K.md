# Seven-action LSTM 200k audit

Run: `runs/mystic_sim/lstm-200k`, seed 42, completed 200,704 transitions,
3,493 training episodes, zero training wins. Seven-action contract and 50-input
terrain/history features were correctly used. The actor still has the two
64-unit Tanh head; the proposed head repair has not been implemented.

## Fresh evaluation

30 held-out seeds 20000..20029, final checkpoint, both greedy and locally seeded
sampled actions. Raw diagnostics: `runs/mystic_sim/lstm-200k/audit_30_episodes.json`;
reproducible script: `runs/mystic_sim/audit_200k.py` (gitignored).

| Metric | Greedy | Sampled | Random seven-action baseline |
|---|---:|---:|---:|
| Wins | 0/30 | 0/30 | 0/30 |
| Kills per episode | 0.233 | 0.400 | 0.233 |
| Survival seconds | 9.35 | 11.41 | 12.42 |
| Damage dealt per episode | 269,993 | 369,512 | 357,310 |
| Invalid cast rate, episode mean | 87.7% | 77.8% | 65.0% |
| Collisions per episode | 0 | 1.37 | 6.10 |
| Cooldown rejections per episode | 41.10 | 38.13 | 17.67 |
| Raw return | -267.14 | -261.67 | -275.31 |

Greedy evaluation chose Tempest for all 1,403 decisions: 170 accepted casts,
1,233 cooldown rejections, no movement. Saved checkpoint evaluations 2–10 also
have identical results on their three fixed evaluation seeds.

Sampled evaluation produced 238 movement attempts (197 successful), 454 Arcane
attempts, 217 Acid attempts and 802 Tempest attempts over 1,711 decisions.
Thus the video is stationary due to argmax spell preference; the stochastic
training policy retains movement but has not learned useful state-dependent
combat timing. The small kill improvement over random is not a reliable combat
advantage established across training seeds.

## Actor/critic diagnosis

A fixed observation bank of 512 random-policy transitions across four simulator
worlds was replayed through checkpoints 1, 2, 5 and 10. Actor second-Tanh
saturation (abs activation >0.99) was 0%, 2.99%, 34.18%, and 9.99% respectively.
This differs materially from the previous run's 99.9% saturation; saturation
alone is not a sufficient diagnosis of this run's failure.

At the final checkpoint, mean action probabilities were approximately
`[.02954, .04972, .01564, .03158, .26635, .12912, .47804]` in action order.
Per-action probability standard deviations across the observation bank were
only ~0.00004–0.00043 (at most ~0.043 percentage points). The policy is nearly
observation-independent on the tested states despite varied HP, MP, enemies,
history and terrain inputs. Actor LSTM output mean feature std was ~0.0345,
versus ~0.0034 in the final hidden head features. This supports testing the head
and adding actor sensitivity diagnostics; it does not isolate one causal bug.

The critic's final explained variance is 0.939 with zero reported hidden Tanh
saturation. Better critic fit does not imply better control. Final training
entropy is 1.397: the policy has not collapsed onto one sampled action, but has
largely collapsed to a similar distribution across states.

The final fifth of training assigns ~12.8% probability mass empirically to
movement and ~48.7% of chosen actions to Tempest. Invalid cast rate is ~78.2%.
PPO used LR annealing over the 200k budget; its final update LR was approximately
1.28e-6. A completed checkpoint cannot simply be resumed with a larger budget
through the existing resume path, which restores the saved configuration.

## Reward interpretation and next experiments

There is already dense enemy-damage reward (+25 per full enemy HP bar), kill
reward (+10), incoming-damage penalty (-50 per player HP bar), death -200,
cooldown rejection -1, and escalating collision penalties -5/-10/-15/-20.
Sampled episodes earn +33.64 damage and +4 kills on average but incur -38.13 in
cooldown penalties, before incoming damage/death. One collision costs as much
as removing 20% of an enemy HP bar; four consecutive collisions cost -50.
These incentives can suppress movement exploration, but this audit does not
prove that collision penalties caused the actor's weak state dependence.

Recommended order (not implemented by this audit):

1. Add actor saturation/sensitivity, per-action failure counts and sampled plus
   greedy evaluation before another long run.
2. Keep LSTM size 128 and the healthy critic. Compare the existing actor head
   with a direct LSTM-to-seven-logit head in fresh, otherwise matched short runs.
   Separately test lower actor LR if needed; do not conflate all interventions.
3. If timing/observation sensitivity improves but successful combat stays rare,
   introduce easier encounters/curriculum while preserving combat mechanics.
4. Only then consider reward shaping or collision-penalty ablation. Do not add
   unconditional movement, spell-button, proximity or survival bonuses. If needed,
   test a small bounded potential-based tactical-position signal with correct
   terminal treatment; reward safer useful positioning, not motion for its own sake.

The actor architecture, rewards, environment and checkpoints were not changed
by this audit. Longer training might help exploration but is not an evidence-backed
fix for the observed weak state dependence. No claim that LSTM itself is incapable
or that this task is unsolvable follows from a single 200k run.
