# Mystic curriculum foundation

## Phase 0 — Freeze the comparison contract (implemented)

The baseline remains the original map53: 80 Innies, five kills to win,
1024-step maximum, existing terrain, spawn boxes, player stats, NPC stats,
combat, respawn, rewards, seven actions and recurrent observation features.
`make_full_evaluation_env` reuses the existing training `make_env` factory.
It is independent of the current encounter factory. The existing CLI still
trains the full scenario; this foundation does not automatically change stages.

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
explicit cross-stage weight transfer is a later pipeline feature.

**Keep all learned actor and critic parameters, including both LSTMs, across
curriculum stages.** Continue optimizer state and the overall learning-rate
schedule. Reset per-episode hidden/cell states and observation history on reset;
those are not learned weights. Phase 0/1 does not modify the networks or add a
checkpoint-transfer mechanism.

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

## Later phases — not enabled yet

Add a stage controller and trainer/CLI integration, with checkpointed stage and
progress, stage changes at episode boundaries, retained LSTM weights, and
explicit compatible weight transfer. A provisional ladder is 1/2/3/5/10–20/40/80
Innies, with kill goals 1/2/3/5/5/5/5. Validate placement difficulty before fixing
distance bands. Keep the original map and reward coefficients throughout.

Use the provisional gate of at least 24 successes in 30 held-out episodes in
two consecutive evaluations (80%). The two seed banks in `curriculum.py` reserve
20000–20029 and 21000–21029 for validation; exclude them from training resets.
Measure greedy and sampled policies separately, and continue evaluating the
full scenario throughout. Reserve a separate final test seed bank and repeat
experiments across training seeds. The constants currently document the gate;
they do not implement advancement or training-seed exclusion.

Validation: full-reset pre-change fingerprint, immutable/validated configuration,
multi-seed legal placements, deterministic action replay, isolated worlds,
impossible-placement rejection, stage victory and original-box respawn tests.
