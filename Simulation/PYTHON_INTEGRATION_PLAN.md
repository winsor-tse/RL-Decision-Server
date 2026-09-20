# Plan: C# Yugen Saga mechanics in a Python Gymnasium environment

## Goal

Build a headless Python `gymnasium.Env` that simulates the level-135 Mystic
combat scenario described in `Simulation/Source-Code`. The first version should
be policy-compatible with `Custom_enviornments/Test_Env/Mage.py`:

- the same 11 action IDs and labels;
- the same 26-element `float32` observation layout;
- the same Gymnasium `reset()` and `step()` return shapes;
- the same reward component names and episode outcomes where those rules still
  make sense;
- no socket, rendering, sleeping, or file I/O in the simulation hot path.

The C# snippets should be treated as the behavioral specification and translated
to Python. They are not a compilable game module: they reference many absent
server types and functions. Embedding a .NET runtime would retain those missing
dependencies, complicate process management, and make parallel environments
harder without improving fidelity.

## What the supplied source establishes

| Area | Known behavior |
|---|---|
| Map | 100 by 100 tiles; collision/terrain data is not supplied. |
| Decision time | One player action every 200 ms. Use simulated time only. |
| Player | Level 135 Mystic; HP 9,463; MP 15,108; displayed combat factors and mitigation are listed in `Current_Player_Stats.txt`. |
| Regeneration | HP and MP regeneration occur every 2 seconds. |
| NPC movement | Base move speed is 1 second in the supplied scenario, randomized by 0.9 to 1.1 at spawn. |
| NPC aggro | Scenario radius is 4; base NPC code checks for new aggro every 1.5 seconds and drops a target beyond distance 18. |
| NPC update | Drop invalid aggro, check new aggro, move/face, then attack. `RedBotNPCScript.Update` additionally calls `TryToCast`. |
| NPC scaling | `RedBotNPCScript.OnCreated` supplies formulas, but enemy HP and damage depend on missing `ScaledCalcs.GetEnemyHP` and `NPCRecursiveDmg.GetDamage`. |
| Mitigation | The NPC script supplies AC, blocking, magic/melee mitigation, death, and respawn logic. |
| Spell 1 file | Single-target cast with optional trinket splash, MP/HP percentage consumption, raw spell damage, mitigation, and per-target crit rolls. |
| Spell 2 file | Targeted AoE with initial damage, radius falloff, a timed damage effect, and MP/HP percentage consumption. |

The spell source filenames and their internal descriptions should be verified
against the live action bar before action IDs are bound permanently.

## Fidelity gaps to resolve

The simulator can be structurally complete before all values are known, but it
cannot claim game-level damage accuracy until these inputs are supplied or
measured:

1. The 100 by 100 collision grid, legal spawn cells, player start, NPC spawns,
   and objective area.
2. Whether `EntityBase.Distance` is Euclidean, Manhattan, Chebyshev, or another
   metric. Mage currently uses Manhattan distance when the client does not send
   a distance.
3. Movement collision, diagonal behavior, facing, `MoveTowardsBasic`, random
   movement, path selection, and occupied-tile rules.
4. The actual Innie template: max HP/MP, base damage, AC, attack speed/range,
   respawn time, bulk factor, balance cap, and any spell list.
5. Implementations or evaluated level-135 values for `GetEnemyHP`,
   `NPCRecursiveDmg.GetDamage`, player `StrengthFactor` and
   `IntelligenceFactor`, player `CalculateRawSpellDamage`,
   `ApplyAoECalculation`, `Ranges.Mystic`, facing bonus, and `CanHit`.
6. Spell property records: cooldown, range, mana/vita consumption, all optional
   formula overrides, duration, tick interval/percent, and trinket/status state.
   The source defaults would consume 100% of current MP, so assuming defaults is
   especially unsafe.
7. Mechanics for actions `attack`, `castSpell:3`, `castSpell:5`,
   `castSpell:6`, and `castSpell:7`.
8. Exact server event ordering when player actions, regeneration, NPC movement,
   attacks, status ticks, deaths, and respawns share a timestamp.
9. RNG details that affect parity: chance comparison, integer range endpoints,
   and when each roll is consumed.

Unknown mechanics should live in explicit configuration fields or raise a
development-time `NotImplementedError`. They should not be hidden behind guessed
constants. A deliberately simplified mode may use documented approximations,
but it should be named and versioned separately from the fidelity model.

## Proposed package layout

Keep the simulation beside the existing environments so project packaging and
imports continue to work:

```text
Custom_enviornments/
  Simulation_Env/
    __init__.py
    Env_Sim.py              # Gymnasium adapter only
    actions.py              # canonical 11-action enum/list
    config.py               # immutable scenario/mechanics configuration
    state.py                # PlayerState, NPCState, effects, map and events
    engine.py               # reset, advance 200 ms, event ordering
    observation.py          # pure state -> Mage-compatible float32 vector
    rewards.py              # reward components and episode rules
    movement.py             # collision, distance, aggro and NPC movement
    combat.py               # rolls, mitigation, damage, death and regeneration
    spells.py               # translated Mystic spells and timed effects
    scenarios.py            # seeded map/spawn construction
Tests/
  test_sim_environment.py
  test_sim_timing.py
  test_sim_combat.py
  test_sim_observation_parity.py
  fixtures/simulation/      # C# golden inputs/outputs and recorded transitions
```

`Env_Sim.py` should remain thin. Rules belong in pure functions or the engine so
they can be tested without Gymnasium, run in batches later, and optimized without
changing the external API.

## Environment contract

### Actions

Use the exact Mage ordering:

| ID | Action |
|---:|---|
| 0 | `up` |
| 1 | `down` |
| 2 | `left` |
| 3 | `right` |
| 4 | `attack` |
| 5 | `castSpell:1` |
| 6 | `castSpell:2` |
| 7 | `castSpell:3` |
| 8 | `castSpell:5` |
| 9 | `castSpell:6` |
| 10 | `castSpell:7` |

Move this list to one shared module and import it from both Mage and the
simulator. That prevents checkpoint-breaking action drift. Invalid or currently
unimplemented actions need a defined policy: for training, a failed/no-op cast
with an `info` reason is preferable to changing the action-space size. Add an
action mask later if the chosen algorithm consumes one.

### Observations

Preserve the existing order and normalization:

```text
[player_x, player_y, direction, hp_pct, mp_pct, map_id,
 enemy_0_distance, enemy_0_direction, enemy_0_hp_pct, enemy_0_mp_pct,
 ... five nearest enemies total]
```

The result must have shape `(26,)` and dtype `numpy.float32`. Enemy ordering must
be stable: sort by distance and then by entity ID to break ties. Simulated IDs
must remain stable until death/despawn.

Refactor the current parser before reuse. `Env_conditions.parse_observation`
writes three debug files during ordinary parsing, which would dominate a fast
simulation. Extract a pure encoder and make live debug persistence an optional
adapter concern. Test the pure live-world-state and simulated-state encoders
against identical snapshots.

The current observation does not include cooldowns, status effects, or obstacle
geometry. That preserves checkpoint compatibility but makes the process partly
observable. After the parity baseline, define a versioned `SimEnv-v1`
observation rather than silently changing the 26 fields. A recurrent policy is
the least disruptive baseline for hidden cooldown/timer state.

### Reset and step

`reset(seed=..., options=...)` should call `super().reset(seed=seed)`, build a
scenario using `self.np_random`, clear all event queues/timers, and return the
initial observation plus diagnostic `info`.

`step(action)` should perform one 200 ms decision interval:

1. Validate and apply the player action at simulated time `t`.
2. Process every scheduled event with a due time in `(t, t + 200 ms]` in a
   documented stable priority order.
3. Remove or mark dead entities and schedule configured respawns.
4. Advance simulated time and step count.
5. Encode the resulting state.
6. Calculate reward from the previous and resulting states/events.
7. Return `(observation, reward, terminated, truncated, info)`.

Store time as integer milliseconds and cooldown/effect deadlines as absolute
integer timestamps. This represents the source's 900–1100 ms movement jitter,
1.5-second aggro checks, and 2-second regeneration without floating-point timer
drift. Event ordering must be part of the public simulator version because it
can change learned behavior.

### Reward and episode behavior

Start with an explicit `reward_mode="mage_compat"` that preserves Mage's
component keys:

```text
health_state, positioning, damage_taken, damage_dealt, terminal, killed
```

Use simulator damage/death events directly instead of inferring kills when an
enemy ID disappears. Keep the Mage five-kill win condition, player-death loss,
and 256-step time limit for the parity scenario. Map departure and the current
Y-coordinate boundaries only apply if the simulated scenario intentionally
recreates map 53; otherwise they must be scenario configuration.

Snapshot the existing reward behavior in tests before refactoring it. Some
conditions deserve separate review rather than accidental preservation, notably
the `player_hp_pct != 0.5` damage condition and the large Y-position penalty.
Introduce corrected rewards as a new named version so results remain comparable.

`info` should include the current fields plus simulation diagnostics:

```text
current_step, simulation_time_ms, reward_components, episode_outcome, is_win,
action_applied, action_failure_reason, damage_events, death_events,
cooldowns_remaining_ms
```

## Translation rules for C# mechanics

Port formulas literally before simplifying them:

- preserve the order of casts, rounding, clamping, mitigation, and HP mutation;
- use `math.ceil` where C# uses `Math.Ceiling` and explicit integer truncation
  where C# casts `double` to `long`;
- centralize seeded chance and integer rolls instead of calling Python's global
  RNG;
- separate raw damage, crit, block/dodge, AC reduction, and final application;
- represent damage and HP as Python integers, while percentages and multipliers
  remain floats;
- keep spell target selection and AoE entity counts stable and deterministic;
- represent DoTs as scheduled effects with absolute next-tick and expiry times;
- apply NPC death once, clear aggro consistently, and cancel effects whose source
  or target rules require cancellation.

The displayed player factors can seed a temporary measured configuration. They
should not be confused with implementations of the missing server factor
functions. Likewise, port the supplied AC reduction formula, but configure the
reported 48.5% player reduction until its full player-side call path is known.

## Implementation phases and acceptance gates

### Phase 0 — Freeze the compatibility contract and collect missing inputs

- Extract/shared-test the 11 actions and 26 observation fields.
- Record at least one reset snapshot and short live trajectory with actions,
  timestamps, player/enemy state, damage, and deaths.
- Obtain the map collision data, NPC template, spell property records, and the
  missing helper functions or evaluated outputs listed above.
- Create a mechanics manifest that labels every value as source-derived,
  measured, or approximated.

Gate: action and observation golden tests pass, and no unknown value is presented
as exact.

### Phase 1 — Deterministic state, clock, map, and movement

- Implement dataclasses/configuration and an integer-ms event clock.
- Implement a 100 by 100 tile map, bounds/collision, cardinal player movement,
  seeded spawn construction, and stable entity IDs.
- Implement NPC move-speed jitter, aggro checks/radius/drop, pursuit, and the
  known NPC update order.
- Expose a minimal Gymnasium environment with movement actions; combat actions
  return documented failure reasons.

Gate: same seed plus same actions produces byte-equal observations/events;
different seeds vary only configured random elements; movement and timer tests
pass at boundary timestamps.

### Phase 2 — Base combat and lifecycle

- Implement player regeneration, NPC attacks, dodge/block/AC reduction, aggro
  from damage, death, despawn, and configurable respawn.
- Port `CalculateACReduction` and the supplied NPC mitigation functions with
  golden numeric cases evaluated from C#.
- Add the base `attack` only after its range, cadence, and damage path are known.

Gate: deterministic combat traces match C# golden results for crit, dodge,
block, normal damage, death, and simultaneous-event cases.

### Phase 3 — Mystic spells and timed effects

- Port spell 1, including target validation, consumption, mitigation, crit, and
  optional splash behavior.
- Port spell 2, including radius selection, AoE scaling, distance falloff, DoT
  scheduling, refresh/stack semantics, and status multipliers.
- Add cooldown/range/mana validation and define failed-cast behavior.
- Add actions 3/5/6/7 only when their mechanics are available; keep their IDs
  reserved until then.

Gate: spell golden cases cover zero/one/multiple targets, edge-of-range targets,
insufficient MP, cooldown boundaries, crits, AoE counts, DoT expiry, and deaths.

### Phase 4 — Mage reward parity and Gymnasium compliance

- Connect pure observation and reward code to the engine.
- Add explicit termination versus time-limit truncation.
- Run `gymnasium.utils.env_checker.check_env`.
- Add regression tests for Mage-compatible action, observation, reward, info,
  reset, and terminal behavior.

Gate: existing policy networks accept simulator observations/actions without
shape or ordering changes, and all current environment tests continue to pass.

### Phase 5 — Training integration and parallelism

- Add a trainer environment factory/CLI option (`live` or `simulation`) instead
  of replacing Mage imports ad hoc.
- First run the existing PPO loop with one simulated environment to establish
  functional parity.
- Then support multiple independent environments. The current PPO trainer
  allocates buffers using `num_envs` but instantiates and steps only one `Mage`,
  so vectorization requires a real `SyncVectorEnv`/`AsyncVectorEnv` path and
  batched reset/terminal handling.
- Benchmark engine steps separately from neural-network training and profile
  before considering Cython, Numba, or a C++ core.

Gate: short seeded training runs complete, reset correctly after both termination
and truncation, save/load checkpoints, and run multiple environments without
shared state or global RNG.

### Phase 6 — Fidelity validation and transfer

- Replay recorded live action sequences in simulation and compare positions,
  HP/MP, target IDs, damage, deaths, and timer events step by step.
- Track error by mechanic rather than only total return.
- Randomize only measured uncertainty ranges (NPC timing, damage parameters,
  spawns); keep an exact deterministic profile for regression tests.
- Evaluate simulator-trained checkpoints in live Mage with exploration off.
- Add discrepancies as fixtures, correct the model, and repeat. Limited live
  fine-tuning comes after simulator behavior is stable.

Gate: define acceptable per-mechanic error thresholds from live traces and meet
them on held-out recordings. High simulator reward alone is not a fidelity test.

## First implementation slice

The smallest useful milestone is a movement-and-survival environment, not the
full spell system:

1. Share/freeze actions and extract the pure observation encoder.
2. Create state/config/engine modules with the 200 ms seeded clock.
3. Load a simple 100 by 100 collision map and spawn one Innie.
4. Implement cardinal movement, aggro radius 4, 0.9–1.1-second NPC pursuit,
   aggro drop beyond 18, and configurable contact attacks.
5. Add player HP/MP and 2-second regeneration.
6. Return Mage-compatible observations and a simple versioned survival reward.
7. Validate the Gym API, determinism, timing, and observation parity.

This slice tests the architecture and enables throughput measurement while the
missing damage, spell properties, and map data are collected. It avoids baking
guesses into the combat model that the policy would later exploit.
