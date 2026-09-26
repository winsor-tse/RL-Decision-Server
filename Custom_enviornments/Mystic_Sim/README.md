# Mystic simulator: Phases 0 through 5

## Play the simulator with Pygame

Double-click **`Play_Mystic_Sim.bat`** at the repository root, or run:

```powershell
.\RL_venv\Scripts\python.exe -m pip install -r requirements-viewer.txt
.\RL_venv\Scripts\python.exe -m Custom_enviornments.Mystic_Sim.viewer
```

The optional display uses [pygame-ce](https://pyga.me/docs/), which imports as
`pygame`. The headless environment does not import or require it. The viewer
calls the existing Gym environment; movement, targeting, damage, regeneration,
cooldowns, Acid effects, and respawns use the same simulation code as training.

The window starts paused. Press **Space** to begin.

| Control | Action |
|---|---|
| WASD / arrow keys | Move; hold to repeat every 200 simulated milliseconds |
| 1 / 2 / 3 | Arcane Blast / Acid Cloud / Tempest Inferno; hold to repeat |
| Space | Pause/resume |
| R / Shift+R | Restart same seed / restart with next seed |
| Tab | Toggle follow camera and full-map overview |
| Mouse wheel | Zoom follow camera |
| - / + | Change playback speed, 0.25x through 4x |
| C / B | Toggle casting rectangle / spawn boxes |
| Escape | Close |

One action executes per decision; held spells take priority over movement.
Targets are selected automatically by the simulator. Blue is the player, amber
is an idle Innie, red is aggro, and green rings indicate Acid. The gold ring
marks the nearest eligible target. Dead Innies display respawn countdowns.
Terrain marks are visual references; this scenario disables terrain collision.
Melee remains disabled as in the baseline. Idle time uses its nonmutating attack
action (4) to advance the clock without adding a ninth training action.

Default **free play** removes the five-kill and 256-step stopping points so you
can explore and watch respawns; player death still ends play. To use the exact
training episode rules:

```powershell
.\RL_venv\Scripts\python.exe -m Custom_enviornments.Mystic_Sim.viewer --training-rules --seed 42
```

Rendering is 60 FPS with independent 200 ms simulation decisions. Focus loss
pauses the viewer. Excess wall-clock lag is capped so returning to a stalled
window does not fast-forward a long combat sequence.

For a display-free smoke test, including a PNG render:

```powershell
.\RL_venv\Scripts\python.exe -m Custom_enviornments.Mystic_Sim.viewer --smoke-test --screenshot viewer.png
```

This package defines contracts, immutable map/configuration data, and seeded reset.
Importing `Custom_enviornments.Mystic_Sim` registers `YugenSaga/MysticSim-v0`.
The environment declares eight actions and 26 float32 observation values.
`reset()` returns a fully initialized world and observation. `step()` advances
200 simulated milliseconds, applying movement and dispatching scheduled NPC
movement/aggro, combat, regeneration, death, and respawn events. Rewards use the
versioned `combat_reward_v1` training profile by default. The three spells are active,
with rectangular casting eligibility and per-caster Acid refresh behavior.

```python
import gymnasium as gym
import Custom_enviornments.Mystic_Sim

env = gym.make("YugenSaga/MysticSim-v0")
observation, info = env.reset(seed=42)
world = env.unwrapped.world
assert len(world.monsters) == 80
assert observation.shape == (26,)
env.close()
```

## Phase 2 clock, movement, and diagnostics

Pass `trace=True` at construction to include the current step's event/RNG/action
trace in `info["trace"]`. Trace mode does not consume additional random draws.
Each entry includes simulated time, kind, entity ID, and relevant positions,
facing, result, or RNG purpose/range/value. Traces are kept in memory per step.

```python
env = gym.make("YugenSaga/MysticSim-v0", trace=True)
env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(0)  # up
assert info["simulation_time_ms"] == 200
env.close()
```

The scheduler uses `(due_ms, enqueue_sequence)` ordering, cancellation tokens,
and NPC/effect generations. It executes deadlines inside each 200 ms interval
without rounding them to decision boundaries. A per-timestamp event limit catches
same-time rescheduling loops. The player action runs before advancing the queue;
events through the interval's end are processed, including newly scheduled events.

NPC pursuit follows `NPC.cs` candidate order, eager coin draw, weighted axis
choice, fallback branches, and blocked facing/timer behavior. Spawn return uses
the NPC's original sampled spawn cell. `OutsideSpawnArea` preserves the C#
inclusive X+W/Y+H return boundary, separate from half-open spawn sampling.
Idle movement is one inclusive 0..3 draw in server direction order. Successful
moves update occupancy immediately, so later equal-time events see the new cell.
The sampled movement interval is never rerolled during a life.

Aggro checks use Euclidean radius 4 and 1500 ms deadlines. Dead/invalid targets
and targets farther than 18 tiles are dropped. Player movement also triggers
the acquisition hook shown in `NPCScript.OnEntityMoved`. Damage acquisition is
applied by the combat pipeline through `engine.add_damage_aggro`.

Timing conventions pending fuller server traces:

- After an acquisition check, the next check is 1500 ms later; the supplied
  fragment does not show an `AggroCheckTimer.Reset` call.
- Adjacent facing leaves the movement deadline unchanged, as C# does. Expired
  timers are reconsidered 200 ms later to avoid zero-time polling loops.
- A successful aggro move partially resets attack readiness to now plus
  `min(attack_ms / 2, attack_ms - 500)`, or 50 ms when `attack_ms <= 500`.
  The confirmed 1000 ms interval therefore becomes ready at now + 500 ms.
  Half-millisecond deadlines round up to the next integer millisecond. Blocked
  moves, adjacent facing, and movement without aggro leave attack readiness alone.
  After attacking, the normal full attack interval applies.
- Effect timing hooks use ticks before expiry, with no tick at the expiry
  timestamp. Exact server Acid Cloud tick/expiry semantics remain unconfirmed.
- The long-idle return jump refuses an occupied destination, enforcing the
  developer's entity-collision rule even on that source branch.

NPC attacks now apply dodge, block, facing, AC mitigation, and HP loss. Death
records once, frees occupancy, clears aggro and invalid effects, and cancels
pending entity events. An NPC respawns 50000 ms after death in its original box,
retaining its ID and sampling one new movement interval for its next life. A
full box retries at the next decision boundary without consuming placement RNG.
Acid effects now apply their saved tick damage.
No on-move status effects or chain-aggro groups are active in this baseline.

Action 4 reports `gear_disabled` by default. To enable facing-tile melee, pass
`ScenarioConfig(gear_enabled=True, gear=GearConfig(weapon_damage=100, attack_ms=1000))`
as `config`; these example gear values are explicit inputs, not baseline stats.
Actions 5, 6, and 7 cast Arcane Blast, Acid Cloud, and Tempest Inferno.
All valid actions still advance time. Invalid actions reject without advancing.
Episodes truncate after 256 steps and require reset before another step. Step
info reports `combat_implemented=True` and `reward_implemented=True`, with
structured `damage_events`, `death_events`, `respawn_events`, and `cast_events`
for that step. Spell support is reported as `spells_implemented=True`.

## Phase 3 regeneration and scaling

Regeneration is configurable through `ScenarioConfig` and `TimingConfig`.
`TimingConfig(regen_enabled=False)` disables healing. The default 2000 ms tick
restores 226 HP and 1664 MP per tick, using the equivalent per-second rates
`PlayerConfig.hp_regen_override=113.0` and `mp_regen_override=832.0`,
clamped to maxima. Dead players do not regenerate. Change `regen_ms` to change
the interval; amounts scale with its duration. For formula-derived rates, supply
`base_hp`/`base_mp` and set the corresponding override to `None`.

`scaled_calcs.py` derives Innie combat stats from effective level and bulk factor.
`combat.py` contains pure stat, crit, dodge, block, mitigation, AoE, and regeneration
functions. Tests cover the level-150 formula boundary and controlled RNG branches.

## Phase 4 casting and effects

`targeting.py` selects the nearest living, nonimmune monster by Euclidean
distance, then entity ID. Acid and Tempest fall back to the caster when none
exists. AoE targets are processed by entity ID, with full Tempest targets before
partial targets. `cooldowns.py` checks fixed HP/MP costs and absolute slot/family
deadlines. `spells.py` performs pure calculations with an explicit crit input;
the engine owns RNG draws and mutations.

| Spell | Fixed MP | Later remaining-MP charge | Cooldown | Radius |
|---|---:|---:|---:|---|
| Arcane Blast | 500 | 33% | 3500 ms | Single target |
| Acid Cloud | 1000 | 33% | 5000 ms | At most 4.25 |
| Tempest Inferno | 750 | 20%, only with targets | 1750 ms | At most 1.5 |

Damage uses MP after the fixed charge. Percentage charges use midpoint-to-even
rounding. First cooldown-ready decisions after a cast at zero are 3600, 5000,
and 1800 ms. Rejected casts leave resources, cooldowns, effects, and combat RNG
unchanged; a Gym step still advances the world and can dispatch existing events.

Acid initial damage applies AoE scaling before mitigation, then distance falloff.
Its per-target effect snapshots AoE-scaled tick damage and crit. Ticks bypass new
crit/block/AC calculations, following the supplied callback. The existing
scheduler convention gives ticks at +1000 through +5000 ms and expiry at +6000
ms, without a tick at expiry. Death removes effects. IDs are `acid:<caster ID>`
per target, with distinct generations for later applications.

Casting eligibility requires `abs(target.x - caster.x) <= 16` and
`abs(target.y - caster.y) <= 10`, including the corners. Euclidean distance ranks
eligible targets and determines AoE radii; it does not define casting eligibility.
Acid refreshes the same caster's effect on each affected target, replacing saved
damage/crit and restarting the tick interval and six-second duration. Old queued
ticks and expiry are cancelled and generation checks prevent stale callbacks.
Different caster IDs retain independent effects on the same target. The baseline
environment still controls one player; effect identity and scheduling support
independent sources.

`SpellRules` explicitly represents disabled Arcane Bomb, Sunburnt, and Tempest
Meteor modifiers; attempts to enable these unsupported modifiers reject during
configuration. Numeric fixtures and unit tests require no C# compiler or server.

## Reset and state

Construction loads the bundled `data/map53.json` once, validates it, and converts
pixel rectangles into immutable tile-coordinate spawn boxes. The bundled file
is byte-identical to the tracked map fixture; runtime does not depend on Tests
or the ignored C# source directory. An explicit `map_path` may be passed for
another copy of the same validated baseline map.

`ScenarioConfig` contains frozen player, Innie, spell, timing, and reward
configuration. The default profile is `map53_open_entities_v1`. Reset options
accept that environment's fidelity `profile` and an optional `reward_profile`;
unknown options fail explicitly. Reward selection is local to the episode and
does not modify the constructor configuration or other environments.
Map properties retain cap 68, while Innie stats use the confirmed level-150
override. Terrain is ignored, entity occupancy is enforced, and fixed template
5399 is excluded.

Reset samples player X then Y in inclusive ranges 45..55 and 40..50, then visits
boxes in Tiled object-ID order and members in local-index order. For each NPC it
draws X then Y until unoccupied and then draws its movement interval once from
900..1100 ms inclusive. A full rectangle fails before consuming RNG. IDs are
player=1 and monsters=2..81, with each monster retaining its box and member index
for later respawn. Facing defaults to up for player and NPCs; initial facing is
a simulator reset convention, not a sampled game mechanic.

Every reset builds fresh mutable state, occupancy, cooldown dictionaries,
effects, and events. Same seed/config/map produces identical state; reset without
a seed continues the environment RNG. Each NPC starts with move, attack, and
aggro deadlines at its sampled interval, 1000 ms, and 1500 ms respectively.
One NPC-update event is recorded at the earliest of those deadlines. Player
regeneration is recorded at 2000 ms. Events are ordered by time and enqueue
sequence; the scheduler executes their Phase 3 gameplay mutations and reschedules
live entities as needed.

`info` includes the seed, profile, spawns, movement intervals, selected entity
IDs, balance-cap provenance, event count, and contract metadata. Reset and
observation encoding perform no file writes, socket calls, rendering, or sleep.
Runtime observation ranks living monsters by Manhattan distance and entity ID,
emits five blocks, and zero-pads unused blocks. Mechanics use a separate
Euclidean distance helper; server and observation direction codes are explicitly
converted. The supplied full-state golden vector is reproduced exactly.

## Phase 5 rewards and episode rules

`rewards.py` and `diagnostics.py` keep `env.py` focused on the Gym five-tuple.
The eight-action space and 26-value float32 observation contract are unchanged.
Default episodes terminate on death or five kills and truncate at 256 steps.
Death takes precedence over a simultaneous kill goal; termination takes
precedence over a simultaneous time limit. `episode_end_reason` distinguishes
`player_death`, `kill_goal`, `step_limit`, and optional `y_boundary` truncation.

`combat_reward_v1` uses actual damage/death events, including DoT damage and
overkill capped at remaining HP. Regeneration cannot mask damage taken, and
moving an enemy outside the observation does not earn a kill reward.

| Dashboard component | Default training calculation |
|---|---|
| `health_state` | -0.001 per decision (time cost) |
| `positioning` | 0 |
| `damage_taken` | -player damage / player maximum HP |
| `damage_dealt` | Sum of player damage / each enemy's maximum HP |
| `terminal` | -5 for death; 0 otherwise |
| `killed` | +1 per confirmed player kill |

The returned reward is exactly the sum of these six components. Weights, kill
goal, and time limit are configurable through frozen `RewardConfig` fields.
`y_bounds=None` disables Y restrictions. For a task needing them, set inclusive
`y_bounds=(25, 99)`; leaving the interval truncates without a death penalty.

`legacy_reward_v0` snapshots the live health thresholds, blocked-move penalty,
distance shaping, signed player HP-delta quirk, and coefficients (25 for enemy
damage, 10 per kill, -100 for death). Enemy damage and kills use explicit events
instead of live disappearance heuristics. Legacy alone retains healing rewards
and the `hp != 0.5` branch for comparison. Its old Y shaping is opt-in via
`legacy_y_penalty_below=31`; it is separate from episode truncation.

```python
observation, info = env.reset(seed=42, options={"reward_profile": "legacy_reward_v0"})
```

A later reset without options restores the configured default profile. Reset and
step diagnostics include seed, profiles, clock/step, action/result, selected
target, components, kills, cooldowns, active effects, and event ledgers. The
selected target on step is captured at action time (None for movement/no target).
Returned dictionaries are detached from engine state. No fixed-fixture reset
option is introduced; seeded reset is the supported reproducible initialization.

## Actions and artifact compatibility

`actions.Action` and `actions.ACTIONS` define the canonical order:
up, down, left, right, attack, castSpell:1, castSpell:2, castSpell:3.
The schema is `mystic-eight-v1`; the observation schema is `mystic-26-v1`.
New eight-action demonstrations use `mystic/BC-v1`.

Live Mystic and MysticBC still use their legacy 11-action schemas, now imported
from this central module. Existing trainers/recorders retain their defaults.
Switching their policy heads and Minari producers is deferred to the plan's
training migration. Do not label an 11-action dataset as BC-v1.

Future eight-action checkpoint writers must serialize
`artifacts.checkpoint_envelope(model.state_dict())`; readers must call
`artifacts.checkpoint_state(checkpoint)` before loading weights. This rejects
unversioned legacy weights and mismatched action order, including a BC movement
permutation with the same number of outputs. It does not migrate policy weights.
Dataset producers must embed `actions.contract_metadata()` and use a new ID.

## Explicit legacy BC remapping

For an exported JSON dataset with top-level `dataset_id`, legacy `actions` label
list, and `episodes` containing integer `actions` arrays:

```powershell
.\RL_venv\Scripts\python.exe -m Offline.remap_mystic_bc legacy.json converted.json
```

The old BC indices 0,1,2,3 map to 0,2,3,1. Indices 4..7 keep their meaning.
Removed spell indices 8..10 reject the whole conversion. No transitions are
dropped, because that would incorrectly pair observations across time. Input
data and existing destination files cannot be overwritten. The result contains
the new schema metadata and dataset ID. This operates on JSON exports; native
Minari export/import and publication belong to the later training phase.

## Fixtures and validation

`Tests/fixtures/simulation` contains byte-for-byte copies of the supplied map and
full-state capture, with source paths and SHA-256 hashes. Tests do not depend on
the ignored `Simulation/Source-Code` directory. The expected observation and
selected IDs are stored separately. The pure
`Env_conditions.encode_observation` writes no files; the existing live
`parse_observation` wrapper retains its diagnostics and stable payload-order ties.

`map_loader.load_map53(path)` validates the supplied baseline export and returns
raw map data. `load_map_definition(path)` normalizes immutable geometry, and
`scenarios.build_scenario` constructs episode state. Changes to map dimensions,
properties, spawn layout, or quantities fail
explicitly so a changed map requires review. Terrain remains disabled in the
baseline even though the blocked layer is validated.

`mechanics_manifest.yaml` records baseline parameters and provenance. It preserves
the JSON cap 68 separately from the confirmed effective level 150, the corrected
50-second respawn, one movement-speed draw per life, and fixed MP deductions
before spell calculations.

```powershell
.\RL_venv\Scripts\python.exe -m unittest Tests.test_mystic_sim_contracts
.\RL_venv\Scripts\python.exe -m unittest Tests.test_mystic_sim_reset
.\RL_venv\Scripts\python.exe -m unittest Tests.test_mystic_sim_timing
.\RL_venv\Scripts\python.exe -m unittest Tests.test_mystic_sim_combat
.\RL_venv\Scripts\python.exe -m unittest Tests.test_mystic_sim_spells
.\RL_venv\Scripts\python.exe -m unittest Tests.test_mystic_sim_rewards
```

Phase 3 validation uses Python unit tests and saved numeric expectations. No C#
compiler, runnable server project, or source files are required to run the tests.
