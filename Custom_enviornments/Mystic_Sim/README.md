# Mystic simulator: Phases 0 through 2

This package defines contracts, immutable map/configuration data, and seeded reset.
Importing `Custom_enviornments.Mystic_Sim` registers `YugenSaga/MysticSim-v0`.
The environment declares eight actions and 26 float32 observation values.
`reset()` returns a fully initialized world and observation. `step()` advances
200 simulated milliseconds, applying movement and dispatching scheduled NPC
movement/aggro events. Rewards are currently zero; combat training requires the
subsequent combat, spell, and reward phases.

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
exposed as `engine.add_damage_aggro(npc_id, attacker_id)` for Phase 3.

Timing conventions pending fuller server traces:

- After an acquisition check, the next check is 1500 ms later; the supplied
  fragment does not show an `AggroCheckTimer.Reset` call.
- Adjacent facing leaves the movement deadline unchanged, as C# does. Expired
  timers are reconsidered 200 ms later to avoid zero-time polling loops.
- The missing `Timer.Reset` implementation is modeled with a negative reset
  value meaning a future timer origin. A successful aggro move at the baseline
  1000 ms attack interval therefore sets attack readiness to now + 1500 ms.
- Effect timing hooks use ticks before expiry, with no tick at the expiry
  timestamp. Actual Acid Cloud tick/expiry semantics must be validated in Phase 4.
- The long-idle return jump refuses an occupied destination, enforcing the
  developer's entity-collision rule even on that source branch.

Attacks, regeneration, effect ticks, and respawn deadlines are trace-only hooks:
they do not apply damage, healing, or resurrect entities yet. An eligible attack
records its deadline and resets the attack timer. `schedule_respawn` accepts only
a dead NPC and schedules its notification 50000 ms later; it does not perform
death cleanup. `schedule_effect` records tick/expiry deadlines with generation
checks. Phase 3/4 supply their gameplay mutations and cancellation lifecycle.
No on-move status effects or chain-aggro groups are active in this baseline.

Action 4 reports `gear_disabled`; spell actions report `spell_not_implemented`.
All valid actions still advance time. Invalid actions reject without advancing.
Episodes truncate after 256 steps and require reset before another step. Step
info explicitly reports `combat_implemented=False` and `reward_implemented=False`.

## Reset and state

Construction loads the bundled `data/map53.json` once, validates it, and converts
pixel rectangles into immutable tile-coordinate spawn boxes. The bundled file
is byte-identical to the tracked map fixture; runtime does not depend on Tests
or the ignored C# source directory. An explicit `map_path` may be passed for
another copy of the same validated baseline map.

`ScenarioConfig` contains frozen player, Innie, spell, timing, and reward
configuration. The default profile is `map53_open_entities_v1`. Reset options
accept only that environment's profile; unknown options fail explicitly.
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
sequence; Phase 2 executes/reschedules them, while damage, death, and actual
respawn remain later-phase work.

`info` includes the seed, profile, spawns, movement intervals, selected entity
IDs, balance-cap provenance, event count, and contract metadata. Reset and
observation encoding perform no file writes, socket calls, rendering, or sleep.
Runtime observation ranks living monsters by Manhattan distance and entity ID,
emits five blocks, and zero-pads unused blocks. Mechanics use a separate
Euclidean distance helper; server and observation direction codes are explicitly
converted. The supplied full-state golden vector is reproduced exactly.

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
```
