# Plan: Yugen Saga Mystic simulation in Python

## Current scope

The single map profile is `map53`, used by both the viewer and headless training.
Terrain and entity collision are always enabled for movement, pathfinding,
spawning, and respawning, alongside the outer 0..99 coordinate bounds. No spell
line-of-sight rule is introduced. Both reward profiles default to
`legacy_y_penalty_below=31` and `legacy_y_penalty_above=85`: subtract
100 * (31 - Y) below 31, or 100 * (Y - 85) above 85. Simulator episodes truncate
at Y <= 29 or Y >= 87 (inclusive continuing bounds 30..86); the final step retains
its penalty. These task rules do not change live Mystic. Old open/blocked
map-profile names are removed.

The next priority is to train PPO on the current Mystic simulator and validate
that the same policy contract works with live Mystic through the existing ZMQ
bridge. Phase 6 establishes PPO training, Phase 7 makes local vector training
fast and reliable, and Phase 8 implements/tests the live adapter, payload contract,
and real sim-to-game gap. No live contract tests or game access are required to
complete Phases 6 or 7. Behavior
cloning, Minari datasets, offline learning, and dataset migrations are deferred;
they must not block PPO training or live parity work.

Build a headless Python `gymnasium.Env` for the Mystic combat loop using the C#
fragments in `Simulation/Source-Code` as the mechanics reference. The first
version includes only:

- cardinal movement: `up`, `down`, `left`, `right`;
- `attack`: a deliberately simple close-range basic-damage action;
- `castSpell:1`, `castSpell:2`, and `castSpell:3`, translated from their three
  supplied C# implementations;
- Innie movement, aggro, attacks, damage, death, and respawn;
- the existing Mystic observation layout and Gymnasium API.

Actions for spell slots 5, 6, and 7 are out of scope. The simulator therefore
has eight actions:


| ID | Action        | Initial behavior                                           |
| -: | ------------- | ---------------------------------------------------------- |
|  0 | `up`          | Move one tile north if legal.                              |
|  1 | `down`        | Move one tile south if legal.                              |
|  2 | `left`        | Move one tile west if legal.                               |
|  3 | `right`       | Move one tile east if legal.                               |
|  4 | `attack`      | Damage one close target using the configured basic damage. |
|  5 | `castSpell:1` | Execute`Spell1_Arcane Blast_single_target_spell.txt`.      |
|  6 | `castSpell:2` | Execute`Spell2_Acid Cloud_large AoE.txt`.                  |
|  7 | `castSpell:3` | Execute`Spell3_Tempest Inferno.txt`.                       |

Before a simulator-trained policy is used with the live game, the selected live
Mystic adapter must expose the same shared eight-action definition. Preserve the
legacy live default for existing models; MysticBC migration is deferred. Changing from 11 to 8
actions changes the policy output layer, so existing 11-action checkpoints will
need retraining or an explicit output-head migration. The 26-value observation
shape can remain unchanged.

The C# should be translated into native Python. These files are partial server
sources rather than a compilable module, so embedding a .NET runtime would not
provide their missing entity, map, spell-property, and status-effect types.

## Source-derived scenario facts


| Area                     | Behavior or value                                                                                                                                                 |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Map                      | Map 53 is a 100 by 100 Tiled map with 32-pixel tiles. `map53.json` is the machine-readable source; `Map_Layout.png` is the visual reference. The `map3471` example state remains a parser fixture only. |
| Player decision interval | 200 ms per action.                                                                                                                                                |
| Player profile           | Level-135 Mystic.`Current_Player_Stats.txt` lists HP 9,463, MP 15,108, AC 3,292, and displayed combat factors.                                                    |
| Regeneration             | HP and MP regenerate every 2 seconds.                                                                                                                             |
| Innie movement           | At spawn, each Innie samples `Roll(900, 1100) / 1000.0`, inclusive: a 900–1,100 ms base movement interval in 1 ms increments. That sampled interval remains constant for the NPC's entire life and is sampled again only when it respawns after death. |
| Innie aggro              | Aggressive radius 4, new-aggro check every 1.5 seconds, and target removal beyond distance 18.                                                                    |
| Innie combat             | Confirmed balance cap 150, attack speed 1 second, and Euclidean attack radius 1. Radius 1 reaches only cardinal neighbors; radius 1.5 would also reach diagonals. |
| NPC update order         | Drop invalid aggro, check new aggro, move/face, then attack. RedBot additionally attempts configured spells.                                                      |
| Innie scaling            | `ScaledCalcs.GetEnemyHP`, `NPCRecursiveDmg.GetDamage`, and the RedBot scaling path are now supplied.                                                              |
| Regeneration timing      | Ten environment steps at the 200 ms decision interval.                                                                                                            |
| World distance           | `EntityBase.Distance` is Euclidean for attack/spell radii and aggro. Casting eligibility uses an inclusive rectangle: abs(dx) <= 16 and abs(dy) <= 10. |
| Observation distance     | The client-supplied distances in the fixture equal Manhattan distance; Mystic also uses Manhattan as its fallback.                                                |
| Spawn boxes              | `map53.json` contains 40 non-fixed 10 by 10 template-5300 boxes with two Innies each, for 80 baseline Innies. A dead Innie respawns at a random unoccupied point in its own box after 50 seconds. |
| Collision scope          | The JSON blocked layer contains 5,088 marked cells, but terrain blocking is enforced. Baseline legality checks map bounds, terrain, and entity occupancy. |
| Mystic ranges            | The exact`Ranges.Mystic` formula is supplied in `TimeFrame_Other_Details.txt`.                                                                                    |
| Spell records            | Exact properties are supplied for Arcane Blast (416), Acid Cloud (417), and Tempest Inferno (418).                                                                |

Use integer simulated milliseconds. Cooldowns, move timers, regeneration,
status ticks, and respawns should store absolute due times so the simulator does
not sleep or accumulate floating-point timer drift.

Do not use one generic distance helper. The simulator needs two named metrics:

- `euclidean_distance` for server mechanics (`EntityBase.Distance`, attack
  radius, spell radius, and aggro comparisons);
- `observation_distance` using Manhattan distance to reproduce the world-state
  payload and Mystic observation.

Casting eligibility separately checks both coordinate deltas against the
inclusive 16-X/10-Y rectangle; it is not a distance metric.

## Developer-answer status


| Question                                | Status after source review | Integration decision                                                                                                                                                                                                         |
| --------------------------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 100x100 map and spawn layout            | Resolved for baseline      | Parse dimensions, tile size, properties, and NPC object rectangles from `map53.json`. Use the 40 template-5300 boxes and exclude the fixed template-5399 NPC from the Innie-only scenario. Do not enable the blocked tile layer yet. |
| Player start and objective              | Resolved for baseline      | Sample the player from `x=50+/-5`, `y=45+/-5`, rejecting occupied cells. The episode objective is five Innie kills; no separate objective-area geometry is required for v0. |
| `EntityBase.Distance` metric            | Resolved                   | Port Euclidean distance exactly for mechanics; retain Manhattan distance for Mystic observations.                                                                                                                            |
| NPC movement, facing, path choice       | Resolved for baseline      | Port `NPC.cs`. A candidate cell is legal when it is inside map bounds and contains neither the player nor another living NPC. Terrain blocking is enforced. |
| Innie combat template                   | Resolved for baseline      | Use template 5300, balance cap 150, bulk 0.5, HP 274,599, MP 0, AC 2,250, toughness 15, raw damage 3,549, one-second attacks, one sampled 0.9–1.1-second movement interval per life, radius 1, 50-second respawn, and no NPC spells. |
| Player/spell shared formulas            | Resolved for baseline      | Port`Player.cs`, `Entitybase.cs`, `BaseSpell.txt`, and the supplied Mystic range formula.                                                                                                                                    |
| Spell properties                        | Resolved                   | Use the supplied records for spell IDs 416, 417, and 418.                                                                                                                                                                    |
| Event scheduling                        | Partly resolved            | `EventHandler.cs` confirms timestamp-priority queues, due checks against the server clock, and serial `CheckAndHandle` execution per map queue. Equal-time ordering is not defined by the supplied queue implementation, so the simulator will use a deterministic enqueue sequence as its documented tie-break. |
| RNG distribution/endpoints              | Resolved                   | Port inclusive`Roll` and integer-threshold `RollChance`; preserve source draw order per branch.                                                                                                                              |

## Map and spawn layout

Use `map53.json` as the authoritative geometry/object input and
`Map_Layout.png` as a review aid. The JSON is a finite orthogonal Tiled 1.8 map:

- width and height: 100 by 100 tiles;
- tile width and height: 32 by 32 editor pixels;
- tile layers: `layer0`, `layer1`, `layer2`, and `blocked`;
- object layers: `data` and `render`;
- map name: `Severed Space`, map ID 53 in the environment configuration;
- 40 non-fixed NPC objects for template 5300, each 320 by 320 pixels, with
  `quantity=2`;
- one fixed template-5399 object at tile `(50, 21)`, which is outside the
  Innie-only training scenario.

Dividing Tiled object coordinates and sizes by 32 yields exact tile rectangles.
The template-5300 boxes start at X values `10, 20, ..., 80` and Y values
`31, 41, 51, 61, 71`; every box is 10 by 10. This produces 40 boxes and 80
Innies at reset. The loader should derive these values from JSON and validate
them rather than hard-code a second copy.

For the first simulator:

- parse the map and object properties from JSON, but ignore the `blocked` layer
  when evaluating movement;
- build a `SpawnBox` from every non-fixed template-5300 NPC object;
- sample the player uniformly from inclusive X range 45 through 55 and Y range
  40 through 50; use deterministic rejection sampling for occupied cells;
- sample each Innie uniformly from an unoccupied cell in its own box at reset
  and again after its respawn timer;
- reject moves and spawns that collide with the player or another living NPC;
- process spawn boxes and entities in stable object-ID/entity-ID order so seeded
  resets do not depend on dictionary ordering;
- keep spawn boxes as scenario data, separate from movement/combat logic;
- store the original map image beside any manually transcribed box-coordinate
  fixture so the transcription can be reviewed;
- add blocked cells later without changing the environment API.

The JSON map property says `balanceCap=68`, while the developer separately
confirmed 150 and the captured Innie HP of 274,599 matches the level-150 formula
with bulk factor 0.5. Preserve both facts in configuration provenance. The
baseline scenario uses `npc_effective_level_override=150`; the raw JSON value
remains available for a later map-exact profile. Do not silently replace the
parsed property.

The supplied Innie template row describes template 5300 as level 55, body 68,
experience 8,150, base HP 47,430, base AC 1,020, base toughness 6, and template
damage 1. The developer-confirmed respawn time is 50 seconds. The map's balance
cap then overrides the combat values described below. The template has no
cast-spell property, so the baseline Innie spell list is empty.

## What `Example_Full_State.txt` tells us

The file is a golden fixture for Mystic parsing, not a clean initial scenario.
It contains:

- top-level current player ID 7 at `(15, 59)`, facing left on `map3471`;
- current player HP and MP at approximately 50%, but with debug-scale maxima of
  roughly 500 million;
- 41 monsters named Innie;
- two player entries in `entities`, one current and one other player;
- a monster whose ID is also 7, so IDs are not globally unique across the
  top-level player and monster collection in this payload;
- Innie max HP 274,599.

The existing Mystic parser produces this exact `float32` observation:

```text
[15, 59, 2, 0.5, 0.5, 3471,
 10, 3, 1, 0,
 14, 3, 1, 0,
 18, 3, 1, 0,
 26, 3, 1, 0,
 26, 3, 1, 0]
```

The selected monster IDs are `80, 75, 40, 84, 85`. They are the five nearest
monsters, and all are to the player's right. The tie at distance 26 currently
inherits payload order. The simulator should use a documented stable tie-break,
preferably `(distance, entity_id)`, and the golden test should make that choice
explicit.

`parse_observation` correctly filters to `type == "monster"`. In contrast,
`parse_entity_state` currently tracks every non-current entity, including the
other player, and returns 42 entries for this fixture. The simulator will contain
only the controlled player and monsters, so its reward entity-state data should
be built directly from monster state. A future refactor of the live parser must
not silently change Mystic reward behavior without regression tests.

The full-state player is not the level-135 combat profile from
`Current_Player_Stats.txt`; use the latter for simulation defaults. Preserve the
full state only as an input/parsing fixture.

## Testing terminology: fixtures and golden values

A **fixture** is a saved, stable test input. Examples in this project are
`Example_Full_State.txt`, `map53.json`, a fixed initial world state, or a recorded
sequence of actions and RNG draws. A fixture lets a test run the same case every
time.

A **golden value** is the expected output for a fixture, taken from authoritative
C# execution, an accepted live-game capture, or a developer-confirmed result. A
**golden test** runs the fixture and asserts that Python produces that exact
output. Current examples include:

- the 26-number observation produced from `Example_Full_State.txt`;
- 40 template-5300 spawn boxes and 80 Innies parsed from `map53.json`;
- Innie maximum HP 274,599 for effective level 150 and bulk factor 0.5;
- future captured damage, cooldown, movement, and spell results paired with the
  exact input state and RNG draws that produced them.

These are regression-test references, not training examples or tunable balance
values. Approximate displayed values such as “strength factor about 5.04” are
reference checks, not exact golden values, until captured with full precision.

## Scaled calculations now available

Port `ScaledCalcs.txt` literally into a pure Python mechanics module:

- `expected_regression_dps(level)`;
- `get_enemy_hp(level)`;
- `NPCRecursiveDamage.get_damage(level)` with the same lookup growth and
  recurrence order.

The fixture's Innie maximum HP is useful evidence about the scenario. With the
RedBot default `bulkFactor = 0.5`, C# truncation gives:

```text
get_enemy_hp(150) * 0.5 -> 274599
```

That exactly matches the full state. The developer confirmed a balance cap of
150. `RedBotNPCScript.OnCreated` therefore replaces the template level with 150
even though the documented player is level 135. Configure these separately as
`player_level = 135` and `npc_effective_level = 150`.

At effective level 150 and the supplied/default Innie properties, the source
path yields HP 274,599, AC 2,250, toughness 15, attack speed 1 second, attack
radius 1, and template damage 3,549. Unspecified Innie stats, including strength
and MP, are zero, so the NPC strength factor is 1 and raw melee damage remains
3,549 before the target's dodge/block/AC path. The template properties contain
no NPC spells.

Golden tests should include at least levels 135, 149, 150, and 151 because the
enemy-HP formula changes branch at 150. Preserve C# cast-to-`long` truncation.

## Movement and range rules from `NPC.cs`

Port the supplied movement state machine rather than substituting a general
pathfinder:

- entities move only in the four cardinal directions;
- `MoveTowardsBasic` faces the target without moving when Euclidean distance is
  at most 1;
- otherwise `NextStepToBasic` evaluates the four legal candidate tiles using
  Manhattan distance to the target;
- equally useful horizontal/vertical candidates use the source's weighted
  random choice based on squared axis distance;
- when the preferred tile is invalid, the source chooses among alternatives
  using its explicit blocked-path branches and inclusive random rolls;
- a successful NPC move changes position and facing, runs the on-move effects,
  resets the move timer, and partially resets attack readiness while aggroed:
  now plus `min(attack_ms / 2, attack_ms - 500)`, or 50 ms for intervals at or
  below 500 ms. A 1000 ms interval becomes ready after 500 ms, not 1500 ms.
  Fractional milliseconds round up on the integer clock;
- a blocked move changes facing, runs on-move effects, and resets the move timer
  without changing position;
- idle random movement uniformly chooses one of four cardinal offsets, returns
  toward the spawn area when outside it, and still respects `IsValidNPCMove`.

For baseline movement, `is_legal_move` checks only `(1)` map bounds and `(2)`
whether the destination is occupied by the player or another living NPC. Keep
terrain collision behind the same map adapter, with `use_blocked_layer=False`,
so the 5,088-cell JSON layer can be enabled later without changing NPC logic.
When multiple NPC events share a time, stable event order makes the first move
claim its destination and later moves see that cell as occupied.

The server `Direction` enum order is left, right, up, down, while Mystic's
observation encoding is up=0, down=1, left=2, right=3, idle=4. Keep separate
enums/conversion functions and test every mapping; do not cast one directly to
the other.

Aggro and attacks use Euclidean distance. With the confirmed Innie radius 1,
only `(x±1, y)` and `(x, y±1)` are attackable. A diagonal is `sqrt(2)` away and
would become attackable at radius 1.5.

Port `Ranges.Mystic` literally:

```text
E  = 0.5 * level^2 + 8 * level + 27.5
DW = int64(caster.dexterity) * int64(caster.strength)
range = min(maxRadius,
            minRadius + (2.57 / 15)
            * max(0, sqrt(DW) - sqrt(E))^0.75)
```

Keep the spell scripts' outer clamping/subtraction separate from this helper;
Acid Cloud and Tempest Inferno call it with different adjusted maxima.

## Shared player and combat formulas now available

Port these implementations from `Player.cs`, `Entitybase.cs`, and
`BaseSpell.txt` before translating the spell bodies:

- player strength and intelligence factors;
- spell crit chance and magic crit multiplier;
- raw spell damage and spell multiplier;
- dodge, block chance, block amount, AC mitigation, and damage application;
- `ApplyAoECalculation` with its three target-count branches;
- front/side/behind facing bonuses;
- `CanHit`, target validation, and player-versus-monster rules;
- regeneration timing and clamping to max HP/MP.

The documented stats provide reference checks: strength factor about 5.04,
intelligence factor about 9.41, spell crit about 42.4%, magic crit multiplier
about 2.01, dodge about 17.7%, and block chance about 0.7%. Inventory-derived
weapon damage and nonzero `MaxStats.SpellMultiplier` still require explicit
configuration because item records are not supplied.

## Spell translations

Each spell should expose a pure calculation layer and a state-mutation layer.
The pure layer makes formula parity testable without constructing a full Gym
environment.

Use the supplied property records rather than the fallback values embedded in
the scripts. This removes the unsafe 100%-MP defaults from spell 1 and spell 2.

### Spell 1

Translate `Spell1_Arcane Blast_single_target_spell.txt` in its written order:

```text
id=416, name=Arcane Blast, target=all, targetMode=targeted
cooldown=3500 ms, family=arcane-blast, familyCooldown=3500 ms
mpCost=500, manaFactor=0.45, damagePercent=0.75
manaConsumption=0.33, animation=29, sfx=27
```

1. Validate the chosen target with the `BaseSpell.CanHit` rules.
2. Require at least 500 MP and subtract the fixed 500 MP cost before entering
   the spell calculation.
3. Read `manaFactor`, `damagePercent`, and `baseDamage` properties, using the C#
   defaults when a property is absent.
4. Calculate base damage from post-cost current MP, strength factor, and
   intelligence factor, including the October 2024 modifiers.
5. Calculate raw spell damage and the main target's crit roll.
6. Apply the main target's magic mitigation.
7. Consume 33% of the then-current MP exactly where the C# spell body does.
8. When Arcane Bomb is enabled, enumerate splash targets and calculate their
   individual raw damage, crit, mitigation, and AoE scaling.
9. Apply splash damage and then main-target damage in the source order.

Without Arcane Bomb this is a single-target spell. The source defaults to zero
splash radius.

### Spell 2

Translate `Spell2_Acid Cloud_large AoE.txt`:

```text
id=417, name=Acid Cloud, target=all, targetMode=targetedOrSelf
cooldown=5000 ms, mpCost=1000
manaFactor=0.275, damagePercent=0.75, manaConsumption=0.33
initialDamagePercent=0.76, tickPercent=0.04
duration=6000 ms, interval=1000 ms
minRadius=2.5, maxRadius=4.25, level=70
effectId=acid, effectLevel=2, animation=118, tickAnimation=118, sfx=5
```

1. Require at least 1,000 MP and subtract the fixed 1,000 MP cost before entering
   the spell calculation.
2. Calculate Mystic range and clamp it between `minRadius` and `maxRadius`.
3. Select all hittable monsters in the target-centered radius.
4. Calculate raw damage from post-cost current MP, then crit, AoE scaling, and
   distance falloff in the same
   order as C#.
5. Apply initial damage and target mitigation.
6. Consume 33% of the then-current MP.
7. Schedule the per-target tick effect using absolute millisecond deadlines.
8. Apply the optional Sunburnt multiplier only when that status is enabled.

For the documented level-135 player (`Dexterity=194`, `Strength=100`), the exact
Mystic calculation produces raw radius about 7.2611, which the spell clamps to
4.25. Acid Cloud ticks every five environment steps for six seconds.

### Spell 3

Translate `Spell3_Tempest Inferno.txt` exactly:

```text
id=418, name=Tempest Inferno, target=all, targetMode=targetedOrSelf
cooldown=1750 ms, mpCost=750
castLevel=134, level=125
manaFactor=0.45, damagePercent=0.775, manaConsumption=0.20
animation=2, sfx=119
```

1. Fall back to the caster when the target is null, as the C# does.
2. Require at least 750 MP and subtract the fixed 750 MP cost before entering
   the spell calculation.
3. Calculate radius from the Mystic range function with defaults 0.5–1.5.
4. Calculate the caster-to-target distance factor.
5. Build full-radius and partial-radius target sets, excluding magic-immune
   monsters.
6. Calculate raw damage from post-cost current MP, then AoE scaling,
   partial-target weights, and per-target allocation with C# truncation at the
   same points.
7. Apply magic mitigation and damage to both target sets.
8. Consume the configured 20% of current MP only when at least one target is
   hit, matching the source location of the consumption code. The script's 15%
   fallback applies only when `manaConsumption` is absent; it is present here.
9. Apply the optional Tempest Meteor stun only when that trinket modifier is
   enabled; baseline simulation keeps it disabled.

The script reads `level` before `castLevel`, so this property record uses level
125 for `Ranges.Mystic`. With the documented player stats, the resulting spell
radius is exactly the 1.5 maximum.

### Shared cast and targeting rules

Port the relevant portion of `BaseSpell.txt`:

- map and range validation;
- MP/HP fixed-cost validation;
- target-mode resolution;
- immunity checks;
- `CanHit` behavior for player-versus-monster combat.

The current discrete actions do not carry a target ID or cursor coordinate.
For the first simulator, select the nearest hittable living monster, breaking
ties by entity ID. Arcane Blast is `targeted` and fails with `no_target` when no
monster is available. Acid Cloud and Tempest Inferno are `targetedOrSelf`; when
no valid monster exists, follow the source target-mode rule and center the spell
on the caster. This nearest-target rule is confirmed for the current simulator
scope. Range and `CanHit` validation still run after selection.

The spell records in `TimeFrame_Other_Details.txt` define both a fixed `mpCost`
and a percentage `manaConsumption`. The supplied `BaseSpell.CanCast` fragment
checks that the caster can afford `mpCost`; its local subtraction lines are
commented because the fixed charge occurs in the surrounding cast pipeline.
For the simulator, a successful cast must execute this resource order:

1. validate target, cooldown, and `CurrentMP >= mpCost`;
2. subtract the fixed cost: 500 for Arcane Blast, 1,000 for Acid Cloud, or 750
   for Tempest Inferno;
3. run the spell body, so every formula reading `caster.CurrentMP` sees the
   post-fixed-cost value;
4. at the spell body's consumption point, subtract 33%, 33%, or 20% of the
   then-current MP using C# `Math.Round`, clamping at zero.

For starting MP `M`, Arcane Blast and Acid Cloud therefore use `M - mpCost` in
their damage formula and percentage base. Tempest Inferno also uses `M - 750`
for damage, but its 20% charge occurs only inside the source branch that found
at least one target. The fixed 750 MP has already been spent by that point.
Tests must cover this no-hit distinction explicitly.

Using the documented starting MP of 15,108 gives concrete resource-order test
cases:

| Spell | After fixed cost | Rounded percentage charge | MP after successful hit |
|---|---:|---:|---:|
| Arcane Blast | 14,608 | `Round(14,608 * 0.33) = 4,821` | 9,787 |
| Acid Cloud | 14,108 | `Round(14,108 * 0.33) = 4,656` | 9,452 |
| Tempest Inferno | 14,358 | `Round(14,358 * 0.20) = 2,872` | 11,486 |

For a successful Tempest cast that finds no hittable entities, the expected MP
is 14,358 because the fixed cost was charged but the conditional 20% charge was
not reached.

Store both slot and family cooldowns as absolute millisecond deadlines. At 200
ms decision boundaries, the first possible recasts are:


| Spell           | Cooldown | First eligible boundary | Decision intervals |
| --------------- | -------: | ----------------------: | -----------------: |
| Arcane Blast    |  3500 ms |                 3600 ms |                 18 |
| Acid Cloud      |  5000 ms |                 5000 ms |                 25 |
| Tempest Inferno |  1750 ms |                 1800 ms |                  9 |

Cooldown eligibility should use the source `Spell.CanCast` behavior when that
class is supplied; until then, use `now_ms >= ready_at_ms` and test boundary
times explicitly.

## Basic attack

`attack` remains in the action contract, but gear is disabled in the baseline
fidelity profile because weapon damage and attack speed were intentionally left
out by the developer. Implement it behind a configuration toggle:

- follow the standard `Player.MeleeAttack` targeting rule: inspect only the
  cardinal tile directly in front of the player;
- with `gear_enabled=False`, return a no-op event with reason `gear_disabled`;
- with `gear_enabled=True`, if the facing tile contains a living monster, apply
  configured integer weapon damage through `HandleMelee`, including dodge and
  target mitigation;
- preserve the standard single-target `3 / (numTargets + 2)` factor, which is
  1 when `numTargets == 1`;
- use the player attack-speed gate once the relevant weapon speed is supplied;
- do not add alternate weapon patterns, splash, or a resource cost in the first
  version;
- return a no-op with `no_target` when the facing tile is empty and
  `cooldown` when the attack-speed gate rejects the action.

The optional gear profile must supply weapon damage, attack interval, and spell
multiplier together. This keeps approximate gear values out of the source-exact
baseline.

## Package design

```text
Custom_enviornments/
  Mystic_Sim/
    __init__.py
    env.py                 # thin Gymnasium adapter
    actions.py             # shared eight-action definition
    config.py              # immutable player, NPC, spell and scenario values
    state.py               # player, monster, effect, map and event dataclasses
    engine.py              # reset and deterministic 200 ms state transition
    observation.py         # pure state -> Mystic-compatible float32 vector
    rewards.py             # reward components and episode rules
    movement.py            # collision, distance, aggro and pursuit
    scaled_calcs.py        # literal ScaledCalcs/NPCRecursiveDmg translation
    combat.py              # rolls, mitigation, damage, death and regeneration
    spells.py              # spell 1, 2 and 3 translations
    map_loader.py          # validated Tiled JSON -> map and spawn definitions
    scheduler.py           # absolute-time heap and deterministic tie-breaks
    scenarios.py           # seeded map53 fidelity profiles
Tests/
  fixtures/simulation/
  test_mystic_sim_state.py
  test_mystic_sim_timing.py
  test_mystic_sim_scaled_calcs.py
  test_mystic_sim_combat.py
  test_mystic_sim_spells.py
  test_mystic_sim_observation.py
  test_mystic_sim_environment.py
```

Rules belong in pure functions or the engine rather than the Gym wrapper. This
keeps formula tests small and leaves a clean path to batching or optimizing only
the measured bottlenecks later.

## Observation contract

Continue producing Mystic's 26-element `numpy.float32` vector:

```text
[player_x, player_y, direction, hp_pct, mp_pct, map_id,
 enemy_0_distance, enemy_0_direction, enemy_0_hp_pct, enemy_0_mp_pct,
 ... five nearest monsters total]
```

Extract a pure encoder from `Env_conditions.parse_observation`. Its current debug
file writes cannot run inside a high-throughput simulation step. Test the pure
world-state encoder with `Example_Full_State.txt`, including exact shape, dtype,
values, nearest IDs, direction values, normalization, and map-ID parsing.

Cooldowns, status effects, target ID, and terrain remain hidden from this
observation. Keep the parity observation first, and use a recurrent policy if
hidden timers cause problems. Any expanded observation must receive a new
version rather than changing trained-model inputs silently.

## RNG parity from `GameServer.cs`

The server uses the process-wide, thread-safe `Random.Shared`. Its helper
semantics are now known:

```text
Roll(min, max)       -> Random.Next(min, max + 1)       # both endpoints included
Roll(options)        -> Random.Next(0, options.Length)  # upper endpoint excluded
RollChance(chance)   -> Random.Next(0, 1_000_000_001)
                        <= chance * 1_000_000_000
```

Implement these as compatibility helpers backed by each environment's seeded
NumPy generator. `roll_chance(0)` retains the source's extremely small success
possibility because integer zero satisfies the `<=` comparison; do not silently
replace it with `rng.random() < chance` in the exact profile. Record RNG draws
in trace mode so formula comparisons can verify when each roll is consumed.

Preserve source call order as well as distributions. Examples now visible in
the supplied files include:

- each Innie samples its movement interval once when that life begins. Store the
  sampled 0.9–1.1-second value on the NPC and reuse it for every move until
  death; a respawn begins a new life and consumes a new movement-speed draw;
- `NextStepToBasic` eagerly rolls its first coin flip after evaluating candidate
  tiles and then rolls the weighted axis choice unless an adjacent early return
  was taken;
- random movement/facing uses one inclusive roll from 0 through 3;
- melee rolls dodge first, followed by attacker crit when applicable, followed
  by target block;
- each raw player spell-damage calculation rolls crit, followed by the target's
  mitigation/block roll.

Tests should assert RNG draw logs for representative movement, melee, and spell
branches so a refactor cannot change later outcomes by skipping an apparently
unused draw.

The live server consumes one shared RNG across concurrent work, so reproducing a
live global random sequence is not practical. Simulator determinism means the
same environment seed, state, and action sequence consume the same local draws.

## Reset and step semantics

`reset(seed=..., options=...)` should call `super().reset(seed=seed)`, construct
the selected scenario with `self.np_random`, reset every timer/effect, and return
the initial observation and diagnostic info.

One `step(action)` covers exactly 200 simulated milliseconds. `EventHandler.cs`
establishes the scheduling model: events carry absolute times, map events enter
a timestamp-priority queue, the single map worker checks the head against the
current clock, and due events execute serially through `CheckAndHandle`.

Implement a heap ordered by `(due_ms, enqueue_sequence)`. The monotonically
increasing sequence supplies deterministic equal-time ordering because the
third-party C# priority queue's equal-priority behavior is not included. An
event scheduled while handling another event receives a later sequence, even if
it has the same due time. Cancelled events remain harmless heap entries and are
ignored using a generation/token check.

The step transaction is:

1. At `t`, validate the discrete action, resolve the nearest target where
   required, and apply or reject the player action.
2. Set `step_end = t + 200`.
3. Pop every event with `due_ms <= step_end` in heap order. Set the engine clock
   to each event's exact due time before executing it. Events may enqueue new
   events that are also due before `step_end`.
4. Advance the clock to `step_end` after the queue is drained.
5. Encode observation, calculate rewards from the step's event ledger, update
   termination/truncation, and return the Gymnasium tuple.

Use explicit event kinds for NPC update, regeneration, effect tick/expiry, and
respawn. Damage and death caused inside one event are applied synchronously so
later equal-time events observe the updated HP/alive state. Inside an NPC update,
preserve the source order: validate/drop aggro, acquire aggro when due, move or
face, then attack when due. This is the baseline simulator contract; a later
live trace can replace only the tie-break policy if necessary.

All randomness uses the environment's seeded generator, including spawn cells,
move-speed jitter, dodge, block, crit, path choices, and respawn placement.

## Rewards and episode boundaries

Preserve Mystic's component names for dashboard compatibility:

```text
health_state, positioning, damage_taken, damage_dealt, terminal, killed
```

Use explicit simulator events for damage and death rather than inferring kills
from disappeared IDs. Start with player death as loss, five Innie kills as win,
and 256 steps as truncation.

The scenario map ID is 53. `Example_Full_State.txt` is on map3471 and is used
only to lock parser compatibility; it is not a simulator reset state. Keep map
ID, Y-position penalties, and Y-based truncation as scenario configuration so
reward experiments do not alter mechanics.

Review the existing `player_hp_pct != 0.5` damage condition separately; it can
reward healing or apply a zero-value component in surprising cases. Corrected
reward behavior should use a named version instead of silently changing old
experiment semantics.

`info` should include:

```text
current_step, simulation_time_ms, reward_components, episode_outcome, is_win,
action_applied, action_failure_reason, selected_target_id, damage_events,
death_events, cooldowns_remaining_ms
```

## Baseline fidelity contract

The first runnable environment can now be implemented without inventing missing
map, target-selection, collision, or event-queue behavior. Name this profile
`map53` and freeze these settings in one immutable scenario
configuration:

| Setting | Baseline value |
|---|---|
| Map | ID 53, 100 by 100, parsed from `map53.json` |
| Terrain collision | Enabled; enforce the parsed blocked cells |
| Occupancy collision | Enabled for the player and living NPCs |
| Player spawn | Uniform legal cell in X 45..55 and Y 40..50, inclusive |
| NPC spawns | All 40 non-fixed template-5300 boxes, quantity two each |
| Other NPCs | Fixed template 5399 excluded |
| Innie movement interval | One inclusive integer sample from 900 through 1,100 ms per life |
| Innie respawn | 50,000 ms after death, in its original spawn box |
| Objective | Kill five Innies |
| Step duration | 200 ms |
| Actions | Four cardinal moves, attack, spells 1 through 3 |
| Target selection | Nearest living candidate, then lowest entity ID |
| Basic attack/gear | Disabled by default; toggleable configuration |
| Cooldowns | Absolute ready times using the supplied spell/family durations |
| Event tie-break | Absolute due time, then enqueue sequence |
| Episode end | Player death or five kills; truncate at 256 steps |

Treat all interval endpoints as inclusive. If a spawn rectangle is full, fail
reset with a diagnostic rather than loop indefinitely. A respawn with no free
cell stays pending and retries at the next 200 ms boundary; record the reason in
the event trace.

### Known discrepancies and deferred fidelity

These items do not prevent implementation, but must remain visible:

1. `map53.json` stores `balanceCap=68`; the developer confirmed 150 and the
   captured 274,599 Innie HP validates the level-150/bulk-0.5 calculation. The
   baseline uses an explicit 150 override and preserves 68 as parsed metadata.
2. `EventHandler.cs` defines priority-by-time and serial handling, but the
   equal-priority behavior of `ConcurrentPriorityQueue` is unavailable. The
   simulator's enqueue-sequence tie-break is therefore a documented deterministic
   rule rather than a verified server detail.
3. “50 +/- 5, 45 +/- 5” defines the player spawn rectangle but not a probability
   distribution. The baseline uses a uniform integer-cell distribution.
4. Gear is intentionally absent. The attack action returns `gear_disabled`, and
   gear-derived damage, speed, and spell multiplier remain zero until one
   complete gear profile is supplied.
5. The blocked layer is parsed, validated, and enforced in map53 for all entities.

Every configuration value should carry provenance (`source`, `developer`,
`fixture`, `derived`, or `simulator_rule`). Approximate mechanics belong in a
separately named profile and must never silently enter the baseline.

## Implementation phases

### Phase 0 - Freeze contracts and fixtures

Implemented: canonical actions and schema metadata, versioned checkpoint helpers,
strict legacy-BC JSON remapping, separate simulator registration scaffold,
pure observation encoding, tracked/hash-identified fixtures, mechanics manifest,
and map53 schema validation. See
`Custom_enviornments/Mystic_Sim/README.md` for usage and compatibility boundaries.
The live environments retain their legacy 11-action defaults until migration;
the registered simulator now supports Phase 1 reset and Phase 2 movement steps.
Damage, spells, and combat reward are now implemented through Phases 3–5 below.

Deliverables:

- Add `Mystic_Sim/actions.py` with one `IntEnum` and the canonical eight labels.
- Make live `Mystic`, `MysticBC`, the simulator, trainers, and inference import
  that definition when they migrate to the eight-action version.
- Give the simulator its own registration, `YugenSaga/MysticSim-v0`, so live
  socket behavior cannot be selected accidentally during simulation training.
- Version action metadata in checkpoints and datasets. Existing 11-output
  checkpoints cannot load into an eight-output policy head without migration.
- Note the current movement-order difference: live Mystic uses up/down/left/
  right, while MysticBC uses up/left/right/down. Define the canonical order as
  up/down/left/right and provide an explicit legacy-BC remapping tool.
- Store new demonstrations under a new dataset version rather than relabeling an
  existing 11-action dataset in place.
- Lock `Example_Full_State.txt` as a golden fixture for its exact observation,
  selected entity IDs, dtype, and shape.
- Add `mechanics_manifest.yaml` with every baseline value and its provenance.
- Add schema assertions for `map53.json`: dimensions, tile scale, required
  layers, map properties, 40 template-5300 boxes, and their quantities.

Exit gate:

- action mappings round-trip by label and integer;
- legacy BC actions remap correctly in a fixture;
- the full-state parser test passes without writing debug files;
- malformed or unexpected map JSON fails with a precise validation message.

### Phase 1 - Build map, state, reset, and observation

Implemented: frozen validated configuration; bundled map53 loading and immutable
tile/spawn definitions; slotted runtime state; player-first seeded placement of
80 non-overlapping Innies; one movement-interval draw per life; initial event
records/deadlines; pure nearest-five observation encoding; and reset diagnostics.
Tests cover seeds, state isolation, full-box rejection, RNG order/endpoints,
map normalization, fixture parity, direction/distance boundaries, and no reset I/O.
Initial player/NPC facing is up, a documented reset convention. Runtime event
execution is implemented in Phase 2 below.

Implement data before behavior:

- `config.py`: frozen player, Innie, spell, timing, reward, and fidelity-profile
  dataclasses. Validate positive maxima, durations, map bounds, and unique spell
  IDs at construction.
- `map_loader.py`: read Tiled JSON once, normalize object pixels to tile
  rectangles, retain raw map properties, and create immutable spawn definitions.
- `state.py`: slotted `PlayerState`, `MonsterState`, `SpawnBox`, `TimedEffect`,
  `CooldownState`, and `WorldState`. Use integer HP/MP, integer tile coordinates,
  integer milliseconds, and stable integer entity IDs.
- `scenarios.py`: construct `map53`, apply the documented
  level-150 override, omit template 5399, and seed all placement from the
  environment generator.
- Spawn player first, then spawn boxes by Tiled object ID and members by local
  index. Reject occupied cells without consuming randomness outside the defined
  retry loop. Keep entity IDs stable across death and respawn.
- `observation.py`: pure state-to-vector encoding with the exact 26-element
  `float32` contract. Rank living monsters by Manhattan observation distance and
  entity ID, encode the nearest five, and zero-pad missing slots.
- Keep Euclidean mechanics distance, Manhattan observation distance, and
  direction encoding as separate tested functions.

Reset must return a fully valid state: 80 living Innies on unique cells, one
legal player cell, no cooldowns/effects, zero kills, time zero, and all first
event deadlines scheduled. Expose reset diagnostics such as seed, profile,
spawn coordinates, parsed balance cap, and effective level override in `info`.

Exit gate:

- the map loader derives all 40 boxes and 80 Innies from JSON;
- every reset entity is in bounds, inside its spawn rule, and non-overlapping;
- the same seed produces identical state and observation;
- a sample of different seeds changes legal spawn cells;
- observation parity and direction/distance boundary tests pass;
- reset performs no socket, sleep, render, or debug-file I/O.

### Phase 2 - Add the clock, scheduler, movement, and aggro

Implemented: absolute-time heap dispatch, stable ties, cancellation/generation
checks, loop guard, 200 ms movement steps, source-ordered pursuit RNG, occupancy,
facing and timer behavior, idle/spawn return, radius and damage aggro hooks,
player-move acquisition, and optional per-step in-memory traces. Phase 3 now adds
attack, regeneration, death, and respawn mutations; Phase 4 adds Acid effect
damage. Phase 5 supplies versioned step rewards.
The README records confirmed partial attack resets after movement, 200 ms expired-timer
polling, 1500 ms aggro rescheduling, and tick-before-expiry policy. These are
explicit simulator conventions where complete server scheduling is unavailable.

- Implement `scheduler.py` with heap keys `(due_ms, enqueue_sequence)`, event
  tokens for cancellation, trace records, and a guard against infinite
  same-timestamp rescheduling.
- Schedule player regeneration every 2,000 ms, NPC updates from each NPC's stored
  per-life movement interval and attack deadlines, effect ticks at their exact
  intervals, and respawns 50,000 ms after death.
- Advance exactly 200 ms per Gym step while executing every due event at its own
  timestamp. Test events both on and between decision boundaries.
- Port the `NPC.cs` cardinal candidate generation and weighted pursuit branches
  in source order. Do not substitute A* or another pathfinder.
- Implement baseline legality: in bounds and destination unoccupied. A blocked
  attempt changes facing and applies the source timer behavior without moving.
- Port spawn-area return and idle random movement.
- Port aggro radius 4, 1,500 ms acquisition checks, distance-18 drop, damage
  aggro, invalid/dead target removal, and Euclidean comparisons.
- Keep server-facing direction values separate from Mystic observation values.
- Log event time, event kind, entity ID, RNG draws, before/after position, and
  action result when trace mode is enabled.

Exit gate:

- equal seed plus equal actions gives an identical event trace;
- occupancy conflicts resolve by stable event order with no overlapping state;
- cardinal pursuit, diagonal distance, blocked facing, random movement, aggro
  acquire/drop, spawn return, and timer-reset branches match focused fixtures;
- movement intervals use one inclusive integer draw from 900 through 1,100 ms,
  remain unchanged during a life, and are resampled after death; 1,500 ms aggro
  boundaries are exact;
- no event scheduled after a step boundary executes early.

### Phase 3 - Port scaling, combat, death, and respawn

Implemented in `scaled_calcs.py`, `combat.py`, and the event engine. Structured
damage/death/respawn records are exposed in step info. Python unit tests in
`Tests/test_mystic_sim_combat.py` validate numerical expectations, controlled RNG
branches, event cancellation, death cleanup, respawn placement and retries,
optional gear, and regeneration. Validation does not require executing the
partial C# files or building a standalone C# project.

Regeneration is toggleable with `TimingConfig.regen_enabled`. The corrected
default restores 226 HP and 1664 MP every 2000 ms, capped at maxima. Internally
these are per-second rates of 113 HP and 832 MP. Both rates
and the interval are configurable. Set an override to `None` and supply the
corresponding base stat to use the formula-derived per-second rate instead.

- Translate `ScaledCalcs` and `NPCRecursiveDmg` literally. Preserve lookup
  growth, recurrence order, floating-point operations, and C# casts to `long`.
- Verify levels 135, 149, 150, and 151, including HP 274,599 for effective level
  150 and bulk factor 0.5.
- Construct baseline Innies with HP 274,599, AC 2,250, toughness 15, raw damage
  3,549, attack interval 1,000 ms, Euclidean radius 1, MP zero, and no spells.
- Port player strength/intelligence factors, crit, dodge, block, facing bonuses,
  AC reduction, magic/melee mitigation, `CanHit`, AoE scaling, HP mutation, and
  regeneration with C# rounding at the same lines as the source.
- Add the exact RNG helpers and preserve branch draw order. Trace every draw by
  purpose so a test can detect an accidentally skipped roll.
- Implement the NPC attack pipeline, including radius-1 cardinal reach and
  diagonal rejection, attack deadline, player damage, and death.
- Emit structured `DamageEvent`, `DeathEvent`, and `RespawnEvent` records. Death
  occurs once, removes occupancy and aggro immediately, cancels invalid effects,
  and schedules the original entity into its own spawn box after 50 seconds.
- If its box has no free cell at the deadline, leave the entity dead and retry
  placement at the next decision boundary without changing its ID.
- Keep player gear disabled. Implement the facing-tile attack path behind
  `gear_enabled`; baseline attempts return `gear_disabled` without RNG draws.

Exit gate:

- Python unit tests match saved numeric expectations at the specified levels
  and formula boundaries, without compiling or executing C#;
- normal, dodge, block, crit, mitigation, lethal, and already-dead cases pass;
- NPCs cannot attack diagonally at radius 1;
- regeneration occurs at exactly 2,000 ms by default, scales per-second rates
  with the configured interval, clamps to maxima, and respects its enable toggle;
- death frees occupancy and respawn restores a legal, fully initialized entity;
- RNG trace tests lock draw endpoints and draw order.

### Phase 4 - Implement Mystic spells 1 through 3

Implemented: pure spell calculations, nearest Euclidean targeting, fixed-cost
eligibility, slot/family cooldowns, cast transactions, all three damage pipelines,
Acid tick snapshots, structured cast records, and Python numeric/state-transition
tests in `Tests/test_mystic_sim_spells.py`.

**Confirmed casting and refresh rules:** eligible targets must satisfy both
`abs(dx) <= 16` and `abs(dy) <= 10`. Rank eligible monsters by Euclidean distance
and entity ID; retain Euclidean AoE radii. Acid refreshes rather than stacks for
the same caster/target pair, replacing the damage/crit snapshot and restarting
tick and expiry deadlines. Different casters' effects coexist independently.
Stable Acid IDs use effect name plus caster ID per target, and each application
receives a new generation; refresh cancels the previous queued callbacks.
Existing scheduler behavior is
unchanged: ticks at +1000..+5000 ms, expiry at +6000 ms, no expiry-time tick. The
missing TickEffect implementation prevents claiming that endpoint is confirmed.
AoE enumeration uses stable entity-ID order because the map enumerator is absent.

Build casting in layers:

1. `targeting.py` filters living monsters by the inclusive 16-X/10-Y casting
   rectangle, then ranks by Euclidean distance and entity ID. Targeted-or-self spells fall
   back to the caster only when no valid monster exists.
2. `cooldowns.py` checks fixed MP/HP affordability plus slot/family absolute
   ready times. Rejected casts consume no resources, cooldown, or combat RNG.
3. Pure calculation functions return target allocations, raw damage, crit flags,
   mitigation inputs, and resource costs without mutating world state.
4. The cast transaction subtracts fixed `mpCost`, passes the post-cost MP into
   the spell calculation, applies results in source order, performs the later
   percentage consumption, emits events, schedules effects, and sets cooldowns.

Implement Arcane Blast (416) first: nearest single target, 500 fixed MP cost,
3,500 ms slot/family cooldown, then 33% remaining-MP consumption, and
baseline-disabled Arcane Bomb splash. Then implement Acid Cloud (417): 1,000
fixed MP cost,
5,000 ms cooldown, radius capped at 4.25, initial damage, 1,000 ms ticks for
6,000 ms, then 33% remaining-MP consumption. Finally implement Tempest Inferno
(418): 750 fixed MP cost, 1,750 ms cooldown, radius capped at 1.5, full/partial
target allocation, 20% remaining-MP consumption from the supplied property
record when at least one target is hit, and baseline-disabled trinket stun.

Resolve an existing plan inconsistency in favor of the supplied property record:
Tempest Inferno uses `manaConsumption=0.20`; do not retain the spell script's
15% fallback when the property is present.

Every successful spell starts a shared 300 ms global cooldown, in addition to
its slot/family cooldowns. No spell can cast before that absolute deadline;
movement remains available. At 200 ms decisions, another ready spell can first
cast at +400 ms. Rejected casts neither start nor extend this cooldown and consume
no resources or combat RNG. Reset/death clears the shared deadline.

Cooldowns begin only after a successful cast. At 200 ms decisions, first recast
boundaries are 3,600 ms, 5,000 ms, and 1,800 ms. Timed effects use absolute
deadlines and stable effect IDs. Refresh is per caster/target; different casters
stack independently. Baseline trinket/status modifiers remain disabled but represented in
configuration.

Exit gate:

- each spell has pure numeric goldens and full state-transition tests;
- tests cover no target, self fallback, exact AoE radius edge, insufficient MP,
  cooldown edge, normal/critical hit, immunity, zero/one/many AoE targets,
  partial-radius allocation, DoT tick/expiry, fixed-before-percentage resource
  order, post-cost damage input, resource rounding, no-hit Tempest cost, and kills;
- a failed cast produces a reason and no hidden mutation or RNG draw;
- cooldown and effect traces are deterministic under replay.

### Phase 5 - Wrap the engine in Gymnasium and define rewards

Implemented in `env.py`, `rewards.py`, and `diagnostics.py`, with tests in
`Tests/test_mystic_sim_rewards.py`. Both reward profiles pass Gymnasium's checker.
The default training weights are enemy damage +1 normalized by enemy max HP,
player damage -1 normalized by player max HP, kill +1, death -5, and time -0.001
per decision under `health_state`. All six dashboard component names are retained.
Enemy damage and kill rewards come from explicit engine events, never lost IDs.
Legacy preserves live coefficients/shaping and its signed player HP-delta quirk
for comparison; training uses damage events even if regeneration heals that step.

`RewardConfig` makes weights, kill goal, step limit, optional inclusive `y_bounds`,
and `legacy_y_penalty_below` configurable. The Y penalty defaults to row 31 for
both reward profiles, with the matching upper penalty above 85. Simulator
Y-boundary truncation defaults to Y <= 29 or Y >= 87.
Death overrides a simultaneous kill goal, and termination overrides truncation.
Reset accepts the existing fidelity profile and an episode-local `reward_profile`;
it validates before mutating state. Fixed fixture reset is not exposed. Diagnostics
are detached snapshots, with the selected target captured at action time.

- Keep `env.py` thin: validate the action, call `engine.advance(action, 200)`,
  encode state, calculate reward, and return the Gymnasium five-tuple.
- Declare `Discrete(8)` and a 26-value `Box` with bounds that reflect coordinates,
  direction codes, percentages, distance, and map ID.
- Implement `reset(seed, options)` through `super().reset(seed=seed)`. Options may
  select a fidelity profile or fixed fixture, but cannot mutate global state.
- Use explicit engine events for damage and kills rather than inferring them from
  disappeared entity IDs.
- Preserve dashboard component names: `health_state`, `positioning`,
  `damage_taken`, `damage_dealt`, `terminal`, and `killed`.
- Add a `legacy_reward_v0` profile that snapshots current Mystic calculations and
  a `combat_reward_v1` profile for training. The latter should reward normalized
  enemy damage and kills, penalize normalized player damage and death, use a
  small time cost, and avoid hard-coding the current `player_hp_pct != 0.5` bug.
- Terminate on player death or the fifth Innie kill. Truncate at 256 steps. Keep
  Y-boundary rules configurable; the simulator task truncates at Y <= 29 or
  Y >= 87 and applies penalties below 31 and above 85.
- Return diagnostics including profile, seed, simulation time, step, selected
  target, action result/reason, reward components, kills, cooldowns, active
  effects, and the step's damage/death/respawn events.

Exit gate:

- `gymnasium.utils.env_checker.check_env` passes;
- reset/step observations always match declared shape, dtype, and bounds;
- termination and truncation are never conflated;
- seeded episode replay produces byte-equal observations, rewards, and traces;
- all existing repository tests remain green.

### Phase 6 - Train PPO on the current Mystic Sim environment

Primary deliverable: repeatable single-environment PPO training, checkpointing,
and evaluation on the current Mystic simulator. This phase is simulation-only:
no live adapter implementation, ZMQ contract tests, payload comparisons, or game
sessions. Training must work without a running game, ZMQ connection, or Pygame window.

1. Add an explicit simulation training path and environment factory to
   `Training/PPO_server.py` first, then `Training/PPO_lstm_server.py`.
   Preserve existing live entry points without extending their contract; use the
   actual `MysticSimEnv` with training episode rules, not the viewer's free-play
   configuration. Initially use one environment per run.
2. Consume Mystic Sim's current `Discrete(8)` and float32 26-value observation
   directly. Verify that policy input/output dimensions, action labels, and
   preprocessing match this simulator. Reject incompatible legacy checkpoints
   when loading into simulation. Live action mapping and payload contract tests
   belong exclusively to Phase 8.
3. Exercise the complete PPO rollout/update loop: observation batching, action
   selection, log probabilities, values, advantages, optimizer updates, and
   TensorBoard metrics. Preserve the six reward component names. Test final
   observations, episode resets, termination masks, and time-limit bootstrapping
   under the chosen Gym vector/autoreset mode. For recurrent PPO, also test
   hidden-state reset and recurrent minibatch boundaries.
4. Use `combat_reward_v1` by default. Record the simulator reward, task, and
   fidelity configuration with each run. Validate episode statistics and the six
   reward components against simulator events. Live reward differences are a
   Phase 8 comparison task, not a requirement for this training implementation.
5. Run a short seeded training smoke test, then a bounded multi-seed learning
   run. Verify finite losses/rewards and actual parameter updates. Report return,
   kills, survival, damage dealt/taken, invalid-cast rate, and throughput against
   a random-action baseline. A successful optimizer run proves trainability;
   it does not by itself prove learning quality or live-game fidelity.
6. Save and reload PPO models for simulator evaluation. Store environment/backend, action labels and
   schema, observation schema/preprocessing, fidelity and reward profiles, map
   and mechanics-manifest hashes, seed, and PPO hyperparameters in run/checkpoint
   metadata. Check compatibility before simulator evaluation or training resume.
7. Verify training resume restores optimizer, counters, schedules, and RNG state.
   Exact mid-episode simulator resume additionally requires world, scheduler,
   pending events/effects, and environment RNG snapshots. Otherwise explicitly
   resume at a fresh episode; do not claim identical next transitions.
8. Keep the simulator factory and rollout code ready for Phase 7 environment-count
   settings, but finish single-environment training/checkpoint tests here. Do not
   make Phase 6 completion depend on vector throughput or live validation.

Exit gate:

- documented commands train and evaluate PPO on Mystic Sim without the live
  bridge, with finite losses and confirmed parameter updates;
- save/load and the declared resume behavior pass regression tests;
- feed-forward and recurrent PPO paths respect action/observation contracts and
  episode boundaries; recurrent state cannot leak across episodes;
- a simulator-trained eight-action checkpoint reloads for simulator evaluation;
- run metadata records the simulator reward and fidelity settings; the trainer has
  an environment-factory boundary ready for Phase 7. No Minari or BC work is required.

### Phase 7 - Vectorize Mystic Sim and increase PPO steps per second

Primary deliverable: PPO collects batches from many independent Mystic worlds,
with measured throughput improvements and correct episode handling. This is a
required phase after single-environment PPO, not an optional optimization after
live validation. It does not require Minari, a game connection, or Pygame.

#### What vectorization means for this custom environment

Keep `MysticSimEnv` as a normal single-world Gym environment. A vector wrapper
creates N separate instances and presents their observations as one batch to
one PPO model. Each world contains its own player, 80 Innies, scheduler, RNG,
cooldowns, and effects. Do not put N players into one map, share one mutable
environment across workers, or train N separate PPO models.

| Value | One environment | N environments |
|---|---|---|
| Observation | `(26,)` float32 | `(N, 26)` float32 |
| Action | One integer 0..7 | `(N,)` integer array |
| Reward | Scalar | `(N,)` array |
| Terminated / truncated | Two booleans | Two `(N,)` boolean arrays |
| One PPO rollout with T decisions | T transitions | T * N transitions |

`SyncVectorEnv` steps the worlds sequentially in one process. Use it first to
debug batching; it can improve policy-inference efficiency but does not parallelize
the Python physics. `AsyncVectorEnv` uses subprocesses so world steps can run on
multiple CPU cores. Both APIs wait for the batch before returning; async here
does not mean different workers train on independently updated policies.
See the official [vector API](https://gymnasium.farama.org/api/vector/) and
[AsyncVectorEnv API](https://gymnasium.farama.org/api/vector/async_vector_env/).

#### 7.1 Make the info payload safe to batch

The current environment already has the correct spaces, but wrapping it directly
is not sufficient. A local four-world smoke probe against Gymnasium 1.2.1 fails
when an integer `selected_target` from one worker is combined with `None` from
another: Gymnasium attempts to store None in an integer array. Treat this as the
first implementation task, not a trainer error.

- Add `TrainingInfoWrapper` for training only. Return a fixed, consistently typed
  info schema: scalar kills/time/step, boolean action success, reward components,
  string outcome/reason (empty string when absent), and target ID with `-1` for
  no target. Use the same types on reset and step.
- Keep rich cast/damage/death/effect lists in the normal diagnostic path. The
  reward is already calculated inside the environment, so PPO does not need to
  transport all event records between processes on every step. Do not remove
  events before reward calculation or weaken the normal debugging contract.
- Test batches containing different outcomes: a cast target, no target, failed
  cast, death, truncation, and continuing play. Inspect Gymnasium's per-key
  presence masks rather than treating vector info as a list of dictionaries.
- Start with a filtering wrapper for correctness. Then add an explicit lightweight
  diagnostic mode to avoid constructing discarded event dictionaries at all if
  profiling shows material overhead. `trace=False` alone currently does not
  suppress all diagnostic serialization.

#### 7.2 Build a small standalone vector smoke program

Create an importable module such as `Training/mystic_vector_smoke.py` using this
starter. This is proposed implementation code, not an existing CLI command.
It uses the installed Gymnasium 1.2.1 API; pin/test that API when upgrading.
The example below was exercised from a temporary script with four workers in
both sync and Windows-spawn async modes: each completed 1,200 transitions and
16 episodes. This validates the starter, not PPO integration or a speedup claim.

```python
import argparse
from functools import partial
import multiprocessing as mp

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode

from Custom_enviornments.Mystic_Sim.env import MysticSimEnv


class TrainingInfoWrapper(gym.Wrapper):
    @staticmethod
    def compact(info):
        target = info.get("selected_target")
        return {
            "kills": int(info["kills"]),
            "current_step": int(info["current_step"]),
            "simulation_time_ms": int(info["simulation_time_ms"]),
            "selected_target": -1 if target is None else int(target),
            "action_applied": bool(info.get("action_applied")),
            "action_failure_reason": info.get("action_failure_reason") or "",
            "episode_outcome": info.get("episode_outcome") or "",
            "reward_components": dict(info["reward_components"]),
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self.compact(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self.compact(info)


def make_worker():
    # Construct inside the worker. Never capture an existing env or open socket.
    # Direct construction preserves the simulator's own episode limits without
    # adding a second registration-level TimeLimit wrapper.
    return TrainingInfoWrapper(MysticSimEnv(trace=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--mode", choices=("sync", "async"), default="sync")
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    factories = [partial(make_worker) for _ in range(args.num_envs)]
    kwargs = {"autoreset_mode": AutoresetMode.SAME_STEP}
    if args.mode == "async":
        envs = gym.vector.AsyncVectorEnv(
            factories, context="spawn", shared_memory=True, **kwargs
        )
    else:
        envs = gym.vector.SyncVectorEnv(factories, **kwargs)
    try:
        obs, infos = envs.reset(seed=[42 + i for i in range(args.num_envs)])
        action_rng = np.random.default_rng(123)
        completed = 0
        for _ in range(300):
            actions = action_rng.integers(0, 8, size=args.num_envs, dtype=np.int64)
            obs, rewards, terminated, truncated, infos = envs.step(actions)
            assert obs.shape == (args.num_envs, 26)
            assert obs.dtype == np.float32
            assert np.isfinite(rewards).all()
            ended = terminated | truncated
            completed += int(ended.sum())
            if ended.any():
                assert infos["_final_obs"][ended].all()
                for i in np.flatnonzero(ended):
                    assert infos["final_obs"][i].shape == (26,)
        print(f"{300 * args.num_envs} transitions; {completed} completed episodes")
    finally:
        envs.close()


if __name__ == "__main__":
    mp.freeze_support()
    main()
```

After saving that module, run from the repository root:

```powershell
.\RL_venv\Scripts\python.exe -m Training.mystic_vector_smoke --num-envs 1 --mode sync
.\RL_venv\Scripts\python.exe -m Training.mystic_vector_smoke --num-envs 4 --mode sync
.\RL_venv\Scripts\python.exe -m Training.mystic_vector_smoke --num-envs 4 --mode async
```

Windows workers use `spawn`: module imports must not start training, open a game
connection, initialize CUDA, or launch a window. Put all launch logic under the
main guard. Keep policy inference and GPU tensors in the parent process; workers
return CPU NumPy arrays. The constructor may run a temporary environment in the
parent to inspect spaces, so factory construction must also be safe there.
Do not repeatedly reseed after each episode: seed each worker once, then let
autoreset advance that worker's RNG stream. `envs.close()` must run on errors.

#### 7.3 Integrate the vector batch into PPO

`Training/PPO_server.py` currently asserts `num_envs == 1`, adds an observation
batch dimension with `unsqueeze(0)`, and has scalar episode accumulators. Its
rollout buffers already contain a `num_envs` axis. Audit the entire collection
path instead of just deleting the assertion; apply the same audit to recurrent PPO.

1. Add explicit simulation vector mode and count arguments to the Phase 6
   factory. Allow `num_envs > 1` only for simulation; preserve the live guard.
   Factor the smoke program's worker factory and info wrapper into reusable
   modules instead of maintaining separate training versions.
2. Read network dimensions from `single_observation_space` and
   `single_action_space`. Pass `(N,26)` observations through the policy in one
   call and send `(N,)` integer actions to `envs.step`. Do not add another batch
   axis, use `.item()` on batched actions/dones, or pass one action to every worker.
3. Store observations as `(T,N,26)` and actions, rewards, log probabilities,
   values, and masks as `(T,N)`. For feed-forward PPO flatten the first two axes
   only when creating optimizer minibatches. Maintain episode returns, lengths,
   component sums, and win counts per worker, resetting only the worker that ends.
4. Use `AutoresetMode.SAME_STEP` consistently, rather than relying on the current
   default `NEXT_STEP`. In the installed version, ended workers return their new
   episode observation; the transition's real final observation and info are
   under `infos["final_obs"]` / `infos["final_info"]`, with `_final_obs` /
   `_final_info` masks. Terminal metrics must come from final info, not reset info.
   Write a helper for extracting nested per-worker info and honoring its masks.
   Check each observation/normalization wrapper's support for this autoreset
   mode, and apply the same preprocessing to final observations used for values.
5. Bootstrap a true terminal transition with zero. Bootstrap a time-limit
   truncation using the value of its **final** observation, not the reset
   observation. For GAE, stop recursion across both termination and truncation:
   `delta = reward + gamma * (1 - terminated) * V(next_transition_obs) - V(obs)`;
   `advantage = delta + gamma * lambda * (1 - ended) * next_advantage`.
   At a rollout boundary, still bootstrap continuing environments normally.
   If using reward correction for truncation instead, do not also bootstrap it
   a second time in GAE. Test hand-calculated terminal/truncated returns.
6. Keep LSTM hidden/cell state separately for each worker. Compute truncated
   final-state values with the appropriate pre-reset recurrent context; then
   reset only ended workers before acting on their new observations. Preserve
   temporal ordering within recurrent minibatches; do not randomly flatten
   individual timesteps as in feed-forward PPO.
7. Record `batch_size = T * N`. For example N=8 and T=128 yields 1024 transitions
   per rollout. Increasing N changes PPO batch size unless T is adjusted, which
   changes optimization frequency and memory use. Keep total transition budgets
   explicit, ensure valid minibatch divisibility, and compare learning outcomes
   as well as speed. Do not silently treat vector calls as individual transitions.

The final-observation handling above is verified against the installed vector
implementation; use the [Gymnasium autoreset guidance](https://farama.org/Vector-Autoreset-Mode)
when changing versions or wrappers. Do not mix old `final_observation` examples
with the installed API's `final_obs` keys without a compatibility layer.

#### 7.4 Benchmark and select the worker count

- Add a benchmark module separate from PPO: seeded random actions, warm-up,
  fixed transition count, repeated measurements, and no rendering/sleep/live I/O.
  A 200 ms simulated decision must execute as fast as CPU work permits, not wait
  200 ms of wall time. Exclude process startup from steady-state SPS but report it.
- Compare direct single-env, sync, and async at 1, 2, 4, 8, and then larger N only
  if hardware and memory allow. Start near physical CPU core count for async,
  leaving capacity for PPO, and measure before selecting a default. Cheap worlds
  and small observations can make process communication costlier than the work;
  async is not guaranteed to outperform sync, nor does N guarantee N-fold speedup.
- Report transitions/second as `vector_steps * N / elapsed_seconds`, vector calls
  per second separately, p50/p95 batch latency, CPU utilization, RAM, and startup
  time. Repeat with PPO to measure end-to-end transitions/second including policy
  inference, rollout transfers, advantage calculation, and optimization.
- Keep shared observation memory enabled for the supported Box space. Rich info
  still travels through process communication, so benchmark compact versus full
  diagnostics. Start with copied outputs; only use `copy=False` after proving
  rollout buffers own their data and cannot be overwritten by the next step.
- Control CPU thread oversubscription: benchmark learner thread limits, and keep
  worker numerical-library threads small where appropriate. One process per core
  each spawning many threads can reduce throughput. Do not import Torch/CUDA into
  workers just to run this pure Python simulator.
- Save hardware, package versions, backend, N, T, diagnostics mode, seeds, reward
  profile, SPS, and learning outcomes in a benchmark table. Select measured
  settings; do not put an unverified numerical SPS target into the exit gate.

#### 7.5 Correctness and checkpoint tests

- Compare every worker with an independent single environment using the same
  initial seed and predetermined action stream, including matching episode resets.
  Repeat across sync and async; scheduling differences must not change mechanics.
- Force one worker to die, one to hit the time limit, and another to continue in
  the same batch. Verify final observations, masks, rewards, counters, and LSTM
  state resets. Verify one world's reset does not reset or reseed another world.
- Test compact info on mixed None/integer targets and heterogeneous terminal
  states. Validate all returned observation shapes, bounds, and dtypes.
- Capture each worker's state/RNG if promising exact checkpoint resume. Otherwise
  label resumed runs as starting fresh episodes. Changing worker count on resume
  changes stream assignment and is not an exact continuation.
- Run bounded PPO training and evaluation with one and multiple workers, and
  verify save/load, finite gradients/losses, parameter updates, and process cleanup
  after normal exit, interruption, and worker errors.

Exit gate:

- synchronous and Windows-spawn asynchronous vector smoke programs pass;
- vector PPO trains on the current simulator with correct terminal bootstrapping,
  independent episodes, and recurrent-state isolation;
- sync/async deterministic worker replay matches the single-env reference;
- a benchmark table identifies the fastest stable settings on this machine,
  with engine SPS and end-to-end PPO SPS measured separately;
- eight-action checkpoints remain compatible with Phase 6 simulator evaluation.
  Vectorization must not change game mechanics to achieve higher throughput.

### Phase 8 - Test the live ZMQ contract, payload differences, and sim-to-game gap

Primary deliverable: a live-versus-simulator comparison report, with repeatable
regression cases for each discovered discrepancy. Matching the connected live
version takes precedence over further optimizations to the Phase 7 vector path.
This is the first phase that implements/tests a simulation-policy live adapter
or claims live contract compatibility. Phases 6 and 7 establish training and SPS
on the current simulator without requiring those live checks.

Before running policy-driven live sessions:

- Implement an explicit eight-action live Mystic adapter over the existing ZMQ
  request/response path. Keep the legacy 11-action default available for existing
  models; do not silently reinterpret output indices. Validate checkpoint action
  labels, observation schema, and preprocessing before transmitting actions.
- Build a payload-difference matrix from real captures and simulator outputs:
  raw field names, nesting, types, missing/null fields, ID representation, coordinate
  and direction conventions, HP/MP units and normalization, map ID, entity
  ordering, nearest-five selection/padding, and distance semantics. Compare raw
  payloads separately from the final encoded 26-value observation; equal shape
  does not imply equal meaning.
- Add mocked ZMQ and saved-payload tests for serialization, action labels,
  request/response correlation, reset handshakes, missing/malformed fields,
  duplicate/stale messages, and timeout/disconnection behavior. Unknown state
  must remain explicit rather than becoming guessed zeros or inferred kills.
- Run a bounded real connection smoke test after mocked tests pass. If game
  access is unavailable, mark the live part of Phase 8 as unvalidated. Mocked
  transport success is not evidence that the simulator matches the game.
- Compare live task/reward behavior separately, including legacy HP-delta and
  enemy-disappearance heuristics. Use comparable outcome metrics when live
  telemetry cannot support simulator-style event rewards. Loading a checkpoint
  does not restore or roll back live game state.

1. Capture bounded live map-53 sessions at the existing ZMQ boundary. Store raw
   requests/responses, request IDs, action index and label, client/server and
   receive timestamps where available, encoded observations, and visible entity
   state. Pair actions with their actual responses, accounting for duplicate or
   stale ticks and variable latency. Keep capture optional and outside simulation
   hot paths; a simple JSONL format is sufficient, without Minari.
2. Use scripted scenarios before policy-driven sessions: each movement direction,
   occupied cells, rectangular cast edges, nearest-target ties, all three spells,
   insufficient MP, cooldown edges, Acid refresh, partial NPC attack-timer reset,
   regeneration, kills, and 50-second respawns. Record which expected fields are
   directly observed versus inferred or unavailable from the live payload.
3. Compare two layers separately. First validate ZMQ action serialization and
   live observation parsing against the simulator contract. Then compare combat
   and movement transitions from matched initial snapshots and action sequences.
   Snapshot import must explicitly initialize known timers, stats, effects, and
   entity identity; do not fill hidden server state with guessed certainty.
4. Require exact matches for deterministic observable rules such as action
   mappings, resource ordering, legal movement, and observation encoding. Define
   timestamp tolerances from server tick cadence and capture uncertainty. Where
   server RNG state or draws are unavailable, compare controlled roll branches
   or distributions over repeated runs; equal client seeds do not imply equal
   live/server random sequences.
5. Report the first divergent observable transition with timestamp, request ID,
   action, before/after state, expected/actual values, tolerance, and evidence.
   Classify transport/parser, mechanics, task/reward, and unobservable-state
   mismatches separately. Retain held-out sessions so fixes are not validated
   only on the same traces used to develop them.
6. Turn confirmed mismatches into focused Python regression fixtures and fix the
   simulator or live adapter at the responsible layer. Check the effective
   level-150 Innie template, movement interval per life, 226 HP/1664 MP per
   two seconds, fixed-before-percentage MP costs, 16-X/10-Y casting rectangle,
   and per-caster Acid refresh against the connected version. Preserve old
   evidence when live behavior differs. Terrain collision is always enabled in
   map53; test the actual obstacle layout against the live task.
7. Evaluate frozen simulator-trained PPO checkpoints through the live ZMQ
   adapter using deterministic action selection. Record both sides' task and
   reward profiles, kills, survival, damage, spell success, and action timing.
   Keep map, class/stats, gear, spawn/task constraints, and episode rules as close
   as possible; report unavoidable differences. Feed failures back into parity
   fixtures and simulator retraining before using live fine-tuning to compensate.
8. Re-run the Phase 7 vector benchmarks after parity fixes. Optimize measured
   bottlenecks without changing RNG consumption or event ordering. NumPy batching,
   compiled extensions, and a C++ core remain optional further work; the required
   sync/async vector training path is already established in Phase 7.

Exit gate:

- the eight-action live adapter passes contract tests and rejects incompatible
  checkpoints before sending actions;
- payload-difference fixtures cover raw transport data and encoded observations,
  with every required difference either reconciled or explicitly reported;
- real ZMQ captures produce a reproducible comparison report with defined
  tolerances, explicit unknowns, and no unexplained failures on required cases;
- action/observation compatibility and agreed observable mechanics pass held-out
  live scenarios; unsupported hidden-state claims are excluded;
- a frozen simulator-trained PPO checkpoint completes bounded live evaluation,
  with simulator/live outcome differences measured and remaining gaps recorded;
- every confirmed fix has a regression test, and any optimization preserves
  deterministic simulator traces. BC/Minari availability is not an exit gate.

### Deferred - Behavior cloning, Minari, and offline learning

Keep existing offline tools functional, but defer new BC training, eight-action
Minari dataset production, legacy demonstration migration, dataset publication,
and AWAC/offline experiments until PPO training and live parity are established.
Existing action/schema safeguards remain in place. Lightweight ZMQ fidelity
captures are validation evidence and do not require a training dataset pipeline.

## First implementation slice

Build the first reviewable vertical slice in this order:

1. Create `Mystic_Sim` with actions, frozen configuration, state dataclasses,
   map loader, scenario builder, and pure observation encoder.
2. Register `YugenSaga/MysticSim-v0`; implement reset only and make map/reset/
   observation tests pass.
3. Add the scheduler and movement actions. Implement NPC movement and aggro with
   attacks temporarily emitting trace-only events.
4. Port scaling and defensive combat, then turn trace-only NPC attacks into real
   damage, death, and respawn.
5. Implement Arcane Blast end to end. This exercises targeting, cooldowns,
   resources, RNG, damage, rewards, and kills before AoE/effects add complexity.
6. Add Acid Cloud and its timed effect, then Tempest Inferno and partial-radius
   allocation.
7. Connect `combat_reward_v1`, termination/truncation, structured `info`, and the
   Gymnasium checker.
8. Complete Phase 6 PPO training, checkpointing, resume, and evaluation solely
   on the current Mystic Sim environment.
9. Complete Phase 7 compact diagnostics, sync/async vector PPO, reset/GAE tests,
   and throughput benchmarking.
10. Complete Phase 8 live adapter/contract tests, payload-difference tests, live
    capture, mechanics comparison, and frozen-policy sim-to-game evaluation.
    BC/Minari remains deferred.

The first mergeable milestone ends after step 3: a deterministic 100x100 map-53
environment with 80 non-overlapping Innies, seeded reset, the exact observation,
four movement actions, source-shaped NPC pursuit/aggro, and a 200 ms event clock.
The second milestone ends after step 5 and is the first useful combat-training
environment. The third milestone completes all three spells and training
integration.

## Definition of done for `MysticSim-v0`

- The environment runs headlessly with no socket, rendering, sleep, or hot-path
  file writes.
- Its public contract is `Discrete(8)` plus the versioned 26-value observation.
- `map53.json` drives dimensions and all template-5300 spawn regions.
- Reset, event scheduling, movement, target selection, combat, cooldowns,
  effects, death, and respawn are deterministic for a seed.
- The three spell translations and all source-derived formulas have numeric and
  state-transition tests.
- Terrain and occupancy collision are enforced in map53.
- Gear is disabled cleanly and can later be enabled through configuration.
- Five kills wins, player death loses, and 256 actions truncate.
- Gymnasium validation and the repository test suite pass.
- Checkpoints record enough schema/profile metadata to prevent incompatible
  live, legacy-11-action, or higher-fidelity models from being mixed silently.
- PPO trains on the simulator without sockets, saves/reloads compatible models,
  and has a tested eight-action live ZMQ evaluation path.
- Sync/async vector PPO supports independent worlds and correct episode handling,
  with measured environment and end-to-end training throughput.
- Required live map-53 scenarios have held-out comparison reports with explicit
  tolerances and unknowns; a frozen PPO policy is evaluated on the connected
  live version. Mock transport tests alone do not satisfy live fidelity.
- Behavior cloning and Minari migrations are deferred, not completion blockers.
