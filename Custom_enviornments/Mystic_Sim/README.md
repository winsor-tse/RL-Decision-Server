# Mystic simulator: Phase 0

This package freezes contracts and fixtures before engine implementation.
Importing `Custom_enviornments.Mystic_Sim` registers `YugenSaga/MysticSim-v0`.
The registered scaffold declares eight actions and 26 float32 observation values,
but `reset()` and `step()` raise `NotImplementedError`. Phase 1 implements reset;
later phases implement movement and combat. The scaffold is not trainable.

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

`map_loader.load_map53(path)` validates the supplied baseline export. It returns
raw map data, not runtime state. Geometry normalization and spawning arrive in
Phase 1. Changes to map dimensions, properties, spawn layout, or quantities fail
explicitly so a changed map requires review. Terrain remains disabled in the
baseline even though the blocked layer is validated.

`mechanics_manifest.yaml` records baseline parameters and provenance. It preserves
the JSON cap 68 separately from the confirmed effective level 150, the corrected
50-second respawn, one movement-speed draw per life, and fixed MP deductions
before spell calculations.

```powershell
.\RL_venv\Scripts\python.exe -m unittest Tests.test_mystic_sim_contracts
```
