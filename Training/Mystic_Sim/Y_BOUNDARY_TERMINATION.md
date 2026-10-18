# Task Y-boundary termination

Crossing the configured inclusive task interval (default Y=30..86) is now
`terminated=True`, `truncated=False`, outcome `loss`, reason `y_boundary`.
Death and reaching the kill goal retain their existing priority over the Y check.
Physical terrain/map collisions still block movement without ending the episode.

The existing positioning penalty is retained: -100 per decision at Y=30/86,
-200 at Y=29/87. No additional death penalty is charged for a Y-boundary loss.
Normal damage, collision, cooldown and time components still apply. The final
reward can therefore differ from -200. The 1,024-step external cutoff still
truncates, charges -25 once and bootstraps. It does not also charge -25 on a
boundary ending. Existing PPO termination masks remove the bootstrap from task
failures and prevent GAE from crossing resets; no PPO formula change is needed.

## Continuing an existing policy

Engine compatibility checks remain strict. Explicitly migrate an old checkpoint
before loading it on the changed task. The migration allows only the audited
old/new engine hashes and requires all other contract metadata to match. It
preserves actor/critic/LSTM weights, optimizer, RNG and counters, records the
source and semantic change, and refuses to overwrite an existing file.

```powershell
python -m Training.Mystic_Sim.migrate_y_boundary runs/mystic_sim/full/full80_7p2m_cp04_continue_1m_lr1e4_seed42/checkpoints/checkpoint_10.pt --output runs/mystic_sim/transfers/source_8p2m_cp10_y_termination.pt
```

Once the migrated checkpoint exists, the prepared one-million-transition branch
uses starting 8.2M CP10, constant LR 0.0001, training seed 42, and the same
evaluation seeds and PPO settings as the earlier branch:

```powershell
python -m Training.Mystic_Sim.curriculum_run --config Training/Mystic_Sim/configs/full80_8p2m_cp10_y_termination_1m.json
```

Use continuation in a new directory, not resume into the historical run. The
old files and reports still describe the original truncation semantics. Evaluate
both the migrated frozen baseline and newly trained policies on the new task
for comparisons. The migrated critic initially reflects the old bootstrapped
targets and must adapt; migration does not retrain it. Other old checkpoints
also require migration for the viewer/comparison tools after this engine change.

Tests verify both Y edges, retained penalty amounts, loss diagnostics, reset
requirements, no value bootstrap on Y failure, continued time-limit bootstrapping,
and migration preservation/rejection behavior.
