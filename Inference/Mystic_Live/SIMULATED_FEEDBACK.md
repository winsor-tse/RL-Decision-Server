# Run existing 80-input weights using simulator-estimated history

Implemented 2026-10-23 at the user's request to assume simulator/game parity.
This keeps the existing 80-input checkpoint and feature ordering. No retraining,
71-input replacement or server feedback producer is required for this mode.

```powershell
.\RunInference.ps1 -Checkpoint "PATH_TO_CHECKPOINT" -AssumeSimParity
```

The launcher loads the actor, waits for its ZMQ bind, then automatically starts
the existing bridge with `--single-run`. It logs startup/errors/shutdown without
per-tick messages and cleans up both processes when inference ends. Load the
updated game extension and enable its AI loop as with ordinary live inference.
Supply your own
checkpoint path. The flag applies only to 80-input policies and is mutually
exclusive with `--confirmed-feedback`. The checkpoint launcher defaults to sampled
actions. Select explicitly with `-ActionMode sampled` or `-ActionMode greedy`.
The launcher defaults to CUDA; use `-Device cpu` for CPU inference or
`-Device cuda:1` to select a different GPU. CUDA requires a compatible GPU and
CUDA-enabled PyTorch installation.
The direct server CLI retains its greedy default and accepts `--sampled`.
Run the launcher again
for a new run. Startup timeout now reports absent snapshots or the rejected
input reason and exits with a failure code. This launcher does not open or log
into the game for you.

## How the existing history is produced

`SimulatedFeedbackInputs` uses the selected checkpoint's simulator configuration
and the existing training `ObservableHistory` encoder:

| Input | Source in this mode |
| --- | --- |
| Base player/enemy features | Current live snapshot |
| Terrain rays | Shared map and existing sensor code |
| Previous action | Actual policy choice, including a no-target attempt |
| Movement applied/blocked | Simulator `collision_kind`, shared terrain/bounds and visible living actor occupancy |
| Cast applied/rejected | Simulator `cooldowns.rejection`, live HP/MP and the explicit live target |
| Slot/family/global cooldowns | Simulator `cooldowns.start` and locally retained deadlines |
| Successful-cast age/seen | Predicted accepted casts and elapsed client snapshot time |

Predictions use the same fixed MP/HP costs and rejection ordering as the simulator.
Targets come from the same explicit command adapter used for dispatch. Each fresh
snapshot replaces positions, resources and enemies; only the cooldown/history
state persists. No independent NPC world, damage RNG or HP/MP prediction runs.
This avoids replacing observed state with an independently evolving simulation.

The model assumes an action happens at its input snapshot time and advances
ages by the next snapshot's timestamp difference. It does not force 200 ms when
the observed interval differs. Each chosen action updates history once, on the
next fresh snapshot. Duplicate requests and redraws do not advance it.

## Starting assumptions

Start after all pre-existing spell cooldowns have expired, and let the policy
control actions for the run. The tracker initializes cooldowns as ready and cast
history as unseen, matching a fresh simulator episode. It cannot recover timers
from casts made before launch or manual casts outside the policy loop.

The assumed parity includes the configured spell rules, no hidden immunity,
the observed occupancy at action time, and commands reaching the game as intended.
Delivery delay, a target changing before execution, or an unseen blocker can make
an estimate differ from the actual game. This mode accepts that uncertainty under
the explicit parity assumption; it does not claim to have measured server success.

## Inspection and validation

Policy inputs expose `history_source=simulator_estimated`, timing basis, initial
cooldown assumption and the estimated transition. Final snapshots retain local
cooldown deadlines and pending/latest estimates under `history_model`. Issued
commands carry a separate `simulator_estimate`; `execution_outcome` remains null
and confirmed-history records remain empty. Actual acknowledgements are not
fabricated or mixed with predictions.

76 focused tests passed, including complete 80-value tensor comparisons against
simulator sequences, terrain/bounds/occupancy, target absence, slot/family/global
cooldown behavior, resource resynchronization, cast ages, duplicate/death handling,
checkpoint/CLI selection and ordinary WebSocket/ZMQ transport. No real game run
or checkpoint selection was performed.
