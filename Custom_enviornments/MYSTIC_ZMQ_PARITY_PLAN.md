# Mystic: minimal simulator-to-game policy deployment plan

Revised 2026-10-22. This replaces the former Phase 8D-8H deployment roadmap.
Phase 8A-8C implementation is retained. This document proposes remaining code
changes; it does not mean those changes or a live run have been completed.

## Objective

Load a simulator-trained checkpoint and control one live player through the
existing browser -> WebSocket bridge -> ZMQ path. Each fresh state produces one
policy decision and one reply. Train in the simulator; run the frozen actor in
the game. Start on a living player on map53 and stop on death, manual stop,
disconnect, or an unsupported map change. No automatic reset or respawn.

The runtime is:

`live worldState -> shared observation encoder -> frozen actor -> command reply`

The shared map supplies terrain sensors and optional rendering. Do not step an
independent simulator to predict live HP, enemies, cooldowns, or movement.

## Keep the completed work

**Preserve, then correct:** [the Phase 8A-8C compatibility audit](../Inference/Mystic_Live/LIVE_COMPATIBILITY_REVIEW.md)
found reproducible live-path blockers despite all 47 local tests passing.
"Implemented" below means local components exist, not that they work against
the current server. Include the audit's feedback-inventory corrections, browser
self-target/stop handling and bridge progression checks in the steps below.

- **8A:** baseline/checkpoint inventory, optional bridge captures and evidence
  tests. See [PHASE_8A.md](../Inference/Mystic_Live/PHASE_8A.md).
- **8B:** single-run lifecycle, stop behavior, bridge integration, final snapshot,
  and existing confirmed-feedback mode. See [PHASE_8B.md](../Inference/Mystic_Live/PHASE_8B.md).
- **8C:** shared 56/80 encoders, seven-action adapter, map loader, terrain sensors,
  strict history validation and parity tests. See [PHASE_8C.md](../Inference/Mystic_Live/PHASE_8C.md).

Preserve these implementations and their strict-mode tests. Extend the lifecycle
and CLI rather than replacing this stack or modifying the legacy live trainers.
The phase documents describe the existing implementation, including its current
limitations; this plan governs new work and supersedes their old phase ordering.

Current status: the CLI is observation-only, the actor hook is present, and
checkpoint loading is already implemented in the simulator evaluator. However,
SingleRun currently waits for an execution acknowledgement after EVERY action,
even with 56 inputs. The ordinary extension does not produce this acknowledgement.
Adding a model loader alone will therefore not produce continuous live inference.

## Input compatibility: decide from the checkpoint

Inspect the explicitly supplied checkpoint before sending any gameplay command.
Use its architecture, feature schema/order, normalization and action labels.
Never select the newest checkpoint implicitly or resize a network to fit it.

| Checkpoint | Live requirements | Deployment route |
| --- | --- | --- |
| Existing 56-input feedforward policy | Current visible world state | Use the existing encoder and the new snapshot-driven lifecycle mode. No execution-outcome producer is needed. |
| Existing 80-input recurrent policy | Base state, terrain, prior action, applied/blocked/cooldown flags, successful-cast ages | Retain strict feedback mode. Existing weights need the missing outcome information for matching input semantics. |
| New recurrent policy using live-available history | A separately versioned shared training/live encoder | Train a new checkpoint with that contract, then use snapshot-driven inference. This is the recommended route when server feedback is unavailable and recurrence is desired. |

LSTM hidden/cell state does not itself require execution acknowledgements. The
existing handcrafted history features do. The map already supplies the eight
terrain values; it cannot establish whether the server accepted a spell.

Do not pad 56 features to 80, zero missing outcome features, or represent command
submission as successful execution. An 80-input checkpoint cannot be promised
exact deployment on the current payload without resolving this dependency.

## Step 1 - Connect the existing frozen actor loader

Implemented 2026-10-23: `policy.py` reuses the evaluation loader and runs the
frozen actor only; the CLI accepts explicit checkpoints, greedy/sampled actions
and device selection. Input size is inferred and conflicting overrides fail.
See [STEP_1.md](../Inference/Mystic_Live/STEP_1.md) for usage and validation.
Step 2 now selects snapshot progression for 56-input checkpoints by default.
`--confirmed-feedback` selects the strict lifecycle, not a telemetry producer.

Code: add `Inference/Mystic_Live/policy.py`; extend `server.py`.

1. Reuse `Training/Mystic_Sim/evaluate.py:load_evaluation_agent` and its existing
   checkpoint validation. Reuse model classes and saved normalization. Loading
   immutable simulator configuration/map data is fine; no live simulation loop.
2. Implement the existing `initial_state()` / `choose(...)` actor interface and
   return `PolicyChoice`. Use eval/inference mode and normalize raw inputs once.
   For recurrent models, retain actor hidden/cell state between decisions and
   reset it only at a new run. Do not evaluate the critic per live decision.
3. Make `--checkpoint PATH` and `--observe-only` mutually exclusive CLI modes.
   Infer the inference input contract from the checkpoint; reject conflicting
   overrides. Default to greedy actions; provide an explicit sampled option.
4. Inspect compatibility before binding/dispatching: actions, schema, scale,
   supported architecture and map/sensor configuration. Report an unsupported
   80-input deployment clearly rather than silently launching a stalled run.
   A strict feedback run may wait for its required initial telemetry as today.

Acceptance: fixed input sequences produce the same normalized inputs, logits,
actions and recurrent states as the simulator actor within numerical tolerance.
The loader does not restore optimizers or modify weights. Existing observation
mode still works.

## Step 2 - Allow continuous decisions from ordinary ZMQ snapshots

Implemented 2026-10-23. The CLI defaults compatible 56-input policies to snapshot
mode; existing 80-input policies require explicit confirmed feedback. Direct
`SingleRun` callers keep the strict default. The bridge accepts explicitly marked
snapshot commands; issued-command reports keep execution outcomes unknown.
See [STEP_2.md](../Inference/Mystic_Live/STEP_2.md) for launch and test coverage.

Code: extend `Inference/Mystic_Live/lifecycle.py`, `inputs.py` and `server.py`.

1. Add explicit snapshot-driven and confirmed-feedback progression modes.
   Keep the existing strict behavior as the default for existing direct callers;
   the checkpoint-aware CLI selects a compatible mode explicitly.
2. In snapshot mode, the next valid fresh request can trigger the next decision
   without an `execution` record. Record the previous command as sent, with its
   execution outcome unknown. Do not call the confirmed-history updater or
   synthesize an acknowledgement. Snapshot mode must reject the existing
   outcome-dependent 80-input contract.
3. Preserve stale/duplicate request handling, connection/run ownership, one
   reply per request, inert replies while waiting, and death/interruption stops.
   No extra actor call for duplicate observations, diagnostic reads or redraws.
4. Record actual observed intervals for diagnostics. Retain the existing browser
   cadence for the initial run; do not require exact server-clock synchronization
   for a snapshot-only policy. Investigate latency if live behavior warrants it.

Acceptance: an actor handles several ordinary captured/synthetic world-state
requests in sequence without acknowledgements, produces one command per fresh
request, and stops before any command on death. Strict-feedback tests still pass.
Test duplicate requests, disconnect and timeout; distinguish transport progress
from game execution success.

## Step 3 - Resolve only the missing inputs and command semantics

Implemented locally 2026-10-23: explicit targeting/no-target handling, browser
self-casts and stop-generation checks, capability forwarding, and an opt-in
71-input recurrent training/live contract. Checkpoint selection stays entirely
in the user's CLI. See [STEP_3.md](../Inference/Mystic_Live/STEP_3.md) for commands,
browser reload instructions and validation limits. Existing 80-input weights
still need a real outcome producer; they are not converted. No real game episode
has been run, and no production checkpoint was trained or selected.

### If deploying an existing 80-input checkpoint

Keep the current strict adapter. Wire a real accepted/rejected-action producer
and its forwarding through the game/extension, then verify it against the input
contract. The existing Phase 8C telemetry protocol is one concrete supported
implementation. Server-clock fields are requirements of that adapter, not of all
policy inference. Any simpler replacement still needs a tested mapping to the
same history semantics and a coherent time basis for cast ages.

Do not make server instrumentation a prerequisite for the 56-input route.
If server changes are unavailable, this route remains blocked; use a compatible
56-input checkpoint or the new recurrent contract below. Merely detecting this
blocker is not completion of deployment.

### If recurrence is needed without server instrumentation

Add a new, explicitly named live-observable history contract shared by training
and inference. A small initial design is the existing 56 base values, eight
static terrain values, and seven previous-chosen-action flags (71 total).
Previous action means the chosen/submitted command even if it failed; initial
flags are zero. No successful-cast ages or rejection flags are included.

Implement the encoder in the training history layer and reuse it live. Add an
opt-in training configuration, checkpoint metadata/schema and loader support.
Use the existing flat recurrent encoder initially: the current cross-attention
encoder explicitly requires the old 80-feature layout. Do not change its layout
or reinterpret old checkpoints. Preserve the existing training defaults and
legacy 80-input evaluation path.

Train a new checkpoint with this contract, verify shared encoding and actor
replay, then deploy it via snapshot mode. Training a new policy is real work;
renaming metadata or dropping input columns does not convert old weights.
This branch is required only if the selected deployment route needs it.

### For every route: verify the seven commands

The action order is `up`, `down`, `left`, `right`, `castSpell:1`,
`castSpell:2`, `castSpell:3`. Legacy action 4 means attack and must not leak into
this route.

Reuse simulator targeting rules where observable: eligible living monsters,
Euclidean ranking, deterministic ID ties, inclusive 16-X/10-Y window and self
fallback for the appropriate spells. Observation ranking remains Manhattan;
these are different purposes. Use explicit target IDs supported by the inspected
browser parser; verify self targeting and no-target behavior in the actual
extension. Preserve the original wire entity identity when normalizing sort IDs.
A no-target spell must not accidentally fall back to another browser-selected
entity. Use an inert command when the simulator action has no executable target.

Make targeted changes in the live command adapter and, only where required, the
sibling browser extension. Keep numeric policy action labels separate from the
wire command containing its target. Confirm movement directions and casting
facing behavior with short scripted checks. Fix concrete discrepancies; do not
block deployment on a complete combat-mechanics audit.

Acceptance: all seven commands, targeting ties/edges, no-target and self-fallback
cases behave as documented. Where extension/server work is needed, name the
actual changed files and validate forwarding through the bridge.

## Step 4 - Run one policy episode and document the launch

Current CLI for a compatible 56/71-input checkpoint and the updated extension:

```powershell
.\RL_venv\Scripts\python.exe -m Inference.Mystic_Live.server --checkpoint PATH_TO_COMPATIBLE_CHECKPOINT
.\RL_venv\Scripts\python.exe -m Automation.Bridge.ws_zmq_bridge --single-run
```

Run in separate terminals and connect the existing extension. A checkpoint using
confirmed feedback must explicitly select that mode and have its producer wired;
ordinary snapshot mode is the simplest deployment path.

With the existing permanent terminal latch, each new run requires restarting
both the backend and bridge. Restarting only the policy process is insufficient.

Before contacting the game, test the actor and snapshot lifecycle together over
local WebSocket/ZMQ with representative payloads. Then launch one real episode.
Confirm repeated policy commands reach the player, death/manual stop prevents
further gameplay commands, and final state/stop reason is retained. Record the
checkpoint, input mode, selected actions and observed timing. Captures are
optional diagnostics; a full evidence dossier is not a launch requirement.

Update the CLI instructions and phase-status notes after the implementation.
The watch UI may consume detached live snapshots later; rendering must never
advance actor memory or block the decision loop. UI integration is not a
prerequisite for proving policy-to-game execution.

Completion means a selected compatible checkpoint actually controls the player
through the existing transport and stopping works. Offline tests alone establish
local readiness, not a completed live run. If game access is unavailable, report
that boundary and provide the tested launch commands without claiming success.

## Order and scope

Implement Steps 1 and 2 first. Complete the selected input route and narrow
command checks in Step 3, then run Step 4. Preserve Phase 8A-8C code throughout.
Do not choose or retrain a different policy silently to bypass incompatibility.

Defer rewards, kill attribution, exhaustive damage/cooldown/regen/NPC/respawn
calibration, multi-episode studies, simulator performance work and a new watch UI.
Investigate those only when a concrete live result calls for them. Do not rebuild
the Gym framework, ZMQ bridge or map system.

## Local references

- Shared map: `Tests/fixtures/simulation/map53.json` and
  `Custom_enviornments/Mystic_Sim/data/map53.json` (user-confirmed live layout).
- Payload shape: `Tests/fixtures/simulation/example_full_state.json`; its example
  map/player values are not the actual map53 run. It is the nested world state.
- Existing adapters: `Inference/Mystic_Live/{server,lifecycle,inputs}.py`.
- Model loading: `Training/Mystic_Sim/evaluate.py:load_evaluation_agent`.
- History and architecture: `Training/Mystic_Sim/{history,recurrent,core}.py`.
- Target selection: `Custom_enviornments/Mystic_Sim/targeting.py`.
- Transport: `Automation/Bridge/ws_zmq_bridge.py` and sibling
  `../yugen-battler/yugen-battler/battler.js`.

The source has been inspected; the actually loaded extension/game version and
real execution remain to be checked during deployment.
