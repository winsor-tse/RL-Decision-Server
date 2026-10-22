# Watch a checkpoint on a chosen seed

From the repository root, with the training and Pygame dependencies installed:

```powershell
.\Watch_Mystic_Policy.bat runs/mystic_sim/full/full80_7p2m_cp04_continue_1m_lr1e4_seed42/checkpoints/checkpoint_10.pt --seed 91000 --mode sampled
```

This example watches the starting 8.2M CP10. Any compatible checkpoint can be
supplied; no checkpoint path is hard-coded in the launcher. Change `--seed` to
the episode seed from a comparison report, or use `--mode greedy` for argmax.
Sampled mode uses the same seed-derived action RNG as `compare`/`evaluate`.
CPU inference is the default; `--device cuda` is optional, but device/kernel
differences can affect exact replay. Match the evaluation device for comparisons.

The portable module equivalent is:

```powershell
python -m Training.Mystic_Sim.watch PATH_TO_CHECKPOINT --seed 91000 --mode sampled
```

Relative checkpoint paths resolve against the repository root. The Windows
launcher uses the active virtual environment, then RL_venv, then .venv.

You can also name the argument explicitly, with options in any order:

```powershell
.\Watch_Mystic_Policy.bat --seed 2000 --mode sampled --checkpoint "runs/my_run/checkpoints/checkpoint_09.pt"
```

The positional `PATH` and `checkpoint PATH` forms remain supported. Supply only
one checkpoint. Quote paths containing spaces. If you omit the `checkpoints/`
subfolder and that exact file exists there, the viewer prints and uses its
resolved location. It never substitutes a different checkpoint number.

- Space: start/pause (starts paused).
- N: advance exactly one policy decision while paused.
- R: reset the entire environment and replay the same seed, resetting LSTM
  hidden states, observation history and policy-sampling RNG.
- Shift+R: reset with the next seed.
- Minus/plus: playback speed; this does not change the simulation decision size.
- Tab: map overview. Mouse wheel over the map: zoom. B: spawn boxes. C: casting range.
- Click inspector tabs: Summary, Player, Enemies, History, Inputs. Wheel over the
  inspector or Page Up / Page Down: scroll. Home: return to the top.
- Click Current state / Previous state to inspect the resulting world or the
  exact earlier state that produced the displayed action probabilities.
- Esc: exit. Manual movement and spell keys are disabled.

The viewer retains the checkpoint's scenario via the existing evaluation loader.
It advances one policy action per 200 ms simulation decision, stops on the actual
environment termination/truncation, displays steps/reward, and prints the episode
outcome to the console. It never trains or writes checkpoint files. No free-play
override is used. Existing `Play_Mystic_Sim.bat` remains the manual viewer.

Headless rendering smoke check (no interactive window):

```powershell
python -m Training.Mystic_Sim.watch PATH_TO_CHECKPOINT --seed 91000 --smoke-test
```

## Watch dashboard

The resizable window reserves the lower-left area for all seven action
probabilities and the right side for a scrollable state inspector. The map stays
visible while inspecting inputs. Probability percentages come directly from the
same softmax distribution used to select the action: the selected action has a
gold marker and `*`; the highest probability has a gold percentage. In sampled
mode these may be different actions. Greedy mode still shows the full underlying
distribution rather than replacing it with a one-hot vector. No action masking
is introduced. Entropy is measured in bits over the seven actions.

Probabilities are retained from the latest executed decision, with its **input
step** labeled explicitly. The current world is one decision later. Select
**Previous state** to see the state used to choose the last action, or **Current state** for
the latest player/enemy/history values. Before the first decision there is no
distribution to display; press N while paused to inspect one. Pausing preserves
the last distribution. Reset clears it, history, and recurrent memory.

The inspector provides:

- **Summary:** player HP/MP, position/facing/map, kill goal, population, target,
  action result/rejection, return, latest reward contributions, and event feed.
- **Player:** observed player state and separately labeled engine cooldown
  diagnostics. Exact cooldown timers are not supplied to the policy.
- **Enemies:** all ten nearest living observation slots, in policy distance/ID
  order, with distance, relative direction, HP/MP fractions and the first five
  signed offsets. IDs/target markers are diagnostic labels; slots 6-10 do not
  expose coordinate offsets to the model. Missing slots remain zero-padded.
- **History:** previous action/outcomes, confirmed successful-cast ages capped at
  ten seconds, cast-seen flags, and four static-terrain rays. Feed-forward models
  explicitly show that they have no history inputs.
- **Inputs:** every indexed raw and normalized input, grouped into enemy slots,
  relative offsets, history and terrain. All values can be reached by scrolling.

Rendering and navigation read detached snapshots without running inference or
advancing recurrent memory, sampling RNG, or simulation time. The same frozen
policy still drives one action per 200 ms simulation decision.

To export a dashboard preview without opening a window:

```powershell
python -m Training.Mystic_Sim.watch PATH_TO_CHECKPOINT --seed 91000 --smoke-test --screenshot runs/mystic_sim/watch_preview.png
```

## TODO: launch the same watch UI from the live server

Phase 8A [baseline and bridge evidence tools](../../Inference/Mystic_Live/PHASE_8A.md)
are implemented. They inventory/capture data only; the inference-driven LIVE
watch mode below remains TODO, and the deployment checkpoint is unselected.
Phase 8B [single-run lifecycle and observation-only server](../../Inference/Mystic_Live/PHASE_8B.md)
are also implemented. Final raw state/history/timing is retained in a detached
JSON report; live Pygame rendering and checkpoint-driven inference remain later
integration. A proposed execution-feedback consumer is tested, but the current
plugin still needs its authoritative feedback producer.

The current launcher watches a local simulator episode. Live-server integration
is still TODO; the dashboard's SIMULATION badge makes the source explicit.
`watch_state.py` defines detached `WatchSnapshot` and `PolicyDecision` display
records so the future live producer can use the same inspector/probability UI.

Confirmed live context supplied by the user:

- **Shared map:** [Tests/fixtures/simulation/map53.json](../../Tests/fixtures/simulation/map53.json)
  is the live map and has the same static obstacles as the simulation. Reuse
  this geometry/collision layer for the LIVE field view and terrain sensors;
  discovering or rebuilding live obstacles is not required. The fixture and
  [packaged simulator map](../../Custom_enviornments/Mystic_Sim/data/map53.json)
  also have matching SHA-256 hashes in this checkout.
- **ZMQ state reference:** the live full state has a structure similar to
  [Tests/fixtures/simulation/example_full_state.json](../../Tests/fixtures/simulation/example_full_state.json):
  `timestamp`, `player`, and `entities`, with coordinates, direction, HP/MP and
  their maximums, entity identity/type, and supplied distances. Use this as the
  parser/adapter reference for `worldState`, not as a fixed live snapshot.
  Its example `map3471`, positions and resource values are illustrative; the
  configured deployment here is map53.

The original reference is
[Simulation/Source-Code/Example_Full_State.txt](../../Simulation/Source-Code/Example_Full_State.txt).
It and `Tests/fixtures/simulation/example_full_state.json` have identical SHA-256
hashes in this checkout. The inspected JavaScript plugin's `buildWorldState()`
produces this structure. The ZMQ backend receives it nested under `worldState`
in an outer message with `type: "ai_tick"`, `requestId`, `pageUrl` and a separate
message `timestamp`; the WebSocket-to-ZMQ bridge forwards that envelope unchanged.

Static map geometry is shared. Current player/enemy positions and resources
still come from each live payload, rather than simulator reset/spawn state.

The intended live workflow is **one inference episode per explicit launch**.
Train and parallelize in the cheap local simulator, save a checkpoint, then run
that frozen policy against one player on the actual game server through ZMQ.
There is no live training, optimizer, rollout buffer, GAE, value bootstrapping,
training checkpoint resume, or automatic episode loop in this integration.
The existing legacy live trainers are separate entry points, outside this TODO.

The default sequence is:

1. The user positions the player in the intended live map and explicitly starts
   inference with a simulator checkpoint, optionally requesting the watch UI.
2. Validate the checkpoint's architecture, action labels, input schema and
   preprocessing. Wait for the first valid living-player snapshot; do not send
   a reset spell or change the game to manufacture a training-style start.
3. Initialize actor recurrent memory, observable history and any action-sampling
   RNG once for this run. Execute the policy in eval/inference mode. The critic
   and its recurrent memory are unnecessary for action selection; a loader may
   read a full training artifact without running the critic or restoring Adam.
4. For each fresh live state, correlate the previous action's confirmed result,
   update observable history, encode the next input, run the actor once, and send
   at most one selected command. Measure elapsed time; do not assume every live
   interval is exactly the simulator's 200 ms. Keep request correlation and stale
   message handling, even though no training transitions are being collected.
5. On a confirmed player-death snapshot, stop action selection and command
   dispatch immediately. Complete any pending ZMQ exchange with an explicitly
   inert transport reply if needed to preserve REQ/REP ordering. Retain the final
   snapshot and last decision in the viewer, labeled **PLAYER DIED / STOPPED**.
   Do not respawn, cast a reset spell, restart inference, or wait for another
   episode automatically. Respawning later does not reactivate the stopped run.
6. Manual stop ends the run. A disconnect, timeout, or unsupported map change
   ends it as interrupted; do not automatically reconnect and resume acting.
   Starting again requires an explicit new launch with fresh memory/history.

The simulator's five-kill goal, Y-boundary failure rules and 1,024-step cap are
training/evaluation task settings, not default live stopping conditions. Live
inference normally runs until death or manual stop. An optional evaluation goal
or duration limit must be explicitly configured and stop without starting a new
episode. Unknown map/input contracts still require stopping action dispatch.

Planned watch integration:

1. Add an inference-launch option that opens the existing watch application in
   a separate **LIVE** display mode. Publish actual actor probabilities, selected
   action, exact previous/current observations and scales, run/request IDs,
   execution outcomes and timestamps over a separate telemetry channel. The
   viewer must not consume the game's ZMQ REP stream, run another decision loop,
   or issue competing commands. Rendering must not delay inference.
2. Populate `WatchSnapshot` / `PolicyDecision`-compatible display data from the
   actual live state. Render those positions and terrain in the field view;
   do not step an independent local replica and present it as the live world.
   Match action labels, input/history encoding and map collision data so the
   player, enemies, observable history and probability panels retain their
   simulator meanings. A live-specific record may make optional fields nullable.
3. Keep authoritative action success/rejection and timing as required input
   dependencies for checkpoints that consume them. Sending a cast is not a cast
   acknowledgement. Single-episode inference removes training bookkeeping;
   it does not remove the policy's need for correct inputs. Do not replace
   missing history with guessed successes or fabricated zeros.
4. Display unavailable engine diagnostics, attributed kills/damage and reward
   totals as **Unavailable**, not zero or inferred certainty. Reward computation,
   matching terminal bonuses, kill accounting and the critic are optional
   diagnostics; they must not block inference when policy inputs are valid.
5. Show connection/staleness, run status and the final stop reason. Current
   state / Previous state keeps the same meaning. UI pause freezes inspection
   only; it does not pause the game or inference. Disable simulator-only reset,
   new-seed, step and speed controls in LIVE mode. A separate explicit Stop
   inference action may end the run; it never restarts it.
6. Verify published inputs, probabilities, selected actions and recurrent memory
   against offline replay. Test that death or interruption sends no further
   gameplay commands and never creates another episode, even if new alive-player
   snapshots arrive. Additional comparison runs are separate explicit launches,
   not a required multi-episode evaluation loop.

The reduced implementation phases and optional deeper mechanics comparisons
are described in [the ZMQ parity plan](../../Custom_enviornments/MYSTIC_ZMQ_PARITY_PLAN.md).

No live-server launch or synchronized live-world replica is implemented yet.
