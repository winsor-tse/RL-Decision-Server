# Step 3: live command semantics and optional observable recurrence

Implemented locally 2026-10-23. No deployment checkpoint is selected. Supply its
path yourself in the CLI. No real game episode or production training run was
started; short training/resume tests used temporary checkpoints only.

## Run a compatible policy

Reload the updated Yuugen Battler extension and refresh the game page so both
the service worker and injected page script use the new code. Start these in
separate terminals, replacing the placeholder with a compatible checkpoint:

```powershell
.\RL_venv\Scripts\python.exe -m Inference.Mystic_Live.server --checkpoint "PATH_TO_CHECKPOINT"
.\RL_venv\Scripts\python.exe -m Automation.Bridge.ws_zmq_bridge --single-run
```

56-input feedforward and newly trained 71-input recurrent policies use snapshot
progression. Greedy is the default; `--sampled` enables sampling. Each new run
requires restarting both Python processes and explicitly restarting the browser
AI loop. No checkpoint is selected automatically.

The extension advertises `mystic_explicit_target_v1` on each tick. The CLI's
command adapter waits with `extension_update_required` if this capability is
missing and sends no gameplay commands. Losing it during a run interrupts the
run. This verifies protocol support as advertised by the client, not server build
identity or successful command execution.

## Commands and browser changes

`commands.py:LiveCommands` preserves the seven policy labels separately from
the wire command. Movement uses the same four existing direction commands.
Spells select living visible monsters inside inclusive 16-X/10-Y bounds, using
Euclidean distance then normalized ID. Observation ranking remains Manhattan.
Wire target IDs retain their original type/value and have a separate target kind
so a player and monster sharing an ID cannot be confused.

No eligible target means an inert Arcane Blast reply, or self-targeted slot 2/3.
The chosen action still enters the new history even when there is no executable
command. No success or rejection is inferred. Delimiter-containing target IDs
are rejected instead of being serialized ambiguously.

The actual sibling extension files were updated:

- `../yugen-battler/yugen-battler/battler.js`: explicit monster/self lookup,
  current-target validity/range checks, no fallback to another target, no automatic
  facing change for this protocol, terminal stop handling and generation checks
  that discard a pending reply after browser stop/restart. Legacy commands retain
  their existing execution path.
- `../yugen-battler/yugen-battler/content-bridge.js`: forwards capabilities.
- `../yugen-battler/yugen-battler/background.js`: forwards capabilities to Python.

Paths above are relative to the repository root. The browser still uses its
existing movement implementation and the game decides whether actions succeed.
The ordinary payload lacks immunity flags; targeting uses observable eligibility
and does not claim to reconstruct hidden server state. A disappeared/dead/out-of-
range explicit target is skipped, not replaced.

The reviewed patch, hashes and original-file backups are under
`runs/mystic_live/step3_extension/`. `Automation/Bridge/prepare_mystic_extension.py`
can prepare the same changes from an unmodified matching source into a separate
directory; it refuses unexpected source text and never modifies its input files.

## Optional training contract: 71 inputs

Use this only for a NEW policy when recurrence is desired without server outcome
telemetry. The schema is
`mystic-ten-enemies-relative-five-chosen-action-terrain-71-v1`:

| Indices | Inputs |
| --- | --- |
| 0-55 | Existing shared base observation |
| 56-62 | Previous chosen action, seven-value one-hot; initially zero |
| 63-70 | Shared static terrain rays, up/down/left/right, two values each |

Training and live inference share `LiveObservableHistory`. It records the choice
regardless of success, and contains no cooldown rejection, applied flags or
successful-cast ages. It uses the existing flat recurrent architecture; the
80-input attention layout is not resized. Training defaults and old checkpoint
contracts remain unchanged.

Example opt-in training command (adjust budget/settings as desired):

```powershell
.\RL_venv\Scripts\python.exe -m Training.Mystic_Sim.train --recurrent --recurrent-encoder flat --history-mode live --device cpu
```

Add `--validate-only` to check configuration without training. Loading, resume,
evaluation and watch support the saved schema. Viewer feature labels distinguish
the 71-input history from the old 80-input history.

Existing 80-input weights are NOT converted. They still require
`--confirmed-feedback` plus a real authoritative context/outcome producer and
forwarding, which are not provided by this extension update. No runnable server
implementation is available here to install that producer. The supported current-
payload alternatives are an existing compatible 56-input policy or newly trained
71-input weights. No fallback model or zero-filled history is substituted.

## Validation

92 tests passed across live commands/history/policy/snapshot/lifecycle/inputs/
evidence and simulator recurrent/attention/watch suites. They include actual
local WebSocket/ZMQ forwarding of explicit target commands and capability fields,
71-input tensor and actor-state replay, short real training plus resume, initial
and episode-reset history, no-target actions, ties, boundaries and namespaces.

The actual installed JavaScript functions also pass:

```powershell
node Tests/mystic_extension_commands.cjs ../yugen-battler/yugen-battler
```

Those checks cover all seven commands, self targeting, stale/range-invalid
targets, facing, terminal stop and delayed responses across stop/restart.
They use controlled browser/network stubs, not a connected game server.

Additional curriculum/comparison suites could not complete: this checkout lacks
their `Training/Mystic_Sim/configs/curriculum_manual.json` and
`compare_4v1_5v1.json` configuration fixtures (19 errors, including subtests).
Focused results are in `runs/mystic_live/step3_tests.log`; the additional failures
are in `step3_extra_tests.log` beside it. Real-game validation remains Step 4.
