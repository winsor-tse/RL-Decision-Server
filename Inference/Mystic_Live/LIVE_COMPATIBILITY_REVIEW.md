# Phase 8A-8C live compatibility review

Update 2026-10-23: [Step 3](STEP_3.md) addresses findings 3 and 4 in the local
command adapter and sibling extension and provides an opt-in 71-input training
alternative for finding 2. Existing 80-input telemetry remains unavailable.
Reloading and validating the extension against the running game is still needed.

Update 2026-10-23: Steps 1 and 2 added the frozen loader and snapshot progression.
Finding 1 below is resolved for compatible snapshot-mode policies; strict mode
retains its outcome requirements. See [STEP_2.md](STEP_2.md). The findings below
record the original audit; game/extension validation is still pending.

Reviewed 2026-10-22 against Python and sibling browser-extension source. The
actually loaded extension/server were not inspected. No gameplay was sent.
Preserve the code as a starting point, not as accepted live integration.

## Findings

1. **High: 8B blocks every actor after its first command without new telemetry.**
   `lifecycle.py:SingleRun.handle` requires an execution acknowledgement even for
   56 inputs. Reproduced using the existing fixture and a stub actor: first reply
   `castSpell:1`, next replies `awaiting_execution_acknowledgement`, then
   `acknowledgement_timeout`. Ordinary browser requests supply no such feedback.
   Add snapshot-driven progression for outcome-independent policies. Retain
   strict mode for existing 80-input history. Update bridge validation alongside
   it: commands currently require `action_pending` status and an action ID.

2. **High: 8C's 80-input protocol has no producer in the inspected path.**
   An ordinary first frame produces `80-value input requires authoritative
   observationContext` (directly reproduced). Later history needs outcomes too.
   `battler.js` supplies neither. `content-bridge.js` and `background.js` whitelist
   fields, so adding them only in the page would still lose them. Implement the
   complete producer/forwarding path or train a separately versioned observable
   policy. This dependency must not block 56-input inference. A Python validator
   does not establish server support.

3. **High: seven labels do not establish simulator spell execution.**
   Python sends bare spells; browser targeting uses Manhattan distance, possible
   non-monster fallback, no simulator rectangle and no self fallback. Explicit
   self IDs also fail: `getEntityById` uses `getValidSpellTargets`, which excludes
   the current player. Ran the actual extracted functions under Node with a stub
   registry: `castSpell:2:42` for player 42 gave null target lookup and zero cast
   calls. Implement explicit targeting and a browser self-target path. Test the
   real JavaScript parser/executor, no-target cases and facing, not just indices.

4. **Medium: browser stop is not integrated.**
   The extension never reads `stopInference`; the bridge suppresses commands but
   the page continues requesting. Also `stopAiLoop()` does not invalidate a
   pending response: `runAiLoop` applies it after awaiting without checking the
   running flag. A command in flight can execute after browser-side manual stop.
   Distinguish Python Ctrl+C from browser controls. Check a run generation and
   running flag after awaiting, handle terminal status, and test delayed replies
   across stop/restart. Python cannot retract an already transmitted command.

5. **Medium: 8A's feedback inventory is stale.**
   `baseline.py:field_matrix` always reports names like `execution.action_applied`
   and `execution.game_server_time` unavailable. The implemented proposed protocol
   uses `applied`, `movementBlocked`, `cooldownRejected`, `attemptedAtMs`,
   `stateTimeMs` and `observationContext`. Supplying these does not update the
   hardcoded rows. Inventory actual fields; distinguish absent, present,
   structurally valid and verified provenance. Add a protocol capture regression.
   Presence alone cannot prove authority.

6. **Reporting limitation: checkpoint inventory is metadata-only.**
   `checkpoint_identity.local_compatibility=passed` does not instantiate an actor
   or validate state-dict shapes. Label this scope explicitly and require actual
   loading/replay for deployment. It does not prove that a checkpoint can run.

## Constraints and useful work

- A new run requires restarting BOTH bridge and backend. This is intentional and
  already documented in PHASE_8B; retain it in launch instructions.
- Strict integer coordinates, resources and explicit monster booleans match the
  example but need real-capture verification. Undefined JS properties disappear
  in JSON. Do not invent missing fields or assume every entity supplies them.
- Queue-age comparisons assume comparable bridge/backend monotonic clocks on the
  same host. Default localhost deployment fits; remote hosts need review.
- Shared map loading, pure encoding and terrain sensing are useful locally.
  Optional capture correctly separates sent commands from confirmed execution.

## Validation and status

Ran `Tests.test_mystic_live_inputs`, `Tests.test_mystic_live_lifecycle` and
`Tests.test_mystic_live_evidence` using `RL_venv/Scripts/python.exe -m unittest`:
**47 tests passed**. Additional one-off reproductions confirmed the 56-input
stall, first-frame 80-input rejection and JavaScript self-target failure.
No selected trained checkpoint or real game episode was run.

8A is usable inventory/capture tooling needing reporting corrections. 8B is a
local strict-protocol controller needing ordinary-snapshot support. 8C is local
encoding with unmet telemetry and execution dependencies. None certifies current
live-server support. Address these findings within the four-step deployment plan.
