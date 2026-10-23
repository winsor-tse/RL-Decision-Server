# Step 2: decisions from ordinary live snapshots

Implemented 2026-10-23. Compatible 56-input checkpoints now use snapshot mode
automatically. Each valid fresh request can produce one action, without an
execution record or authoritative server clock. Browser cadence is unchanged.

```powershell
.\RL_venv\Scripts\python.exe -m Inference.Mystic_Live.server --checkpoint PATH_TO_56_INPUT_CHECKPOINT
.\RL_venv\Scripts\python.exe -m Automation.Bridge.ws_zmq_bridge --single-run
```

Run in separate terminals. Use `--sampled` to sample instead of taking the greedy
action. Restart both processes for a new run. Observation-only commands remain
supported.

## Progression and evidence

`SingleRun(..., progression='snapshot', input_adapter=LivePolicyInputs(56))`
has no pending acknowledgement and never calls the confirmed-history updater.
Replies identify `progression=snapshot` and stay `running`. The bridge accepts
commands in this state only when explicitly marked as snapshot progression.
Each command still has a unique action ID.

The final snapshot includes progression and up to 128 recent `issued_commands`.
`dispatch_status=reply_issued` means the controller returned a command for
transport; it does not prove browser delivery or game acceptance.
`execution_outcome=null` preserves that distinction. Confirmed history stays
untouched even if unsolicited feedback appears in snapshot requests. Observed
client/receipt intervals remain available without claiming exact 200 ms server
execution. Duplicates and viewer reads do not advance model memory.

Strict behavior remains the default for direct `SingleRun` callers.
`--confirmed-feedback` explicitly selects it in the CLI, including for 56 inputs.
Existing 80-input policies are rejected before socket creation unless that option
is supplied; their context/outcome producer is still required. Snapshot mode
rejects adapters that require outcomes or do not declare that requirement.
It never pads or fabricates history.

## Validation and remaining scope

All 61 tests passed on 2026-10-23:

```powershell
.\RL_venv\Scripts\python.exe -m unittest Tests.test_mystic_live_snapshot Tests.test_mystic_live_policy Tests.test_mystic_live_lifecycle Tests.test_mystic_live_inputs Tests.test_mystic_live_evidence
```

Coverage includes repeated no-feedback decisions through actual local WebSocket
and ZMQ sockets, duplicate suppression, disconnect/death terminal latching,
manual stop, idle timeout, unsupported maps, unchanged strict-mode tests, unknown
outcomes, CLI selection and rejection of an 80-input snapshot launch.

No real game run was performed. [Step 3](STEP_3.md) now implements explicit
targeting, self-casts, browser stop/in-flight-response handling and a separately
versioned 71-input recurrent option. Existing 80-input policies still need real
outcome telemetry. Reload the updated extension before deployment.
