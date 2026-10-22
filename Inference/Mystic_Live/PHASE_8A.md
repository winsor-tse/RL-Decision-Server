# Phase 8A: freeze the baseline and establish evidence

Phase 8A implements an offline baseline inventory and optional bridge capture.
It does not run a policy, train, change action selection, start/reset a game, or
implement the single-episode inference lifecycle from Phase 8B. The deployment
checkpoint is **unselected**, as requested. Existing live backends retain their
own behavior; capture does not make a legacy trainer inference-only.

## Freeze a baseline

From the repository root, choose a new output directory:

```powershell
.\RL_venv\Scripts\python.exe -m Inference.Mystic_Live.baseline --output-dir runs/mystic_live/baseline_01
```

This writes `baseline.json` and `report.md`, recording:

- Git revision and dirty working-tree status, plus source SHA-256s.
- Explicit checkpoint selection status. Optional `--checkpoint PATH` records its
  hash, saved architecture, action/observation schema, normalization scales,
  scenario/task settings and local compatibility result. No latest checkpoint is
  chosen. Nothing is restored into an optimizer or inference loop.
- Current seven-action, 56/80-input simulator reference, model defaults/scales
  and task settings; legacy eleven-action/26-input live config. Reference defaults
  do not select a deployment model.
- Shared map53 hashes, dimensions and blocked-cell count. Live obstacle equality
  is user-confirmed, not rediscovered from snapshots.
- Full-state fixture and original source-file hash, when present. The fixture is
  nested `worldState`; the plugin adds `ai_tick`, `requestId`, `pageUrl` and an
  outer timestamp. Its map3471 values are examples, not the deployment map.
- Python/package versions, capture inventory, field availability and named
  missing producers for execution feedback, authoritative timing and identity.

Bundles and captures never overwrite an existing path. Optionally inventory
local plugin files:

```powershell
.\RL_venv\Scripts\python.exe -m Inference.Mystic_Live.baseline --extension-dir ../yugen-battler/yugen-battler --output-dir runs/mystic_live/baseline_02
```

The plugin manifest/version and five source hashes are **local source only**,
not proof of which copy Chrome loaded. Optional `--deployment-info PATH` retains
operator-reported identity JSON as unverified. The current plugin does not supply
validated loaded-extension/game-build identity. Those remain explicit unknowns;
this phase changes no sibling extension or game-server files.

## Capture an existing boundary

Start the bridge with capture instead of starting a second bridge:

```powershell
.\RL_venv\Scripts\python.exe -m Automation.Bridge.ws_zmq_bridge --capture runs/mystic_live/captures/movement_01.jsonl --capture-label movement
```

It still needs your existing backend on port 5555 to answer requests. Phase 8A
does not supply live inference or send scripted commands to a game. The human
recorder is unchanged. For an existing launcher that starts the bridge:

```powershell
$env:MYSTIC_ZMQ_CAPTURE = 'runs/mystic_live/captures/session_01.jsonl'
# Start your existing stack normally; use a fresh filename each launch.
# After that stack has stopped:
Remove-Item Env:MYSTIC_ZMQ_CAPTURE
```

Capture is off by default. Records are copied into a bounded queue; a worker
writes off the async bridge loop. Defaults: 10,000 evidence records, 64 MiB,
256 queued records. `--capture-max-records` / `--capture-max-bytes` change disk
limits. The small final marker is outside that budget to explain truncation.
Limits discard evidence only, never gameplay exchanges. Losses and writer errors
are logged; captures with loss or no final marker are incomplete. Graceful Ctrl+C
flushes queued records. Forced termination may leave an incomplete file; truncated
JSON lines are rejected explicitly by analysis.

`--capture-actions legacy-live` is the default annotation order. Use `sim-seven`
only with a backend actually using the seven-action contract. This annotates
indices; it does not remap commands or infer a contract by size. Target IDs are
recorded only when explicit in commands; implicit auto-targeting stays unknown.

## Versioned capture format

Every `mystic_zmq_capture_v1` JSON line contains `schema`, `session_id`,
`episode_id`, `sequence`, `event`, `wall_time_unix_ns`, `monotonic_ns`,
`evidence_origin` and an optional operator `label`. Sessions identify bridge
recordings; `episode_id` stays null because no authoritative episode ID exists
in the current protocol. Connections get `connection_id`; `request_id` preserves
the original `requestId`.

| Event | Evidence |
|---|---|
| `capture_start` / `capture_end` | Action ordering, limits, final counts/losses/errors |
| `client_connected` / `client_disconnected` | Transport boundaries |
| `client_request` | Untouched parsed request/full state and bridge receive clocks |
| `backend_sent` | Exact request forwarded to ZMQ and local send-completion clock |
| `backend_reply` | Raw backend reply before optional request-ID filling |
| `client_response_sent` | Actual plugin reply and command annotations |
| `backend_timeout` / `backend_error` | Correlated transport failure |
| `malformed_message` | Invalid JSON/non-object input |

Bridge times are local observations. Request timestamps are client clocks;
reply `serverTime` is Python wall time, not authoritative game execution time.
Execution confirmation, application time, implicit target and rejection reason
are not inferred from sent commands or changing coordinates. Extra payload fields
are retained raw; their meaning requires a validated acknowledgement contract.

Supply repeated `--capture PATH` arguments to analyze several files. Reports
include unmatched replies, duplicate correlation keys, ID mismatches, sequence
gaps, completion/loss status and local bridge roundtrip timing. Optional
`clientInfo` / `serverInfo` fields are sender-reported and unverified.

## Scripted fixtures and field matrix

```powershell
.\RL_venv\Scripts\python.exe -m Inference.Mystic_Live.baseline --capture Tests/fixtures/live/phase8a_capture.jsonl --output-dir runs/mystic_live/scripted_baseline_01
```

The checked-in fixture is **synthetic**, covering movement, intended blocked
movement, spells 1/2/3, no-target/cooldown attempts, death, map transition and
disconnect. Labels describe intended cases, not confirmed mechanics. Real evidence
requires explicitly exercising those cases with an appropriate existing backend
or manual controls and recording the boundary. Additional runs are explicit
starts; no reset loop is required.

The field matrix distinguishes observed, derived and unavailable values, with
reference-only versus supplied-capture provenance:

| Input / diagnostic | Producer / gap |
|---|---|
| Player/enemy coordinates, HP/MP, type/identity and facing | JS full-state reference; actual captures check presence/types |
| Relative offsets and distance fallback | Derived from coordinates; exact encoding is Phase 8C |
| Terrain rays | Shared map53 geometry and existing sensors |
| Previous chosen action | Correlated outbound label under explicit action ordering |
| Applied / blocked / cooldown rejection, confirmed cast ages | Missing validated acknowledgement producer |
| Authoritative execution/application time | Missing in current payload |
| Loaded extension and game build | Not established by local files or current payload |
| Rewards and attributed damage/kills | Optional diagnostics, not actor inputs |

## Validation and status

```powershell
.\RL_venv\Scripts\python.exe -m unittest Tests.test_mystic_live_evidence
```

Tests cover copy-on-enqueue, loss/limits, disk failure, unselected checkpoint,
provenance, malformed/truncated data, timeouts and capture-disabled behavior.
Actual local WebSocket/ZMQ sockets with a stub backend check unchanged messages
for the scripted cases; they do not connect to or control the real game.

Implemented: baseline generator, optional bounded capture, versioned synthetic
fixtures, field matrix/report, source inventory and offline/transport checks.
Outstanding external evidence: loaded extension identity, connected game build
and real scripted-game captures. Missing producers are explicit; no live
compatibility or real-game parity is claimed. The checkpoint remains unselected.

See [the phase plan](../../Custom_enviornments/MYSTIC_ZMQ_PARITY_PLAN.md) and
[live watch TODO](../../Training/Mystic_Sim/WATCH_POLICY.md#todo-launch-the-same-watch-ui-from-the-live-server)
for subsequent single-episode inference work.

Follow-up implemented: [Phase 8B lifecycle and observation-only server](PHASE_8B.md).
It adds single-run startup/stop and feedback correlation; it does not select a
deployment checkpoint or claim that the current plugin publishes confirmed outcomes.
