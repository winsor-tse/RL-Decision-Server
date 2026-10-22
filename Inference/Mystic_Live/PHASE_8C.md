# Phase 8C: actions and observable policy inputs

Implemented locally: the seven-action adapter, strict 56/80-value input streams,
saved-payload parity tests, an authoritative telemetry specification, and a capture audit.
The deployment checkpoint remains **unselected**. The CLI remains observation
only. Real game-server feedback wiring and real capture coverage are outstanding
exit evidence; the synthetic tests do not establish live mechanics parity.

`inputs.py` calls the existing simulator encoder, uses the actual simulator Box,
and reuses `ObservableHistory` and the map53 terrain sensors. It constructs an
environment only to load its immutable map/space, without reset, engine, monsters,
training or a policy. No simulator source, engine hash, checkpoint metadata or
legacy 26-input/11-action path changes.

## Run and inspect

The ordinary JS payload can supply the base 56 features:

```powershell
.\RL_venv\Scripts\python.exe -m Inference.Mystic_Live.server --observe-only --input-size 56
.\RL_venv\Scripts\python.exe -m Automation.Bridge.ws_zmq_bridge --single-run
```

Run these in separate terminals. Use `--input-size 80` only with the authoritative
observation/outcome fields below. Missing initial clock telemetry leaves the run
waiting without initializing an actor. Subsequent missing outcomes keep a chosen
action pending until timeout. No automatic conversion of 80-input weights to a
56-input policy is provided. If server outcomes cannot be exposed, a new input
schema and separately trained policy are needed before deployment.

The final JSON snapshot includes `current_base_observation`,
`current_policy_input` and `policy_input_status`. A ready policy input records its
schema, action ordering, raw values, diagnostic normalized values and source
timestamp/context. `last_decision.policy_input` preserves the exact previous
input used for the chosen action. A pending newer frame has no eligible policy
input until its outcome is confirmed. With observation-only 80 mode, only the
initial history reset can be encoded; subsequent frames explicitly report
`awaiting_confirmed_transition` while still exposing their base observation.

`SingleRun(actor, input_adapter=LivePolicyInputs(80))` supplies
`actor.choose(raw_float32_vector, confirmed_history, memory)`. The model receives
raw values and divides by the saved Box high **once**, as in simulator training.
The snapshot's `normalized` array is for inspection, not another preprocessing
step. Existing raw-world actor hooks remain available without an input adapter.
Actual frozen actor/checkpoint loading is Phase 8F. To use the legacy live stack,
continue using its existing environment/model entry point and ordinary bridge
mode; input size never selects a legacy model or remaps action indices.

## Exact feature contract

Indices below are zero based. Values use `numpy.float32`.

| Indices | Values |
| --- | --- |
| 0–5 | Player tile x, y, facing, HP/maxHP, MP/maxMP, numeric map ID |
| 6–45 | Ten ranked monsters, four values each: Manhattan distance, relative direction, HP fraction, MP fraction |
| 46–55 | Signed dx/dy for the first five of those same ranked monsters |
| 56–62 | Previous confirmed chosen action, seven-value one-hot |
| 63–65 | Applied, movement blocked, cooldown rejected |
| 66–68 | Cast ages for slots 1–3, capped at ten seconds and divided by 10000 ms |
| 69–71 | Cast-seen flags for slots 1–3 since this explicit run began |
| 72–79 | Four terrain rays, up/down/left/right, each free-distance fraction and detection flag |

Actions are exactly `up`, `down`, `left`, `right`, `castSpell:1`,
`castSpell:2`, `castSpell:3`. The shared schemas remain
`mystic-seven-v2`, `mystic-ten-enemies-relative-five-56-v2` and
`mystic-ten-enemies-relative-five-local-terrain-80-v4`.

The player requires map53 and integer tile coordinates 0–99, with +x right and
+y down. Only living `type: "monster"` entities with
`isCurrentPlayer: false` are eligible. Every living observed monster needs ID,
coordinates, HP/maxHP and MP/maxMP, even outside the selected ten. Missing fields,
nonfinite numbers, invalid coordinates or duplicate normalized monster IDs are
errors. Dead monsters need no coordinates/resources beyond their finite HP and
valid ID/classification; other entity types do not consume slots. A player and a
monster may legitimately share an ID in their different entity namespaces.

Canonical decimal string IDs normalize to integers; numeric IDs sort numerically.
Opaque nonempty string IDs sort lexically after numeric IDs. Ambiguous strings
such as `"002"` remain opaque; whitespace, booleans and floats are rejected.
Ranking uses `(Manhattan distance, normalized ID)`, independent of payload order.
The redundant wire `distance` field is unused. Direction codes are up=0,
down=1, left=2, right=3, idle=4; vertical wins a diagonal tie. HP/MP fractions
clamp to [0,1], with zero MP capacity yielding zero. Normal observed absence
pads with zeros; no full-map or hidden enemies are added.

Bounds and divide-by-high normalization come from `MysticSimEnv`, including
[-99,99] signed offsets and maximum Manhattan distance 198. Terrain comes from
the confirmed shared map53 layer, loaded once. Rays inspect four tiles and stop
at blocked cells or bounds; adjacent blockage is (0,1), a blocker at tile four
is (0.75,1), and no detection is (1,0). Entity occupancy does not enter these rays.

## Authoritative telemetry specification

The inner world projection retains the shape of
`Simulation/Source-Code/Example_Full_State.txt` and
`Tests/fixtures/simulation/example_full_state.json`. Their illustrative map3471
must not be treated as an actual map53 observation. The live geometry is the
confirmed shared `Tests/fixtures/simulation/map53.json`.

For 80 inputs, add an outer `observationContext` describing the **same
authoritative world projection**, using a server clock distinct from browser
`worldState.timestamp`. Each resulting frame with a completed action also carries
the Phase 8B `execution` record:

```json
{
  "type": "ai_tick",
  "requestId": "next-tick",
  "runId": "run",
  "worldState": {"timestamp": 1789086219890, "player": {}, "entities": []},
  "observationContext": {
    "schema": "mystic_live_observation_v1",
    "clockId": "server-process-clock",
    "stateTimeMs": 1175
  },
  "execution": {
    "schema": "mystic_live_execution_v1",
    "runId": "run",
    "actionId": "run:1",
    "requestId": "causing-tick",
    "actionLabel": "castSpell:1",
    "applied": true,
    "movementBlocked": false,
    "cooldownRejected": false,
    "clockId": "server-process-clock",
    "attemptedAtMs": 1000,
    "stateTimeMs": 1175
  }
}
```

The example abbreviates player fields; a real payload requires the complete
projection. A game-server hook must preserve the selected command's run, action
and causing request IDs. `attemptedAtMs` is the authoritative action validation/
application instant; for an applied spell it is its confirmed cast instant.
`stateTimeMs` is the instant of the resulting world projection. The feedback and
observation clocks/times must match exactly. Clock changes, backward timing or
mixing feedback with a newer unrelated snapshot prevent another decision.

The episode starts with the same explicit history reset as simulator inference.
After that, only a matching completion updates history, once. Actual state
intervals advance the shared history, including unseen cast-age channels (their
seen flags stay zero). A newly confirmed spell's age uses snapshot time minus
its actual application time, accounting for execution delay. Rejected spells
preserve previous successful cast times. No fixed 200 ms interval is assumed.

The producer must report all three outcome booleans explicitly. Sending a
command, local movement prediction, coordinate/MP changes or cooldown guesses
cannot set them. `movementBlocked` corresponds to the simulator's collision
outcome, while `cooldownRejected` covers slot/family/global cooldown rejection.
Other rejection reasons may have all three flags false. Server validation must
establish these semantics before using trained policies on the game.

## Remaining game/extension wiring

This Python repository defines and validates the received telemetry contract.
The producer belongs in the actual game server and browser-plugin repository;
JavaScript forwarding changes belong in that plugin. No producer is installed
by the Python adapter.

1. Add a game-server execution hook for accepted/rejected movement and casts.
   Carry correlation IDs through the game command path, expose authoritative
   application times/reasons, and project the resulting visible world at a known
   time. The supplied C# source excerpts do not include a runnable server project
   or its movement packet implementation; no server build can be validated here.
2. Connect that hook to the browser plugin and send its exact observation and
   feedback together using the contract above. Carry explicit outcomes and server
   times through the actual server/extension path. Serialize one action at a time.
3. In the sibling extension, preserve `observationContext`, `execution` and
   `runId` through `battler.js`'s page message, `content-bridge.js`'s
   `port.postMessage`, and `background.js`'s `sendToPython` envelope. Both latter
   forwarding points currently whitelist fields and drop these additions.
   The Python single-run bridge already forwards these fields unchanged.
4. Record real map53 ticks and confirmed outcomes. Verify the available visible
   entity set and tile axes, replay all seven actions and rejection cases, and
   inspect tensor differences before enabling an actor in Phase 8F.

The sibling extension and a running game server are not changed by this local
implementation. Their producer and forwarding changes remain outstanding
dependencies for live 80-input deployment.

## Capture coverage and validation

```powershell
.\RL_venv\Scripts\python.exe -m Inference.Mystic_Live.inputs --capture Tests/fixtures/live/phase8a_capture.jsonl --output runs/mystic_live/phase8c_input_audit.json
.\RL_venv\Scripts\python.exe -m unittest Tests.test_mystic_live_inputs Tests.test_mystic_live_lifecycle Tests.test_mystic_live_evidence Tests.test_mystic_sim_expanded_observation Tests.test_mystic_sim_contracts
```

Use another output filename for another audit; files are exclusive. Repeated
`--capture` arguments compare multiple captures. The report preserves evidence
origin, counts valid/dead frames, reports missing required fields/maps, maximum
visible monster count and whether ten slots were observed, and inventories
authoritative context/outcome fields. Fewer than ten observed monsters can be
legitimate; it does not prove truncation or justify invented enemies. Clock field
presence alone does not prove authority or correlated history parity.

The available Phase 8A fixture exposes one living monster and no authoritative
feedback/context. It cannot establish real visibility or 80-input readiness.
`phase8c_inputs.json` is a separate synthetic payload with eleven living
monsters, tied distances, reversed order, a numeric string ID, a dead monster,
and an unrelated player, plus hand-calculated 56 features and saved transitions.
Tests also project actual seeded simulator worlds, compare all 80 values with
the training history, exercise sensor boundaries and timing/rejection cases,
and carry synthetic 80-input feedback through real
local WebSocket/ZMQ sockets. No checkpoint is loaded and no game is contacted.
