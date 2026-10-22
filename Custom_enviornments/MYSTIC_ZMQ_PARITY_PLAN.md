# Mystic ZMQ and simulation parity plan

Prepared 2026-10-21 from the local implementations. Phase 8A evidence tools and
Phase 8B lifecycle are implemented; Phases 8C-8H remain planned. No simulator
policy/checkpoint contracts changed. Phase 8B's CLI is observation-only until
the live encoder, feedback producer and actor loader are integrated.

## Scope and recommendation

Compare the live `Test_Env/Mystic.py` WebSocket/ZMQ path with the working
Python replica in `Mystic_Sim/` and its trainers in `Training/Mystic_Sim/`.
The separate `Simulation/YugenSimEnv.hpp` is an older C++ scaffold whose step
does not implement combat. It is not the current Python training replica.

Implement this as Phases 8A-8H of the existing `PYTHON_INTEGRATION_PLAN.md`.
Prioritize correct action/state alignment, observable inputs, and execution
semantics before further training experiments or performance optimization.
Use a separate seven-action live adapter so existing live models retain their
original contracts. Preserve the current simulator as a versioned baseline;
change mechanics only when connected-game evidence supports the change.

### Confirmed map and ZMQ payload context

The user confirms that the live map is
[Tests/fixtures/simulation/map53.json](../Tests/fixtures/simulation/map53.json)
and is the same as the simulator map, including static obstacles. The fixture
and [packaged simulator map](Mystic_Sim/data/map53.json) have matching SHA-256
hashes in this checkout. Treat obstacle-layout parity as established context:
reuse the existing map loader/collision layer for live rendering and terrain
rays. No separate obstacle discovery, map reconstruction or alternative live
map asset is needed for map53.

The full state passed through ZMQ has a structure similar to
[Tests/fixtures/simulation/example_full_state.json](../Tests/fixtures/simulation/example_full_state.json).
Use its `timestamp`, `player` and `entities` structure as the raw `worldState`
parser reference, including IDs/types, map coordinates, direction, HP/MP and
maximums, and supplied distances. The fixture's `map3471`, player stats and
entity positions are example values, not the configured map53 run's state.
Live player/enemy positions and resources are read from each current payload;
sharing the map does not require stepping an independent local world.

The payload fixture is a byte-identical copy (matching SHA-256) of
[Simulation/Source-Code/Example_Full_State.txt](../Simulation/Source-Code/Example_Full_State.txt).
The inspected JavaScript `buildWorldState()` produces this structure, and the
extension wraps it as `worldState` in an `ai_tick` message with `requestId`,
`pageUrl` and an outer `timestamp`. The bridge forwards that message unchanged
to ZMQ. The example file represents the nested world state, not the entire
request envelope.

### Deployment scope: single-episode live inference

Train and parallelize in the simulator. The new ZMQ path serves a frozen actor
against one real player, normally for one episode per explicit launch. It does
not train, collect PPO rollouts, restore optimizers, bootstrap values, or cycle
episodes. The existing live trainers remain separate legacy entry points.

Start from the first valid living-player state in the configured map, without
resetting or teleporting the game. Initialize actor memory/history once. Stop
dispatching gameplay commands on confirmed death or manual stop, preserve the
final UI state, and require an explicit new launch to run again. Disconnect,
timeout, or unsupported map changes stop the run as interrupted; automatic
reconnect/resume and respawn/reset loops are outside this workflow.

The simulator's kill goal, Y failures, step cap, rewards and critic are not live
inference requirements. Optional evaluation goals/limits and outcome metrics
must be explicitly enabled; missing optional diagnostics display as unavailable.
The action/observation contract, actor architecture, recurrent history, outcome
acknowledgements required by that history, and measured action timing remain
necessary for existing checkpoint compatibility.

Findings below are established from local code, including the sibling
`../yugen-battler/yugen-battler/battler.js`. The browser's actually loaded
extension and connected server version have not been verified. Distinguish
these code differences from mechanics discrepancies that still need live data.

## Major differences and their impact

| Priority | Area | Current ZMQ/browser path | Python replica | Required response |
|---|---|---|---|---|
| P0 | Action contract | Mystic defaults to 11 actions: four moves, attack, slots 1/2/3/5/6/7. | Seven: four moves and slots 1/2/3. ID 4 means Arcane Blast rather than attack. | Explicit seven-action live adapter; validate labels and schema before sending commands. |
| P0 | Observation contract | 26 values, five monsters; distance-only stable sorting preserves payload order on ties. No explicit dead-monster filter in observation parsing. | 56 base values: ten living enemies and signed dx/dy for the first five; distance then entity-ID sorting. Recurrent policies consume 80 values. | Shared, versioned encoding for the new adapter, including filtering, ID handling, padding, signed bounds, and tie rules. |
| P0 | Recurrent inputs | No confirmed action outcomes, collision reasons, cooldown rejections, or cast acknowledgements in the inspected world-state builder. | Adds 24 history/sensor values, including confirmed outcomes, successful-cast ages, and four local terrain rays. | Add required outcome telemetry; reuse the confirmed shared map53 collision layer. Missing required telemetry blocks deployment of the existing 80-input policy. |
| P0 | Transition alignment | `step(a)` receives a world snapshot, sends action a, then returns that same received snapshot and its reward. | Applies a, advances 200 ms, then returns the resulting observation/reward. | Correlate actions/outcomes for observable history and the Previous state display. Live inference needs no reward-bearing training transitions. |
| P0 | Decision timing | Browser awaits the response, applies the action, then sleeps 200 ms. Snapshot timestamps use client `Date.now()`. Bridge `serverTime` is Python wall time. | Exact 200 ms simulated decisions and integer event deadlines. | Measure command application and authoritative state times. The browser interval includes response/processing delay; neither existing timestamp is proof of game-server execution time. |
| P0 | Targeting | Browser ranks by Manhattan distance, favors monsters but can fall back to other entities, and has no 16-X/10-Y eligibility filter in target selection. With no target, all three spell commands are skipped. | Ranks eligible monsters by Euclidean distance then ID inside an inclusive 16-X/10-Y rectangle. Slots 2/3 fall back to self. | Select and validate one execution specification against connected behavior; test ties, rectangle edges, unavailable targets, and self fallback. |
| P0 | Facing and movement | Browser faces the selected target before casting; replica cast does not update player facing. Browser also mutates local coordinates before authoritative movement confirmation. | Movement checks terrain, occupancy, and bounds synchronously. | Validate cast-facing behavior and distinguish client prediction from accepted movement. Do not count local coordinate changes as movement acknowledgements. |
| P0 | Reset | Mystic returns the pre-reset snapshot immediately after sending `reset=True`. Browser implements reset by self-casting slot 4. No reset-completion handshake is checked. | Rebuilds a seeded world and returns its new initial observation. | Bypass the legacy reset mechanism in the new inference entry point. Start on a valid living-player snapshot; stop on death without another episode. |
| P2 | Damage and kills | HP-delta rewards; nearby disappeared entities can be treated as kills. Entity reward tracking includes non-current-player entities without the observation's monster-type filter. Healing can yield positive damage-taken reward. | Attributed damage/death events, damage clamped to actual HP removed, no healing reward in the default combat profile. | Optional reporting: use attributed events when available, otherwise show unknown. Reward and kill attribution do not gate actor inference. |
| P2 | Reward scale and shaping | +10 per inferred kill, -100 on death/map departure, no explicit win bonus; proximity shaping, flat -10 blocked-move penalty, low-HP shaping, incoming HP delta weighted 20. | +10 per confirmed kill, +200 win, -200 death, -25 timeout; default incoming damage weight 50, time cost, escalating blocked-move penalties and cooldown penalty; no proximity reward. | Keep rewards in simulator training. Live rewards are optional diagnostics; compare raw returns only under an explicit equivalent evaluation profile. |
| P1 | Episode rules | Default 256 steps; truncates at Y < 25; penalizes Y < 31; leaving map53 is a loss. | Default 1024 steps; Y outside inclusive 30..86 terminates as failure; penalties below 31 and above 85. Five kills is the default goal on both paths. | Live inference runs until death/manual stop, or interruption. Do not inherit simulator task caps/goals by default. No live PPO bootstrapping is required. |
| P1 | Model loading | Live LSTM uses a shared embedding/LSTM and Tanh actor head; current evaluator loads that architecture. | Separate actor/critic branches; configurable flat or attention encoder; default linear actor; saved preprocessing scales and architecture metadata. | Load the exact checkpoint architecture and preprocessing. Resizing the live network does not make simulator weights compatible. |
| P1 | World and mechanics coverage | Visible client registry, actual player equipment/stats, other actors, and hidden server timers. | Fixed level-135 Mystic baseline, gear off, map53, Innie-only population (80 in full scenario), selected status modifiers off. | Match observable task setup; measure remaining mechanics differences. Do not equate client registry visibility with complete server state. |
| P2 | Throughput and resume | One external game stream; socket recovery does not restore world state. | Many independent seeded worlds; training resume starts fresh episodes. | Parallelize training locally. Live death/interruption ends inference; another run requires an explicit new launch, not training resume. |

The README's historical 15-action list and several integration-plan references
to 26/50-input policies are stale. Current code and the selected checkpoint's
metadata must determine the contract. In particular, simulator Y exits currently
terminate; older prose describing them as truncations is outdated.

## Phased implementation

### Phase 8A - Freeze the baseline and establish evidence

Implementation and commands: [PHASE_8A.md](../Inference/Mystic_Live/PHASE_8A.md).
The offline baseline generator, opt-in bridge JSONL capture, synthetic scripted
fixtures, field matrix and transport tests are implemented. The deployment
checkpoint remains unselected. Actual loaded extension/game-build identities
and real scripted-game captures remain unverified external evidence; the tools
record these gaps explicitly and do not claim live compatibility.

Address: uncertain deployed versions, stale documentation, missing captures.

1. Record repository revision and selected checkpoint metadata, architecture,
   feature schema, action ordering, preprocessing, map hash, and task settings.
   Use the confirmed map53 fixture and example full-state fixture above as the
   geometry and payload references; obstacle-layout parity is already established.
2. Identify the actually loaded browser extension and connected game build.
3. Add optional JSONL capture at the live boundary: session/episode/request IDs,
   raw state, command, selected target, receipt/send/application times and
   authoritative timestamps or outcomes where available. Separate unavailable
   fields from observations. Keep the existing human recorder independent.
4. Capture scripted moves, blocked moves, each spell, no-target casts, cooldown
   rejection, player death, map transitions, and disconnect. Capture kills/damage
   attribution for optional metrics when available; reset-loop testing is not
   required for the single-episode inference entry point.
5. Produce a field-availability matrix and correct obsolete contract references.

Deliverables: baseline manifest, versioned capture format, capture fixtures,
and a difference report with observed/inferred/unavailable evidence labels.

Exit: every required input and outcome has a known producer or an explicit
missing-data dependency. Use the example fixture for payload-structure tests,
with map53-specific values in deployment fixtures.

### Phase 8B - Correct action history and single-run lifecycle

Implementation and commands: [PHASE_8B.md](../Inference/Mystic_Live/PHASE_8B.md).
`SingleRun` provides readiness, pending-action correlation, exactly-once confirmed
history, measured timing and irreversible stop states. The observation-only REP
server preserves a final state/timing report. The opt-in `--single-run` bridge
pins the connection and backend run ID, forwards disconnect events and latches
terminal replies. Tests cover death/reconnect/restart without another episode.
The proposed execution-feedback consumer is implemented; the current plugin does
not emit that feedback. Live actor loading/feature encoding and Pygame telemetry
remain subsequent work; the deployment checkpoint stays unselected.

Address: action/history alignment, variable time, and reliable stopping.

1. Introduce a separate inference adapter with waiting-for-first-valid-state,
   running/action-pending, and stopped/interrupted states. Preserve ZMQ REQ/REP
   ordering, using an explicitly inert transport reply where necessary, outside
   policy action IDs. Do not call the legacy reset or send its reset spell.
2. Correlate each action with its input and confirmed outcome for history and
   Current state / Previous state inspection. There is no PPO transition buffer
   or live reward requirement. Request IDs identify exchanges; they alone do not
   establish execution acknowledgement.
3. Measure elapsed time and jitter. Do not replace missing server timing with a
   claimed fixed 200 ms interval. Decide from measurements whether live scheduling
   can match training or a new timing-tolerant training contract is required.
4. Initialize actor memory/history once on a valid living-player state. On
   confirmed death, stop selecting/sending actions before processing another
   policy decision. Preserve the final frame and stop reason for inspection.
5. Reject duplicate/stale/cross-run results. Manual stop ends inference; timeout,
   disconnect, or unsupported map change interrupts it. Do not replay pending
   casts, automatically reconnect/resume, reset, respawn or start another episode.
   A new launch explicitly creates fresh actor memory/history.
6. Keep slow per-tick diagnostic file writes optional in the new adapter and
   measure their contribution to latency.

Deliverables: single-run inference lifecycle, timing report, action/history and
death/interruption fixtures.

Exit: action outcomes attach exactly once; the first state needs no game reset;
death/interruption emits no further gameplay commands. A later alive-player tick
or reconnection cannot restart the stopped run. Pending replies remain inert and
REQ/REP-valid.

### Phase 8C - Match actions and observable policy inputs

Address: 11-versus-7 outputs, 26-versus-56/80 inputs, unavailable history.

1. Use the shared seven-action labels and a pure encoder for the new adapter.
   Keep the legacy live encoder and models explicitly selectable.
2. Match monster filtering, typed ID normalization, Manhattan observation
   distance, direction, ID tie-breaking, ten slots, five aligned dx/dy pairs,
   HP/MP fractions, map ID parsing, zero padding, dtype and signed Box bounds.
3. Verify that real captures expose enough enemies and their coordinates.
   Normal absence uses padding; missing required telemetry is an error rather
   than invented unseen enemies or guessed feature values.
4. Extend telemetry upstream for authoritative application/rejection outcomes
   and cast times, then reuse `ObservableHistory`. A client command-send event
   does not mean a successful cast. This dependency may require game-server work.
5. Reuse the confirmed map53 collision layer and existing map loader to compute
   the same four four-tile terrain rays. Exclude entity occupancy. Check map ID
   and coordinate conventions at startup; unsupported maps stop as described
   in 8B. A separate live obstacle-layout investigation is not required.
6. If authoritative outcomes cannot be provided, use a separately versioned
   observation design and retrain. Existing 80-input weights cannot be deployed
   by substituting zero history or guessed success flags.

Deliverables: shared encoding, telemetry specification, seven-action adapter,
and saved-payload feature parity tests.

Exit: equivalent observed states/actions produce matching 56/80-value tensors,
including history and sensor edges. No required field is silently fabricated.

### Phase 8D - Align execution and calibrate mechanics

Address: targeting/facing, client prediction, and unmeasured server mechanics.

1. Reconcile target ranking, eligibility, ties, self fallback, auto-facing, and
   dead/non-monster handling across browser execution and replica. Explicit
   target IDs are already supported by the inspected browser command parser.
2. Compare authoritative accepted/blocked movement with the shared map53 terrain,
   occupancy, map bounds, and client correction behavior. Check coordinate and
   collision handling; the obstacle geometry itself is already confirmed equal.
3. Validate spell costs and ordering, 300 ms global cooldown, slot/family
   cooldowns, targeting rectangle, AoE falloff, Acid refresh/tick endpoints,
   regeneration, Innie aggro/movement/attack timing, and 50-second respawn.
4. Match player stats/gear, NPC template/scaling, enabled modifiers and initial
   encounter setup before attributing differences to formulas.
5. Use exact comparisons for deterministic cases; define timing tolerances
   from measured cadence/uncertainty. Compare stochastic branches under
   controlled rolls or repeated-run distributions, not matching client seeds.
6. Report the first divergent transition and its evidence. Turn confirmed bugs
   into regressions; version mechanics changes and preserve baseline traces.

Deliverables: mechanics comparison report, confirmed fixes, regression fixtures.

Exit: required deterministic cases have no unexplained divergence; measured
timing/randomness limits and unresolved hidden state are explicit. Missing
server timers are not reconstructed as certain state during trace replay.

### Phase 8E - Optional evaluation diagnostics

Address: understandable live results without requiring training machinery.

1. Default live stopping is death/manual stop, with interruptions handled in 8B.
   Do not inherit the simulator's five-kill goal, Y failures or step cap. Allow
   an explicitly configured evaluation goal/duration to stop this one run only.
2. Report survival time, chosen actions, confirmed cast outcomes, latency and stop
   reason. Add attributed kills/damage when supported; distinguish disappearance
   and other players' kills from confirmed attributed events.
3. Rewards, return, terminal bonuses and matching a simulator evaluation profile
   are optional. Show unavailable quantities as unavailable, never fabricated
   zeros. No critic targets, GAE or bootstrapping are needed in the live path.

Deliverables: optional metrics/profile and explicit availability labels.

Exit: available metrics are correctly labeled and unavailable diagnostics do
not block inference. This phase is not a prerequisite for 8F or the first live
run; compare returns only under equivalent explicit task/reward profiles.

### Phase 8F - Load the exact frozen simulator policy live

Address: incompatible architecture, preprocessing, checkpoint/session handling.

1. Add a dedicated inference entry point that reconstructs the selected
   simulator checkpoint's exact flat/attention encoder, actor head, recurrent
   branches, and normalization. Reuse its model code and validated loading
   behavior rather than duplicating the legacy live LSTM.
2. Validate action labels, observation schema, feature order, architecture,
   preprocessing, and map/task requirements before transmitting policy commands.
   Record simulator engine provenance; do not claim the live server has its hash.
3. Run the frozen actor in eval/inference mode, with actor memory/history created
   once per explicit run. The critic and critic memory need not run. Do not
   restore optimizers, calculate returns, bootstrap values, or advance actor
   memory for duplicate observations or viewer redraws.
4. Replay fixed observation sequences through offline and live inference;
   compare normalized inputs, logits, greedy actions, and hidden/cell states.

Deliverables: live evaluation CLI, strict compatibility checks, inference replay.

Exit: replay matches within defined numerical tolerance; incompatible models
fail before any game action. Evaluation changes no weights or optimizer state.

### Phase 8G - Measure transfer on held-out live scenarios

Address: the remaining measured gap between simulator and game performance.

1. Start with scripted checks and one explicitly launched frozen-policy episode
   after input/execution gates pass. Verify stopping on death/interruption and
   preserving the watcher's final state.
2. Additional held-out comparisons are optional separate launches. Record greedy
   versus sampled mode; do not require a ten/fifty-episode pilot or build an
   automatic live episode/reset loop to complete inference deployment.
3. Match map, stats, gear, encounter population, task and duration as far as the
   live game permits. State unavoidable setup differences in each report.
4. Report survival time, cast outcomes, action failures, accepted movement,
   latency and stop reason, with attributed damage/kills when available. A
   single run provides outcomes, not a statistically established win rate.
   Include reward only under an explicit comparable evaluation profile.
5. Feed failures into capture, regression, and mechanics fixes. Retrain only when
   mechanics/contracts changed or evidence supports robustness training. Bound
   any timing/stat randomization by measurements. Retraining happens in the
   simulator; live fine-tuning is outside this inference-only workflow.

Deliverables: held-out transfer report, remaining ranked discrepancies, and
evidence-based choice to retain the actor or retrain it in the simulator.

Exit: agreed observable parity tests pass; transfer performance and limitations
are measured against criteria set before the main evaluation, rather than a
success percentage chosen after results are known.

### Phase 8H - Make parity repeatable and update the documentation

1. Run feature, lifecycle, inference, reward, and mechanics replay checks as a
   focused regression suite, retaining held-out captures for final validation.
2. Keep real-server checks separately invocable; mocked transport success does
   not establish game fidelity. Mark real checks unvalidated when unavailable.
3. Update README action counts, current 56/80-value schemas, Y termination rules,
   launcher usage, artifact compatibility, and the original Phase 8 references.
4. Rebenchmark simulator throughput after mechanics changes, preserving event
   ordering/RNG behavior. Optimize only measured regressions.

Exit: a fresh checkout can reproduce offline parity reports and invoke the
documented bounded live check. Every remaining unsupported case is recorded.

## Dependencies and completion order

Execute 8A -> 8B -> 8C -> required 8D execution checks -> 8F -> 8G -> 8H.
8E diagnostics and deeper 8D mechanics comparisons are optional follow-up work,
not dependencies on implementing rewards or reconstructing hidden server state.
Model-loader preparation
can begin after checkpoint inventory, but policy-driven live evaluation depends
on confirmed telemetry, transition handling, execution, and input compatibility.
Mechanics calibration and optional metrics can proceed independently once the
capture and lifecycle are reliable. Do not let BC/Minari migration, additional
classes, a C++ rewrite, or a new curriculum block this work.

The first concrete implementation should be capture plus a single-run
action/history/death-stop test harness. It establishes trustworthy policy inputs
and stop behavior without implementing a live training/reset system. Launching
the watch UI, its read-only telemetry channel, LIVE controls and final-state
retention are detailed in [WATCH_POLICY.md](../Training/Mystic_Sim/WATCH_POLICY.md).

## Local evidence

- User-confirmed shared live map: `../Tests/fixtures/simulation/map53.json`;
  matches `Mystic_Sim/data/map53.json` by SHA-256 in this checkout.
- User-supplied ZMQ state structure reference:
  `../Tests/fixtures/simulation/example_full_state.json` (example values).
- `Test_Env/Mystic.py`: default actions, receive/send/return order, reset and wins.
- `Test_Env/Env_conditions.py`: live encoding, HP deltas, disappearance heuristic,
  reward coefficients, map loss and truncation.
- `Config.yaml` and `BaseEnv.py`: 26-input default, 256-step limit, REP transport.
- `Mystic_Sim/actions.py`, `observation.py`, `env.py`: seven actions and 56 inputs.
- `../Training/Mystic_Sim/history.py`: 24 appended values, confirmed outcomes,
  cast ages and static terrain sensors.
- `Mystic_Sim/targeting.py`, `engine.py`, `movement.py`, `config.py`, `rewards.py`:
  execution, timing, source-derived baseline, rewards and actual end conditions.
- `../Training/PPO_lstm_server.py` and `../Training/Mystic_Sim/recurrent.py`:
  shared live trunk versus separate replica trunks and configurable encoders.
- `../Training/Mystic_Sim/core.py` and `evaluate.py`: metadata, normalization,
  checkpoint validation and exact architecture reconstruction.
- `../Automation/Bridge/ws_zmq_bridge.py`: serialized request stream and socket
  recovery, without authoritative game execution acknowledgements.
- Sibling extension `../../yugen-battler/yugen-battler/battler.js`: payload,
  targeting, predicted movement, auto-facing, reset spell, and loop scheduling.
- `PYTHON_INTEGRATION_PLAN.md`, existing Phase 8: intended live parity scope;
  older feature-size references need updating during implementation.
