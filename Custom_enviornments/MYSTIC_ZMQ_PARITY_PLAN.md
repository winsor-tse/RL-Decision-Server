# Mystic ZMQ and simulation parity plan

Prepared 2026-10-21 from the local implementations. Status: planning only;
no runtime behavior or checkpoint contracts changed.

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

Findings below are established from local code, including the sibling
`../yugen-battler/yugen-battler/battler.js`. The browser's actually loaded
extension and connected server version have not been verified. Distinguish
these code differences from mechanics discrepancies that still need live data.

## Major differences and their impact

| Priority | Area | Current ZMQ/browser path | Python replica | Required response |
|---|---|---|---|---|
| P0 | Action contract | Mystic defaults to 11 actions: four moves, attack, slots 1/2/3/5/6/7. | Seven: four moves and slots 1/2/3. ID 4 means Arcane Blast rather than attack. | Explicit seven-action live adapter; validate labels and schema before sending commands. |
| P0 | Observation contract | 26 values, five monsters; distance-only stable sorting preserves payload order on ties. No explicit dead-monster filter in observation parsing. | 56 base values: ten living enemies and signed dx/dy for the first five; distance then entity-ID sorting. Recurrent policies consume 80 values. | Shared, versioned encoding for the new adapter, including filtering, ID handling, padding, signed bounds, and tie rules. |
| P0 | Recurrent inputs | No confirmed action outcomes, collision reasons, cooldown rejections, or cast acknowledgements in the inspected world-state builder. | Adds 24 history/sensor values, including confirmed outcomes, successful-cast ages, and four local terrain rays. | Add authoritative telemetry and the matching static collision map. Missing required telemetry blocks deployment of the existing 80-input policy. |
| P0 | Transition alignment | `step(a)` receives a world snapshot, sends action a, then returns that same received snapshot and its reward. | Applies a, advances 200 ms, then returns the resulting observation/reward. | Pair each action with its actual subsequent state. Current live reward can be associated with a different action; blocked-movement shaping is particularly exposed. |
| P0 | Decision timing | Browser awaits the response, applies the action, then sleeps 200 ms. Snapshot timestamps use client `Date.now()`. Bridge `serverTime` is Python wall time. | Exact 200 ms simulated decisions and integer event deadlines. | Measure command application and authoritative state times. The browser interval includes response/processing delay; neither existing timestamp is proof of game-server execution time. |
| P0 | Targeting | Browser ranks by Manhattan distance, favors monsters but can fall back to other entities, and has no 16-X/10-Y eligibility filter in target selection. With no target, all three spell commands are skipped. | Ranks eligible monsters by Euclidean distance then ID inside an inclusive 16-X/10-Y rectangle. Slots 2/3 fall back to self. | Select and validate one execution specification against connected behavior; test ties, rectangle edges, unavailable targets, and self fallback. |
| P0 | Facing and movement | Browser faces the selected target before casting; replica cast does not update player facing. Browser also mutates local coordinates before authoritative movement confirmation. | Movement checks terrain, occupancy, and bounds synchronously. | Validate cast-facing behavior and distinguish client prediction from accepted movement. Do not count local coordinate changes as movement acknowledgements. |
| P0 | Reset | Mystic returns the pre-reset snapshot immediately after sending `reset=True`. Browser implements reset by self-casting slot 4. No reset-completion handshake is checked. | Rebuilds a seeded world and returns its new initial observation. | Wait for confirmed task readiness and a fresh post-reset state; reset memory/history then. Do not pair reset activity with a normal policy action. |
| P1 | Damage and kills | HP-delta rewards; nearby disappeared entities can be treated as kills. Entity reward tracking includes non-current-player entities without the observation's monster-type filter. Healing can yield positive damage-taken reward. | Attributed damage/death events, damage clamped to actual HP removed, no healing reward in the default combat profile. | Use typed identities, authoritative damage/death attribution, and deduplication. Treat disappearance as unknown until confirmed. |
| P1 | Reward scale and shaping | +10 per inferred kill, -100 on death/map departure, no explicit win bonus; proximity shaping, flat -10 blocked-move penalty, low-HP shaping, incoming HP delta weighted 20. | +10 per confirmed kill, +200 win, -200 death, -25 timeout; default incoming damage weight 50, time cost, escalating blocked-move penalties and cooldown penalty; no proximity reward. | Define a shared task/reward profile for new live training when telemetry supports it. Compare frozen policies through outcome metrics before comparing returns. |
| P1 | Episode rules | Default 256 steps; truncates at Y < 25; penalizes Y < 31; leaving map53 is a loss. | Default 1024 steps; Y outside inclusive 30..86 terminates as failure; penalties below 31 and above 85. Five kills is the default goal on both paths. | Version the task definition and termination precedence; distinguish task failures from time limits and test PPO bootstrapping. |
| P1 | Model loading | Live LSTM uses a shared embedding/LSTM and Tanh actor head; current evaluator loads that architecture. | Separate actor/critic branches; configurable flat or attention encoder; default linear actor; saved preprocessing scales and architecture metadata. | Load the exact checkpoint architecture and preprocessing. Resizing the live network does not make simulator weights compatible. |
| P1 | World and mechanics coverage | Visible client registry, actual player equipment/stats, other actors, and hidden server timers. | Fixed level-135 Mystic baseline, gear off, map53, Innie-only population (80 in full scenario), selected status modifiers off. | Match observable task setup; measure remaining mechanics differences. Do not equate client registry visibility with complete server state. |
| P2 | Throughput and resume | One external game stream; socket recovery does not restore world state. | Many independent seeded worlds; training resume starts fresh episodes. | Keep simulator batching and live session handling separate; live resume resets episode state and memory. |

The README's historical 15-action list and several integration-plan references
to 26/50-input policies are stale. Current code and the selected checkpoint's
metadata must determine the contract. In particular, simulator Y exits currently
terminate; older prose describing them as truncations is outdated.

## Phased implementation

### Phase 8A - Freeze the baseline and establish evidence

Address: uncertain deployed versions, stale documentation, missing captures.

1. Record repository revision and selected checkpoint metadata, architecture,
   feature schema, action ordering, preprocessing, map hash, and task settings.
2. Identify the actually loaded browser extension and connected game build.
3. Add optional JSONL capture at the live boundary: session/episode/request IDs,
   raw state, command, selected target, receipt/send/application times and
   authoritative timestamps or outcomes where available. Separate unavailable
   fields from observations. Keep the existing human recorder independent.
4. Capture scripted moves, blocked moves, each spell, no-target casts, cooldown
   rejection, kills, map transitions, reset, and disconnect/reconnect.
5. Produce a field-availability matrix and correct obsolete contract references.

Deliverables: baseline manifest, versioned capture format, capture fixtures,
and a difference report with observed/inferred/unavailable evidence labels.

Exit: every required input and outcome has a known producer or an explicit
missing-data dependency. A parser fixture from map3471 is not live map53 proof.

### Phase 8B - Correct transitions, reset, and session lifecycle

Address: action/reward misalignment, variable time, pre-reset observations.

1. Introduce a separate live adapter with explicit waiting-for-state,
   action-pending, waiting-for-result, and reset-pending states. Preserve the
   ZMQ REQ/REP protocol's required reply sequence while waiting for later state;
   use an explicit transport idle response where necessary, outside policy IDs.
2. Define a transition as `(state_t, action_t, result_t+1)`. Correlate commands,
   authoritative outcomes, and the subsequent eligible snapshot. Request IDs
   identify exchanges; they alone do not establish execution acknowledgement.
3. Measure elapsed time and jitter. Do not replace missing server timing with a
   claimed fixed 200 ms interval. Decide from measurements whether live scheduling
   can match training or a new timing-tolerant training contract is required.
4. Wait for a confirmed fresh map53/task-ready state after reset. Clear history,
   recurrent memory, kill counters, and pending transitions on episode boundaries.
5. Reject duplicate/stale/cross-session results. On timeout or reconnect,
   invalidate uncertain transitions and resynchronize without replaying casts.
6. Keep slow per-tick diagnostic file writes optional in the new adapter and
   measure their contribution to latency.

Deliverables: adapter lifecycle, timing report, transition/reset fixtures.

Exit: scripted successes/failures attach to the correct action exactly once;
reset returns a post-reset observation; timeout/reconnect tests preserve ZMQ
ordering and cannot leak one episode's state into another.

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
5. Load and verify the live static collision layer and compute the same four
   four-tile terrain rays. Exclude entity occupancy and reload on map changes.
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
2. Compare authoritative accepted/blocked movement with the actual terrain,
   occupancy, map bounds, and client correction behavior.
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

### Phase 8E - Align task, rewards, and outcome accounting

Address: incomparable returns, inferred kills, healing artifacts, episode limits.

1. Add a versioned task profile for map53, five confirmed attributed kills,
   death/map-change handling, Y bounds and penalties, and time limit. Initially
   target the simulator's current task for transfer comparisons where feasible.
2. Deduplicate authoritative damage/death events; distinguish despawn, visibility
   loss, map departure, and other players' kills from the policy's own kills.
3. Share default combat reward calculation only when the live event stream can
   support equivalent attribution. Otherwise label live rewards separately and
   compare confirmed outcomes; snapshot HP loss cannot recover simultaneous
   damage and healing exactly.
4. Test death/fifth-kill/boundary/timeout precedence, one-time terminal rewards,
   and the training rule: task termination does not bootstrap; external time
   limits bootstrap from the final observation with pre-reset recurrent memory.

Deliverables: task profile, outcome reducer, reward/end-condition regression cases.

Exit: matched event streams produce matching outcomes and reward components;
remaining live limitations are visible. Raw returns are compared only under
equivalent task and reward profiles.

### Phase 8F - Load the exact frozen simulator policy live

Address: incompatible architecture, preprocessing, checkpoint/session handling.

1. Add a dedicated inference entry point that reconstructs the selected
   simulator checkpoint's exact flat/attention encoder, actor head, recurrent
   branches, and normalization. Reuse its model code and validated loading
   behavior rather than duplicating the legacy live LSTM.
2. Validate action labels, observation schema, feature order, architecture,
   preprocessing, and map/task requirements before transmitting policy commands.
   Record simulator engine provenance; do not claim the live server has its hash.
3. Maintain independent actor/critic memory and history per session; reset both
   on confirmed episode/reconnect/map boundaries. Do not advance memory twice
   for bootstrap or duplicate observations.
4. Replay fixed observation sequences through offline and live inference;
   compare normalized inputs, logits, greedy actions, and hidden/cell states.

Deliverables: live evaluation CLI, strict compatibility checks, inference replay.

Exit: replay matches within defined numerical tolerance; incompatible models
fail before any game action. Evaluation changes no weights or optimizer state.

### Phase 8G - Measure transfer on held-out live scenarios

Address: the remaining measured gap between simulator and game performance.

1. Start with scripted checks and a bounded frozen-policy session after prior
   gates pass, then collect a pilot such as ten comparable episodes.
2. Expand to an agreed held-out bank (for example, 50 episodes per mode where
   live access permits). Evaluate greedy and sampled behavior separately and
   report uncertainty; these counts are proposed budgets, not proof thresholds.
3. Match map, stats, gear, encounter population, task and duration as far as the
   live game permits. State unavoidable setup differences in each report.
4. Report confirmed kill/win rate, survival time, attributed damage, cast success,
   failures by reason, accepted movement, latency and reset reliability. Include
   reward only with its named profile and equivalent elapsed-time context.
5. Feed failures into capture, regression, and mechanics fixes. Retrain only when
   mechanics/contracts changed or evidence supports robustness training. Bound
   any timing/stat randomization by measurements; live fine-tuning follows
   explicit parity validation rather than hiding adapter defects.

Deliverables: held-out transfer report, remaining ranked discrepancies, and
evidence-based choice to retain, retrain, or begin limited live fine-tuning.

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

Execute 8A -> 8B -> 8C -> 8D -> 8E -> 8F -> 8G -> 8H. Model-loader preparation
can begin after checkpoint inventory, but policy-driven live evaluation depends
on confirmed telemetry, transition handling, execution, and input compatibility.
Mechanics calibration and task accounting can proceed independently once the
capture and lifecycle are reliable. Do not let BC/Minari migration, additional
classes, a C++ rewrite, or a new curriculum block this work.

The first concrete implementation should be capture plus a transition/reset
test harness. It gives the remaining work trustworthy evidence and avoids
training against rewards assigned to the wrong action.

## Local evidence

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
