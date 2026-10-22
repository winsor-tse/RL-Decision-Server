"""One explicitly started inference run. No training, resets, or inferred outcomes.

The actor interface accepts Phase 8C inputs; checkpoint loading is Phase 8F.
The current JS payload has no execution acknowledgements; missing feedback keeps
an action pending, then interrupts the run. It never becomes a guessed success.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import math
import statistics
import time
import uuid

import numpy as np

from Custom_enviornments.Mystic_Sim.actions import ACTIONS

BRIDGE_SCHEMA = 'mystic_single_run_bridge_v1'
EXECUTION_SCHEMA = 'mystic_live_execution_v1'


class RunState(str, Enum):
    WAITING = 'waiting_for_first_valid_state'
    RUNNING = 'running'
    ACTION_PENDING = 'action_pending'
    STOPPED = 'stopped'
    INTERRUPTED = 'interrupted'


@dataclass(frozen=True)
class PolicyChoice:
    action: int
    next_memory: object = None
    probabilities: tuple | None = None


@dataclass(frozen=True)
class PendingAction:
    action_id: str
    request_id: str
    action: int
    label: str
    input_state: dict
    selected_at_monotonic: float


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def map_id(value):
    if type(value) is int:
        return value
    if isinstance(value, str) and value.startswith('map') and value[3:].isdigit():
        return int(value[3:])
    raise ValueError('Expected an integer map ID or mapNN string')


class TimingSeries:
    def __init__(self):
        self.count = 0
        self.total = 0.
        self.minimum = self.maximum = None
        self.recent = deque(maxlen=1024)

    def add(self, value):
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        self.recent.append(value)

    def report(self):
        return dict(samples=self.count, mean_ms=self.total / self.count if self.count else None,
                    minimum_ms=self.minimum, maximum_ms=self.maximum,
                    recent_median_ms=statistics.median(self.recent) if self.recent else None)


class ConfirmedHistory:
    """Latest confirmed facts; cast ages come from one explicit execution clock."""
    def __init__(self):
        self.previous_action = None
        self.applied = self.blocked = self.cooldown_rejected = None
        self.clock_id = self.state_time_ms = None
        self.cast_times = {slot: None for slot in (1, 2, 3)}
        self.completed = deque(maxlen=128)

    def view(self):
        return dict(previous_action=self.previous_action, action_applied=self.applied,
                    movement_blocked=self.blocked, cooldown_rejected=self.cooldown_rejected,
                    clock_id=self.clock_id, state_time_ms=self.state_time_ms,
                    cast_seen={str(slot): stamp is not None for slot, stamp in self.cast_times.items()},
                    cast_age_ms={str(slot): min(10000., max(0., self.state_time_ms - stamp))
                                 if stamp is not None and self.state_time_ms is not None else None
                                 for slot, stamp in self.cast_times.items()})

    def apply(self, pending, acknowledgement):
        self.previous_action = pending.action
        self.applied = acknowledgement['applied']
        self.blocked = acknowledgement['movementBlocked']
        self.cooldown_rejected = acknowledgement['cooldownRejected']
        self.clock_id = acknowledgement['clockId']
        self.state_time_ms = acknowledgement['stateTimeMs']
        if pending.action >= 4 and self.applied:
            self.cast_times[pending.action - 3] = acknowledgement['attemptedAtMs']
        self.completed.append(dict(action_id=pending.action_id, request_id=pending.request_id,
                                   action=pending.action, acknowledgement=deepcopy(acknowledgement)))


class SingleRun:
    """Pure controller. actor.initial_state() runs once; actor.choose() never trains.

    choose(input, history, memory) returns PolicyChoice. With input_adapter,
    input is the raw float32 56/80-vector; the model divides by Box.high once.
    Without an adapter, the existing raw-world actor interface is retained.
    """
    def __init__(self, actor=None, *, input_adapter=None, run_id=None, expected_map=53,
                 startup_timeout=30., idle_timeout=2., acknowledgement_timeout=2.,
                 monotonic=time.monotonic, wall_time=time.time):
        if any(not finite_number(v) or v <= 0 for v in
               (startup_timeout, idle_timeout, acknowledgement_timeout)):
            raise ValueError('Run timeouts must be positive finite seconds')
        self.actor = actor
        self.input_adapter = input_adapter
        if input_adapter is not None and input_adapter.encoder.map.map_id != expected_map:
            raise ValueError('Input adapter map must match the run map')
        self.run_id = run_id or uuid.uuid4().hex
        self.expected_map = expected_map
        self.startup_timeout, self.idle_timeout = startup_timeout, idle_timeout
        self.acknowledgement_timeout = acknowledgement_timeout
        self.monotonic, self.wall_time = monotonic, wall_time
        self.created_at = monotonic()
        self.last_received_at = None
        self.state = RunState.WAITING
        self.stop_reason = None
        self.owner = self.player_identity = None
        self.memory = None
        self.initialized = False
        self.history = ConfirmedHistory()
        self.pending = None
        self.current_state = self.previous_state = self.last_decision = None
        self.current_base_observation = self.current_policy_input = None
        self.policy_input_status = 'waiting_for_first_valid_state'
        self.last_timestamp = None
        self.seen_requests = set()
        self.completed_action_ids = set()
        self.action_count = 0
        self.events = deque(maxlen=128)
        self.client_intervals = TimingSeries()
        self.receive_intervals = TimingSeries()
        self.confirmed_intervals = TimingSeries()
        self.ack_latencies = TimingSeries()
        self.queue_age = TimingSeries()

    @property
    def terminal(self):
        return self.state in (RunState.STOPPED, RunState.INTERRUPTED)

    def _event(self, name, **details):
        self.events.append(dict(event=name, monotonic=self.monotonic(), **details))

    def stop(self, reason='manual_stop', *, interrupted=False):
        if not self.terminal:
            self.state = RunState.INTERRUPTED if interrupted else RunState.STOPPED
            self.stop_reason = reason
            self._event('run_ended', reason=reason, pending_action_id=self.pending.action_id if self.pending else None)
        # Pending facts and the final frame remain available, never retried.

    def check_deadlines(self):
        if self.terminal:
            return
        now = self.monotonic()
        if self.pending and now - self.pending.selected_at_monotonic >= self.acknowledgement_timeout:
            self.stop('acknowledgement_timeout', interrupted=True)
        elif self.state == RunState.WAITING and now - self.created_at >= self.startup_timeout:
            self.stop('startup_timeout', interrupted=True)
        elif self.last_received_at is not None and now - self.last_received_at >= self.idle_timeout:
            self.stop('state_timeout', interrupted=True)

    def _reply(self, request_id, *, move=None, reason=None, action_id=None):
        return dict(type='ai_result', requestId=request_id, move=move, reset=False,
                    serverTime=self.wall_time(), runId=self.run_id, actionId=action_id,
                    inferenceStatus=self.state.value, stopInference=self.terminal,
                    stopReason=self.stop_reason, idleReason=reason)

    def _context(self, message):
        context = message.get('bridgeContext')
        if not isinstance(context, dict) or context.get('schema') != BRIDGE_SCHEMA:
            return None
        bridge_id, connection_id = context.get('bridgeId'), context.get('connectionId')
        if not all(isinstance(v, str) and v for v in (bridge_id, connection_id)):
            return None
        return bridge_id, connection_id

    def _validate_world(self, world):
        if not isinstance(world, dict) or not isinstance(world.get('player'), dict):
            raise ValueError('Missing player state')
        player = world['player']
        identity = player.get('id')
        if type(identity) is not int and not (isinstance(identity, str) and identity):
            raise ValueError('Missing/invalid player identity')
        if not finite_number(world.get('timestamp')):
            raise ValueError('Missing/invalid client snapshot timestamp')
        # A confirmed death can already include death-map/teleport coordinates.
        # It must stop inference without requiring another valid living input.
        if finite_number(player.get('hp')) and player['hp'] <= 0:
            return type(identity).__name__, identity
        if map_id(player.get('mapID')) != self.expected_map:
            raise ValueError('unsupported_map')
        for name in ('mapX', 'mapY'):
            if type(player.get(name)) is not int or not 0 <= player[name] < 100:
                raise ValueError(f'Invalid player {name}')
        for name in ('hp', 'maxHp', 'mp', 'maxMp'):
            if not finite_number(player.get(name)) or player[name] < 0:
                raise ValueError(f'Invalid player {name}')
        if player['maxHp'] <= 0:
            raise ValueError('Invalid player maxHp')
        if player.get('direction') not in ('up', 'down', 'left', 'right', 'idle'):
            raise ValueError('Invalid player direction')
        if not isinstance(world.get('entities'), list):
            raise ValueError('Missing entities list')
        return type(identity).__name__, identity

    def _accept_acknowledgement(self, message):
        ack = message.get('execution')
        if not isinstance(ack, dict):
            return 'awaiting_execution_acknowledgement'
        pending = self.pending
        if ack.get('runId') != self.run_id:
            return 'foreign_run_acknowledgement'
        if not all(isinstance(ack.get(key), str) and ack[key]
                   for key in ('actionId', 'requestId', 'actionLabel')):
            return 'invalid_acknowledgement'
        if ack.get('actionId') in self.completed_action_ids:
            return 'duplicate_acknowledgement'
        if (ack.get('schema') != EXECUTION_SCHEMA or ack.get('actionId') != pending.action_id
                or ack.get('requestId') != pending.request_id or ack.get('actionLabel') != pending.label):
            return 'unmatched_acknowledgement'
        if not all(type(ack.get(key)) is bool for key in ('applied', 'movementBlocked', 'cooldownRejected')):
            return 'invalid_acknowledgement'
        if ((ack['applied'] and (ack['movementBlocked'] or ack['cooldownRejected']))
                or (ack['movementBlocked'] and pending.action >= 4)
                or (ack['cooldownRejected'] and pending.action < 4)):
            return 'invalid_acknowledgement'
        if not isinstance(ack.get('clockId'), str) or not ack['clockId']:
            return 'missing_execution_clock'
        if not all(finite_number(ack.get(k)) and ack[k] >= 0 for k in ('attemptedAtMs', 'stateTimeMs')):
            return 'invalid_execution_timing'
        if ack['attemptedAtMs'] > ack['stateTimeMs']:
            return 'invalid_execution_timing'
        if self.history.clock_id is not None:
            if ack['clockId'] != self.history.clock_id:
                self.stop('execution_clock_changed', interrupted=True)
                return 'execution_clock_changed'
            if (ack['stateTimeMs'] <= self.history.state_time_ms
                    or ack['attemptedAtMs'] < self.history.state_time_ms):
                return 'stale_execution_timing'
        interval = ack['stateTimeMs'] - (self.history.state_time_ms if self.history.state_time_ms is not None
                                        else ack['attemptedAtMs'])
        self.confirmed_intervals.add(interval)
        self.ack_latencies.add((self.monotonic() - pending.selected_at_monotonic) * 1000)
        self.history.apply(pending, ack)
        self.completed_action_ids.add(pending.action_id)
        self.pending = None
        self.state = RunState.RUNNING
        self._event('action_confirmed', action_id=pending.action_id)
        return None

    def handle(self, message):
        """One REP request -> one response, including stopped/malformed requests."""
        request_id = message.get('requestId') if isinstance(message, dict) else None
        if self.terminal:
            return self._reply(request_id, reason='run_already_ended')
        self.check_deadlines()
        if self.terminal:
            return self._reply(request_id, reason='run_deadline')
        if not isinstance(message, dict):
            return self._reply(request_id, reason='invalid_message')
        if message.get('runId') not in (None, self.run_id):
            return self._reply(request_id, reason='foreign_run')
        owner = self._context(message)
        if owner is None:
            return self._reply(request_id, reason='missing_bridge_context')
        if self.owner is not None and owner != self.owner:
            return self._reply(request_id, reason='foreign_connection')
        if message.get('type') == 'bridge_event':
            if message.get('event') in ('client_disconnected', 'bridge_timeout', 'bridge_error'):
                self.owner = owner
                self.stop(message['event'], interrupted=True)
                return self._reply(request_id)
            return self._reply(request_id, reason='unknown_bridge_event')
        if message.get('type') != 'ai_tick' or not isinstance(request_id, str) or not request_id:
            return self._reply(request_id, reason='invalid_request')
        if request_id in self.seen_requests:
            return self._reply(request_id, reason='duplicate_request')
        self.seen_requests.add(request_id)
        self.owner = owner
        world = message.get('worldState')
        try:
            identity = self._validate_world(world)
            base = (self.input_adapter.encoder.encode(world)
                    if self.input_adapter is not None and world['player']['hp'] > 0 else None)
        except ValueError as error:
            self.policy_input_status = str(error)
            self.current_policy_input = None
            self.current_base_observation = None
            if self.initialized or str(error) == 'unsupported_map':
                if str(error) == 'unsupported_map':
                    self.current_state = deepcopy(world)
                self.stop('unsupported_map' if str(error) == 'unsupported_map' else 'invalid_state', interrupted=True)
            return self._reply(request_id, reason=str(error))
        if self.player_identity is not None and identity != self.player_identity:
            self.stop('player_identity_changed', interrupted=True)
            return self._reply(request_id)
        timestamp = world['timestamp']
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            return self._reply(request_id, reason='stale_snapshot')
        now = self.monotonic()
        context = message['bridgeContext']
        received = context.get('receivedMonotonicNs')
        if type(received) is int:
            age = now * 1000 - received / 1_000_000
            if age < 0 or age > self.idle_timeout * 1000:
                self.stop('stale_bridge_request', interrupted=True)
                return self._reply(request_id)
            self.queue_age.add(age)
        if self.last_timestamp is not None:
            self.client_intervals.add(timestamp - self.last_timestamp)
        if self.last_received_at is not None:
            self.receive_intervals.add((now - self.last_received_at) * 1000)
        self.last_timestamp, self.last_received_at = timestamp, now
        self.current_state = deepcopy(world)
        self.current_base_observation = base.tolist() if base is not None else None
        self.current_policy_input = None
        if world['player']['hp'] <= 0:
            self.policy_input_status = 'player_death'
            self.stop('player_death')
            return self._reply(request_id)
        completion = None
        if self.pending is not None:
            reason = self._accept_acknowledgement(message)
            if reason:
                self.policy_input_status = reason
                return self._reply(request_id, reason=reason)
            completion = self.history.completed[-1]
        if self.input_adapter is not None:
            from .inputs import InputUnavailable
            try:
                self.current_policy_input = self.input_adapter.encode(
                    world, observation_context=message.get('observationContext'), completion=completion)
                self.policy_input_status = 'ready'
            except InputUnavailable as error:
                self.policy_input_status = str(error)
                return self._reply(request_id, reason=str(error))
            except ValueError as error:
                self.policy_input_status = str(error)
                self._event('policy_input_error', detail=str(error))
                if self.initialized:
                    self.stop('invalid_policy_input', interrupted=True)
                return self._reply(request_id, reason=str(error))
        self.check_deadlines()
        if self.terminal:
            return self._reply(request_id)
        if not self.initialized:
            self.player_identity = identity
            try:
                self.memory = self.actor.initial_state() if self.actor is not None else None
            except Exception as error:
                self._event('actor_error', detail=str(error))
                self.stop('actor_initialization_error', interrupted=True)
                return self._reply(request_id)
            self.initialized = True
            self.check_deadlines()
            if self.terminal:
                return self._reply(request_id)
            self.state = RunState.RUNNING
            self._event('run_started', observation_only=self.actor is None)
        if self.actor is None:
            return self._reply(request_id, reason='observation_only')
        try:
            policy_input = (np.asarray(self.current_policy_input['values'], dtype=np.float32)
                            if self.input_adapter is not None else deepcopy(world))
            choice = self.actor.choose(policy_input, deepcopy(self.history.view()), self.memory)
        except Exception as error:
            self._event('actor_error', detail=str(error))
            self.stop('actor_error', interrupted=True)
            return self._reply(request_id)
        self.check_deadlines()
        if self.terminal:
            return self._reply(request_id)
        if not isinstance(choice, PolicyChoice) or type(choice.action) is not int or not 0 <= choice.action < len(ACTIONS):
            self.stop('invalid_policy_action', interrupted=True)
            return self._reply(request_id)
        if choice.probabilities is not None:
            probs = choice.probabilities
            if (not isinstance(probs, (tuple, list)) or len(probs) != len(ACTIONS) or not all(finite_number(p) and 0 <= p <= 1 for p in probs)
                    or not math.isclose(sum(probs), 1., abs_tol=1e-5)):
                self.stop('invalid_policy_distribution', interrupted=True)
                return self._reply(request_id)
        self.action_count += 1
        action_id = f'{self.run_id}:{self.action_count}'
        self.previous_state = deepcopy(world)
        self.memory = choice.next_memory
        self.pending = PendingAction(action_id, request_id, choice.action, ACTIONS[choice.action],
                                     deepcopy(world), self.monotonic())
        self.last_decision = dict(action_id=action_id, request_id=request_id,
                                  action=choice.action, label=ACTIONS[choice.action],
                                  probabilities=list(choice.probabilities) if choice.probabilities is not None else None,
                                  input_state=deepcopy(world), history=deepcopy(self.history.view()),
                                  policy_input=deepcopy(self.current_policy_input))
        self.state = RunState.ACTION_PENDING
        self._event('action_selected', action_id=action_id, label=ACTIONS[choice.action])
        return self._reply(request_id, move=ACTIONS[choice.action], action_id=action_id)

    def snapshot(self):
        return deepcopy(dict(schema='mystic_live_run_v1', run_id=self.run_id, state=self.state.value,
                             stop_reason=self.stop_reason, observation_only=self.actor is None,
                             initialized=self.initialized, action_count=self.action_count,
                             current_state=self.current_state, previous_state=self.previous_state,
                             current_base_observation=self.current_base_observation,
                             current_policy_input=self.current_policy_input,
                             policy_input_status=self.policy_input_status,
                             last_decision=self.last_decision, history=self.history.view(),
                             completed_actions=list(self.history.completed),
                             pending_action_id=self.pending.action_id if self.pending else None,
                             events=list(self.events), timing=dict(
                                 client_snapshot_intervals=self.client_intervals.report(),
                                 bridge_receive_intervals=self.receive_intervals.report(),
                                 confirmed_execution_intervals=self.confirmed_intervals.report(),
                                 acknowledgement_latency=self.ack_latencies.report(),
                                 bridge_queue_age=self.queue_age.report(),
                                 simulator_reference_step_ms=200,
                                 scheduling_parity='unvalidated; measure real confirmed timings before changing scheduling or simulator training')))
