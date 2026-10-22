"""Strict live inputs using the simulator's encoder, bounds and observable history.

Only construction reads the shared map. Encoding never resets/steps a simulator,
loads weights, fabricates enemies, or infers an outcome from a sent command.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np

from Custom_enviornments.Mystic_Sim.actions import (
    ACTIONS, ACTION_SCHEMA, OBSERVATION_SCHEMA,
)
from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
from Custom_enviornments.Mystic_Sim.map_loader import DEFAULT_MAP_PATH
from Custom_enviornments.Mystic_Sim.observation import encode_observation, nearest_monsters
from Custom_enviornments.Mystic_Sim.state import Direction
from Training.Mystic_Sim.history import HISTORY_SCHEMA, ObservableHistory, observation_high

from .lifecycle import EXECUTION_SCHEMA, finite_number, map_id

OBSERVATION_CONTEXT_SCHEMA = 'mystic_live_observation_v1'
PREPROCESSING = 'divide_by_box_high_v1'
DIRECTIONS = {name.lower(): value for name, value in Direction.__members__.items()}


def entity_key(value):
    """Numeric IDs sort numerically; opaque strings follow in lexical order.

    Canonical decimal strings and integers identify the same entity. Booleans,
    floats and empty/whitespace strings cannot become IDs by coercion.
    """
    if type(value) is int:
        return (0, value)
    if isinstance(value, str) and value and value == value.strip():
        if re.fullmatch(r'-?(0|[1-9][0-9]*)', value):
            return (0, int(value))
        return (1, value)
    raise ValueError(f'Invalid entity ID: {value!r}')


def _resource(record, name, path, *, positive=False):
    value = record.get(name)
    if not finite_number(value) or value < 0 or (positive and value == 0):
        raise ValueError(f'{path}.{name}: required finite {"positive" if positive else "nonnegative"} number')
    return value


class LiveObservationEncoder:
    """Convert a JS world projection into the exact existing 56-value contract."""
    def __init__(self, *, map_path=DEFAULT_MAP_PATH):
        # Reuse both the immutable map and actual Box, without reset or an engine.
        env = MysticSimEnv(map_path=map_path)
        self.map = env.map_definition
        self.space = env.observation_space
        env.close()

    def _character(self, record, path):
        coordinates = []
        for name, limit in (('mapX', self.map.width), ('mapY', self.map.height)):
            value = record.get(name)
            if type(value) is not int or not 0 <= value < limit:
                raise ValueError(f'{path}.{name}: required integer tile coordinate in [0, {limit - 1}]')
            coordinates.append(value)
        return SimpleNamespace(
            entity_id=entity_key(record.get('id')), x=coordinates[0], y=coordinates[1],
            hp=_resource(record, 'hp', path), max_hp=_resource(record, 'maxHp', path, positive=True),
            mp=_resource(record, 'mp', path), max_mp=_resource(record, 'maxMp', path),
            alive=record['hp'] > 0,
        )

    def world(self, payload):
        if not isinstance(payload, dict) or not isinstance(payload.get('player'), dict):
            raise ValueError('worldState.player: required object')
        if map_id(payload['player'].get('mapID')) != self.map.map_id:
            raise ValueError('unsupported_map')
        player = self._character(payload['player'], 'player')
        direction = payload['player'].get('direction')
        if not isinstance(direction, str) or direction not in DIRECTIONS:
            raise ValueError('player.direction: required up/down/left/right/idle')
        player.facing = DIRECTIONS[direction]
        entities = payload.get('entities')
        if not isinstance(entities, list):
            raise ValueError('worldState.entities: required list (an empty list means no observed entities)')
        monsters, seen = {}, set()
        for index, entity in enumerate(entities):
            path = f'entities[{index}]'
            if not isinstance(entity, dict) or not isinstance(entity.get('type'), str) or not entity['type']:
                raise ValueError(f'{path}.type: required entity classification')
            if entity['type'] != 'monster':
                continue
            if type(entity.get('isCurrentPlayer')) is not bool:
                raise ValueError(f'{path}.isCurrentPlayer: required boolean for monsters')
            if entity['isCurrentPlayer']:
                continue
            key = entity_key(entity.get('id'))
            if key in seen:
                raise ValueError(f'{path}.id: duplicate normalized monster ID {entity["id"]!r}')
            seen.add(key)
            if not finite_number(entity.get('hp')):
                raise ValueError(f'{path}.hp: required finite number')
            if entity['hp'] <= 0:
                continue
            # Validate all living visible monsters, including those beyond slot 10.
            monsters[key] = self._character(entity, path)
        return SimpleNamespace(player=player, monsters=monsters, map=self.map)

    def encode(self, payload):
        result = encode_observation(self.world(payload))
        if not self.space.contains(result):
            raise ValueError('Encoded world is outside the simulator observation Box')
        return result

    def inspect(self, payload):
        world = self.world(payload)
        return dict(visible_living_monsters=len(world.monsters),
                    ranked_ids=[list(m.entity_id) for m in nearest_monsters(world)],
                    coordinates='integer tiles; +x right, +y down; origin 0,0',
                    distance='Manhattan recomputed from coordinates; wire distance is unused')


def validate_observation_context(context):
    if not isinstance(context, dict) or context.get('schema') != OBSERVATION_CONTEXT_SCHEMA:
        raise ValueError('80-value input requires authoritative observationContext')
    if not isinstance(context.get('clockId'), str) or not context['clockId']:
        raise ValueError('observationContext.clockId: required server clock identity')
    if not finite_number(context.get('stateTimeMs')) or context['stateTimeMs'] < 0:
        raise ValueError('observationContext.stateTimeMs: required finite nonnegative server time')
    return context


class InputUnavailable(ValueError):
    """A valid base observation exists, but no confirmed history transition does."""


class LivePolicyInputs:
    """One causal input stream per explicitly launched run; no guessed history.

    `completion` is a correlation-validated entry from SingleRun.history.completed.
    The authoritative observation context must describe this exact world snapshot.
    """
    def __init__(self, size=56, *, encoder=None):
        if type(size) is not int or size not in (56, 80):
            raise ValueError('Live input size must be explicitly 56 or 80')
        self.size = size
        self.encoder = encoder or LiveObservationEncoder()
        self.high = (self.encoder.space.high.copy() if size == 56
                     else observation_high(self.encoder.space.high))
        self.low = (np.concatenate((self.encoder.space.low, np.zeros(24, dtype=np.float32)))
                    if size == 80 else self.encoder.space.low.copy())
        self.schema = OBSERVATION_SCHEMA if size == 56 else HISTORY_SCHEMA
        self.history = ObservableHistory(self.encoder.space.high, blocked_cells=self.encoder.map.blocked_cells)
        self.clock_id = self.state_time_ms = None
        self.completed_ids = set()

    def encode(self, world, *, observation_context=None, completion=None):
        base = self.encoder.encode(world)
        if self.size == 56:
            values = base
        else:
            context = validate_observation_context(observation_context)
            stamp = context['stateTimeMs']
            if self.clock_id is not None and context['clockId'] != self.clock_id:
                raise ValueError('execution_clock_changed')
            if self.state_time_ms is None:
                if completion is not None:
                    raise ValueError('Cannot attach a completion without the initial observation clock')
                # This is the explicit episode history reset, never a missing-feedback fallback.
                self.clock_id, self.state_time_ms = context['clockId'], stamp
            else:
                if completion is None:
                    raise InputUnavailable('awaiting_confirmed_transition')
                self._advance(completion, context)
            values = self.history.encode(base)
        if values.dtype != np.float32 or values.shape != (self.size,) or not np.isfinite(values).all():
            raise ValueError('Invalid encoded policy input')
        if np.any(values < self.low) or np.any(values > self.high):
            raise ValueError('Policy input outside simulator bounds')
        return dict(schema=self.schema, action_schema=ACTION_SCHEMA, actions=list(ACTIONS),
                    values=values.tolist(), normalized=(values / self.high).tolist(),
                    preprocessing=PREPROCESSING, size=self.size,
                    source_snapshot_timestamp=world.get('timestamp'),
                    observation_context=deepcopy(observation_context))

    def _advance(self, completion, context):
        ack = completion.get('acknowledgement') if isinstance(completion, dict) else None
        if not isinstance(ack, dict) or ack.get('schema') != EXECUTION_SCHEMA:
            raise ValueError('Required confirmed execution acknowledgement')
        action_id = completion.get('action_id')
        if not isinstance(action_id, str) or not action_id or action_id in self.completed_ids:
            raise ValueError('Missing or duplicate history action ID')
        action = completion.get('action')
        if type(action) is not int or not 0 <= action < len(ACTIONS):
            raise ValueError('Invalid confirmed seven-action index')
        if (ack.get('actionId') != action_id or ack.get('requestId') != completion.get('request_id')
                or ack.get('actionLabel') != ACTIONS[action]):
            raise ValueError('Unmatched history completion')
        if (ack.get('clockId') != context['clockId'] or ack.get('stateTimeMs') != context['stateTimeMs']):
            raise ValueError('Execution feedback must describe the same clock and snapshot time')
        attempted, stamp = ack.get('attemptedAtMs'), context['stateTimeMs']
        if (not finite_number(attempted) or not self.state_time_ms <= attempted <= stamp
                or stamp <= self.state_time_ms):
            raise ValueError('Invalid history execution interval')
        if not all(type(ack.get(key)) is bool for key in ('applied', 'movementBlocked', 'cooldownRejected')):
            raise ValueError('Missing explicit confirmed action outcome flags')
        if ((ack['applied'] and (ack['movementBlocked'] or ack['cooldownRejected']))
                or (ack['movementBlocked'] and action >= 4) or (ack['cooldownRejected'] and action < 4)):
            raise ValueError('Inconsistent confirmed action outcome flags')
        # Reuse training's update, including its advancing unseen-age channels.
        # Measured state intervals replace the simulator's default 200ms.
        self.history.update(None, action, applied=ack['applied'], blocked=ack['movementBlocked'],
                            cooldown_rejected=ack['cooldownRejected'], elapsed_ms=stamp - self.state_time_ms)
        if action >= 4 and ack['applied']:
            # Live command execution can start later than the previous snapshot.
            self.history.features[10 + action - 4] = min(1., (stamp - attempted) / 10000)
        self.state_time_ms = stamp
        self.completed_ids.add(action_id)


def audit_capture(path, *, encoder=None):
    """Check observed enemy/coordinate availability; never claim live mechanics parity."""
    from .server import decode_message
    encoder = encoder or LiveObservationEncoder()
    report = dict(path=str(path), evidence_origins=[], frames=0, valid_56=0, dead_frames=0,
                  errors=[], maximum_visible_living_monsters=0, authoritative_context_frames=0,
                  execution_feedback_frames=0, history_parity='requires correlated authoritative replay',
                  coordinate_convention='0..99 integer tiles, +x right, +y down')
    origins = set()
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = decode_message(line)
                if not isinstance(record, dict):
                    raise ValueError('Capture record must be an object')
                origin = record.get('evidence_origin')
                if isinstance(origin, str):
                    origins.add(origin)
                message = record.get('request') if record.get('event') == 'client_request' else None
                if not isinstance(message, dict) or message.get('type') != 'ai_tick':
                    continue
                report['frames'] += 1
                world = message.get('worldState')
                player = world.get('player', {}) if isinstance(world, dict) else {}
                if finite_number(player.get('hp')) and player['hp'] <= 0:
                    report['dead_frames'] += 1
                    continue
                encoder.encode(world)
                report['valid_56'] += 1
                info = encoder.inspect(world)
                report['maximum_visible_living_monsters'] = max(report['maximum_visible_living_monsters'], info['visible_living_monsters'])
                if 'observationContext' in message:
                    validate_observation_context(message['observationContext'])
                    report['authoritative_context_frames'] += 1
                if isinstance(message.get('execution'), dict):
                    report['execution_feedback_frames'] += 1
            except (ValueError, TypeError) as error:
                report['errors'].append(dict(line=line_number, error=str(error)))
    report['evidence_origins'] = sorted(origins) or ['unverified']
    report['ten_slot_coverage_observed'] = report['maximum_visible_living_monsters'] >= 10
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='Audit saved live payload coverage; no game or policy execution')
    parser.add_argument('--capture', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True, help='New JSON report; existing evidence is never overwritten')
    args = parser.parse_args(argv)
    report = dict(schema='mystic_live_input_audit_v1', captures=[audit_capture(path) for path in args.capture],
                  deployment_checkpoint=None, live_parity_validated=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'Input audit saved: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
