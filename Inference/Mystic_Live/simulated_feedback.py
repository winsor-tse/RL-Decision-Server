"""Estimate the old 80-input history under an explicit sim/game parity assumption.

Only deterministic action acceptance and cooldown state are modeled. Positions,
resources and visible entities are resynchronized from every live snapshot.
No NPC simulation, random combat rolls, inferred damage or fabricated server ack.
"""
from copy import deepcopy

from Custom_enviornments.Mystic_Sim import cooldowns
from Custom_enviornments.Mystic_Sim.movement import collision_kind, OFFSETS
from Custom_enviornments.Mystic_Sim.state import CooldownState, Direction
from .commands import LiveCommands
from .inputs import LivePolicyInputs, entity_key
from .lifecycle import finite_number


class SimulatedFeedbackInputs(LivePolicyInputs):
    history_source = 'simulator_estimated'

    def __init__(self, config, *, encoder=None):
        super().__init__(80, encoder=encoder)
        self.requires_execution_feedback = False
        self.config = config
        self.cooldowns = CooldownState()
        self.commands = LiveCommands(self.encoder)
        self.last_timestamp = self.origin_timestamp = None
        self.pending_estimate = None
        self.latest_estimate = None
        self.world = self.payload = None

    def encode(self, world, *, observation_context=None, completion=None):
        if completion is not None:
            raise ValueError('Simulator estimates cannot be mixed with confirmed feedback')
        base = self.encoder.encode(world)
        stamp = world.get('timestamp')
        if not finite_number(stamp):
            raise ValueError('Simulator-estimated history requires a client snapshot timestamp')
        if self.last_timestamp is not None and stamp <= self.last_timestamp:
            raise ValueError('Simulator-estimated history requires a fresh snapshot')
        projected = self.encoder.world(world)
        projected.player.cooldowns = self.cooldowns
        projected.player.magic_immune = False  # Baseline simulator disables immunity.
        projected.terrain_collision = self.config.terrain_collision
        projected.occupancy = {(m.x, m.y): m.entity_id for m in projected.monsters.values()}
        # Other observed living players can also occupy a destination tile.
        for entity in world['entities']:
            if entity.get('type') == 'player' and not entity.get('isCurrentPlayer'):
                if not finite_number(entity.get('hp')):
                    raise ValueError('Occupancy prediction requires visible player HP')
                if entity['hp'] > 0:
                    x, y = entity.get('mapX'), entity.get('mapY')
                    if (type(x) is not int or type(y) is not int or
                            not 0 <= x < self.encoder.map.width or not 0 <= y < self.encoder.map.height):
                        raise ValueError('Occupancy prediction requires visible player tile coordinates')
                    projected.occupancy[(x, y)] = entity.get('id')
        for monster in projected.monsters.values():
            monster.magic_immune = False
        if self.origin_timestamp is None:
            self.origin_timestamp = stamp
        if self.pending_estimate is not None:
            estimate = self.pending_estimate
            self.history.update(None, estimate['action'], applied=estimate['applied'],
                                blocked=estimate['movement_blocked'],
                                cooldown_rejected=estimate['cooldown_rejected'],
                                elapsed_ms=stamp - self.last_timestamp)
            self.latest_estimate = dict(estimate, resulting_snapshot_timestamp=stamp,
                                        elapsed_ms=stamp - self.last_timestamp)
            self.pending_estimate = None
        self.world, self.payload = projected, deepcopy(world)
        self.last_timestamp = stamp
        result = self.describe(self.history.encode(base), world)
        result.update(history_source=self.history_source,
                      timing_basis='client_snapshot_intervals; action assumed at input snapshot',
                      initial_cooldowns='assumed_ready_at_launch',
                      estimated_transition=deepcopy(self.latest_estimate))
        return result

    def record_action(self, action):
        if self.world is None or self.pending_estimate is not None:
            raise ValueError('An estimate requires one fresh snapshot per chosen action')
        # Use the exact target chosen for live dispatch, preserving target namespace.
        command = self.commands.encode(self.payload, action)
        now = self.last_timestamp - self.origin_timestamp
        player = self.world.player
        blocked = None
        if action < 4:
            dx, dy = OFFSETS[Direction(action)]
            blocked = collision_kind(self.world, player.x + dx, player.y + dy)
            reason = 'blocked' if blocked else None
        else:
            spell = next(s for s in self.config.spells if s.slot == action - 3)
            target = (player if command['targetKind'] == 'self' else
                      self.world.monsters.get(entity_key(command['targetId']))
                      if command['targetKind'] == 'monster' else None)
            reason = cooldowns.rejection(player, spell, now, target)
            if reason is None:
                cooldowns.start(player, spell, now,
                                global_cooldown_ms=self.config.spell_rules.global_cooldown_ms)
        self.pending_estimate = dict(source=self.history_source, action=action,
                                     applied=reason is None, movement_blocked=blocked is not None,
                                     cooldown_rejected=reason in ('cooldown', 'family_cooldown', 'global_cooldown'),
                                     failure_reason=reason, collision_kind=blocked,
                                     assumed_attempt_timestamp=self.last_timestamp,
                                     target_id=command.get('targetId'), target_kind=command.get('targetKind'))

    def diagnostics(self):
        return dict(history_source=self.history_source,
                    initial_cooldowns='assumed_ready_at_launch',
                    clock_basis='elapsed_client_snapshot_ms',
                    cooldowns=dict(global_ready_at_ms=self.cooldowns.global_ready_at_ms,
                                   slots=dict(self.cooldowns.slots), families=dict(self.cooldowns.families)),
                    pending_estimate=deepcopy(self.pending_estimate),
                    latest_estimate=deepcopy(self.latest_estimate))
