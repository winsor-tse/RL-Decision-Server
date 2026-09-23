"""Runtime state owns all mutable episode data; map definitions are immutable."""
from dataclasses import dataclass, field
from enum import IntEnum
import heapq

from .config import PlayerConfig, InnieConfig


class Direction(IntEnum):
    """Observation codes, deliberately distinct from server direction codes."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    IDLE = 4


@dataclass(frozen=True, slots=True)
class SpawnBox:
    object_id: int
    template_id: int
    x: int
    y: int
    width: int
    height: int
    quantity: int
    fixed: bool = False

    def contains(self, x, y):
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height


@dataclass(frozen=True, slots=True)
class MapDefinition:
    map_id: int
    width: int
    height: int
    tile_width: int
    tile_height: int
    properties: tuple[tuple[str, object], ...]
    spawn_boxes: tuple[SpawnBox, ...]
    excluded_npcs: tuple[SpawnBox, ...]
    blocked_cells: frozenset[tuple[int, int]]

    @property
    def parsed_balance_cap(self):
        return dict(self.properties)["balanceCap"]


@dataclass(slots=True)
class CooldownState:
    slots: dict[int, int] = field(default_factory=dict)
    families: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class PlayerState:
    entity_id: int
    x: int
    y: int
    hp: int
    mp: int
    stats: PlayerConfig
    facing: Direction = Direction.UP
    cooldowns: CooldownState = field(default_factory=CooldownState)
    next_regen_ms: int = 2000
    next_attack_ms: int = 0
    magic_immune: bool = False
    death_recorded: bool = False

    @property
    def max_hp(self):
        return self.stats.max_hp

    @property
    def max_mp(self):
        return self.stats.max_mp

    @property
    def class_spec(self):
        return self.stats.class_spec

    @property
    def alive(self):
        return self.hp > 0


@dataclass(slots=True)
class MonsterState:
    entity_id: int
    x: int
    y: int
    hp: int
    mp: int
    stats: InnieConfig
    spawn_box: SpawnBox
    spawn_member: int
    move_interval_ms: int
    next_move_ms: int
    next_attack_ms: int
    next_aggro_ms: int
    facing: Direction = Direction.UP
    aggro_target: int | None = None
    respawn_at_ms: int | None = None
    life_generation: int = 0
    spawn_x: int = 0
    spawn_y: int = 0
    last_aggro_update_ms: int = 0
    magic_immune: bool = False
    death_recorded: bool = False

    @property
    def max_hp(self):
        return self.stats.max_hp

    @property
    def max_mp(self):
        return self.stats.max_mp

    @property
    def alive(self):
        return self.hp > 0


@dataclass(slots=True)
class TimedEffect:
    effect_id: str
    source_id: int
    target_id: int
    next_tick_ms: int
    expires_at_ms: int
    interval_ms: int
    generation: int = 0


@dataclass(frozen=True, slots=True)
class DamageEvent:
    time_ms: int
    attacker_id: int
    target_id: int
    raw_damage: int
    damage: int
    hp_before: int
    hp_after: int
    damage_type: str
    crit: bool = False
    dodged: bool = False
    blocked: bool = False


@dataclass(frozen=True, slots=True)
class DeathEvent:
    time_ms: int
    entity_id: int
    killer_id: int


@dataclass(frozen=True, slots=True)
class RespawnEvent:
    time_ms: int
    entity_id: int
    x: int
    y: int
    generation: int
    move_interval_ms: int


@dataclass(frozen=True, slots=True, order=True)
class ScheduledEvent:
    due_ms: int
    sequence: int
    kind: str = field(compare=False)
    entity_id: int = field(compare=False)
    generation: int = field(default=0, compare=False)


@dataclass(slots=True)
class WorldState:
    map: MapDefinition
    player: PlayerState
    monsters: dict[int, MonsterState] = field(default_factory=dict)
    occupancy: dict[tuple[int, int], int] = field(default_factory=dict)
    effects: list[TimedEffect] = field(default_factory=list)
    events: list[ScheduledEvent] = field(default_factory=list)
    time_ms: int = 0
    step_count: int = 0
    kills: int = 0
    next_event_sequence: int = 0

    def enqueue(self, due_ms, kind, entity_id, generation=0):
        """Return a cancellation token; order equal-time events by insertion."""
        if type(due_ms) is not int or due_ms < self.time_ms:
            raise ValueError("Event deadline must be integer milliseconds at or after current time")
        token = self.next_event_sequence
        heapq.heappush(self.events, ScheduledEvent(due_ms, token, kind, entity_id, generation))
        self.next_event_sequence += 1
        return token
