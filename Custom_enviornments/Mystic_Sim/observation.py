"""Pure numeric observation encoding with deterministic entity-ID ties."""
from math import hypot
import numpy as np

from .state import Direction


def euclidean_distance(x1, y1, x2, y2):
    return hypot(x2 - x1, y2 - y1)


def observation_distance(x1, y1, x2, y2):
    return abs(x2 - x1) + abs(y2 - y1)


def direction_from_delta(dx, dy):
    if dx == dy == 0:
        return Direction.IDLE
    if abs(dx) > abs(dy):
        return Direction.RIGHT if dx > 0 else Direction.LEFT
    return Direction.DOWN if dy > 0 else Direction.UP


def direction_from_server(code):
    # C# Direction order is Left, Right, Up, Down.
    if type(code) is not int or code not in range(4):
        raise ValueError(f"Invalid server direction {code!r}")
    return (Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN)[code]


def fraction(value, maximum):
    return min(1.0, max(0.0, value / maximum)) if maximum > 0 else 0.0


def nearest_monsters(world):
    p = world.player
    return sorted((m for m in world.monsters.values() if m.alive),
                  key=lambda m: (observation_distance(p.x, p.y, m.x, m.y), m.entity_id))[:5]


def encode_observation(world):
    p = world.player
    result = np.zeros(26, dtype=np.float32)
    result[:6] = (p.x, p.y, int(p.facing), fraction(p.hp, p.max_hp),
                  fraction(p.mp, p.max_mp), world.map.map_id)
    for index, monster in enumerate(nearest_monsters(world)):
        start = 6 + index * 4
        result[start:start + 4] = (
            observation_distance(p.x, p.y, monster.x, monster.y),
            int(direction_from_delta(monster.x - p.x, monster.y - p.y)),
            fraction(monster.hp, monster.max_hp), fraction(monster.mp, monster.max_mp))
    return result
