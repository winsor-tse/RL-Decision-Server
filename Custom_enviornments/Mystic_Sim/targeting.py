"""Euclidean spell selection, independent of observation ranking."""
from math import hypot
from .combat import can_hit


def distance(a, b):
    return hypot(a.x - b.x, a.y - b.y)


def in_casting_range(caster, target):
    """Inclusive rectangular targeting window, independent of AoE distance."""
    return abs(caster.x - target.x) <= 16 and abs(caster.y - target.y) <= 10


def select_target(world, spell):
    caster = world.player
    candidates = [m for m in world.monsters.values()
                  if can_hit(caster, m, magic=True) and in_casting_range(caster, m)]
    if candidates:
        return min(candidates, key=lambda m: (distance(caster, m), m.entity_id))
    return caster if spell.target_mode == "targetedOrSelf" else None


def area_targets(world, center, radius, *, inner=None):
    # Map enumeration implementation is absent; stable entity-ID order is explicit.
    return tuple(m for m in sorted(world.monsters.values(), key=lambda m: m.entity_id)
                 if can_hit(world.player, m, magic=True)
                 and distance(center, m) <= radius
                 and (inner is None or distance(center, m) > inner))
