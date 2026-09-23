"""Pure spell calculations. Crit is an input; no world mutation or RNG occurs."""
from dataclasses import dataclass
from math import sqrt
from .combat import stat_factor, crit_multiplier, apply_aoe


@dataclass(frozen=True, slots=True)
class Allocation:
    target_id: int
    raw_damage: int  # Input to magic mitigation, before Acid's distance falloff.
    falloff: float = 1.0
    tick_damage: int = 0  # Snapshot damage: ticks bypass crit/block/AC.


@dataclass(frozen=True, slots=True)
class Calculation:
    base_damage: int
    total_damage: int
    crit: bool
    radius: float
    raw_radius: float
    allocations: tuple[Allocation, ...]
    mp_consumption: int
    hp_consumption: int


def mystic_range(stats, level, minimum, maximum):
    e = .5 * level ** 2 + 8 * level + 27.5
    return min(maximum, minimum + (2.57 / 15) * max(0, sqrt(stats.dexterity * stats.strength) - sqrt(e)) ** .75)


def radii(spell, stats):
    if spell.slot == 1:
        return 0.0, 0.0
    level = spell.level or spell.cast_level or stats.level
    if spell.slot == 2:
        raw = max(spell.min_radius, mystic_range(stats, level, spell.min_radius, spell.max_radius + 25))
        return raw, min(spell.max_radius, raw)
    raw = max(spell.min_radius, mystic_range(stats, level, spell.min_radius, spell.max_radius + 1.5) - 1.5)
    return raw, min(spell.max_radius, raw)


def tempest_allocation(total, radius, full_count, partial_count):
    count = full_count + partial_count
    if count == 0:
        return 0, 0
    scaled = apply_aoe(total, count)
    weight = (radius - (.5 if radius < 1 else 1)) / .5
    full = full_count * apply_aoe(total, full_count) if full_count else 0
    budget = min(scaled * count, int(full + (scaled * count - full) * weight))
    total_weight = full_count + weight * partial_count
    # Degenerate boundary with zero-weight partial targets has no damage budget.
    if total_weight <= 0:
        return 0, 0
    return int(budget / total_weight), int(budget * weight / total_weight)


def calculate(spell, stats, post_cost_mp, post_cost_hp, target_distances, *,
              caster_distance=0.0, partial_ids=(), crit=False, spell_multiplier=0.0):
    """target_distances is an ordered (ID, distance-from-center) tuple."""
    intelligence, strength = stat_factor(stats.intelligence), stat_factor(stats.strength)
    mp = post_cost_mp
    raw_radius, radius = radii(spell, stats)
    if spell.slot == 1:
        base = spell.base_damage + 6.6 * ((mp + 22.9) * spell.mana_factor) ** spell.damage_percent
        base *= (max(1, strength / 7.5) * max(1, intelligence / 11.8)) ** .5
        base *= min(1, (35 / intelligence) ** .7) * min(1, (25 / strength) ** .2)
    elif spell.slot == 2:
        base = 777.43 + 3.5 * (mp * spell.mana_factor) ** spell.damage_percent
        base *= max(1, intelligence / 16) ** .4
        base *= max(1, intelligence / 11.8) * max(1, intelligence / 12.58) / (max(1, intelligence / 17.5) * max(1, intelligence / 24))
        base *= min(1, (16 / intelligence) ** .5) * max(1, (intelligence / 25) ** .7) * min(1, (80 / intelligence) ** .2)
        base *= min(1, (21 / intelligence) ** .15) * min(1, (60 / intelligence) ** .15)
    else:
        falloff = min(1, (caster_distance + 1) / max(1, 1.5 * caster_distance - 1))
        base = (1600 + 3.5 * (mp * spell.mana_factor) ** spell.damage_percent) * falloff
        base *= max(1, intelligence ** .7 / 4.9) * max(1, (intelligence / 13) ** .5)
        base *= max(1, strength / 7.3) / (max(1, (intelligence / 20) ** .5) * max(1, intelligence / 9.65))
        base *= min(1, (13 / strength) ** .65)
        base *= max(1, (intelligence / 20) ** .25) * min(1, (50 / intelligence) ** .2)
    base = int(base)
    total = int(base * intelligence * (crit_multiplier(stats, True) if crit else 1) * (1 + spell_multiplier))
    count = len(target_distances)
    allocations = []
    if spell.slot == 1:
        allocations = [Allocation(i, total) for i, _ in target_distances]
    elif spell.slot == 2 and count:
        initial = apply_aoe(int(total * spell.initial_damage_percent), count)
        tick = apply_aoe(int(total * spell.tick_percent), count)
        allocations = [Allocation(i, initial, min(1, 1.05 * raw_radius / (raw_radius + d - 1)), tick)
                       for i, d in target_distances]
    elif spell.slot == 3 and total > 0 and count:
        full, partial = tempest_allocation(total, radius, count - len(partial_ids), len(partial_ids))
        allocations = [Allocation(i, partial if i in partial_ids else full) for i, _ in target_distances]
    consume = spell.slot != 3 or (total > 0 and count > 0)
    return Calculation(base, total, crit, radius, raw_radius, tuple(allocations),
                       min(mp, int(round(mp * spell.mana_consumption))) if consume else 0,
                       min(max(0, post_cost_hp - 1), int(round(post_cost_hp * spell.vita_consumption))) if consume else 0)
