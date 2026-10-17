"""Read-only episode diagnostics. Never supplied to the policy or reward function."""
from collections import deque

from Custom_enviornments.Mystic_Sim.targeting import in_casting_range


METRICS = ('enemy_in_cast_range_fraction', 'outside_cast_range_fraction',
           'no_living_enemy_fraction', 'seconds_since_damage', 'longest_no_damage_seconds',
           'damage_per_in_range_second', 'successful_move_revisit_fraction',
           'movement_reversal_fraction', 'mean_recent_unique_cells')


class EncounterDiagnostics:
    """32-position window; range sampled BEFORE each action, damage AFTER advance.

    Revisit/reversal metrics are not rewards or proof of a bad loop: kiting can
    legitimately revisit cells. Interpret them alongside damage and range access.
    """
    def __init__(self, world):
        self.positions = deque([(world.player.x, world.player.y)], maxlen=32)
        self.previous_delta = None
        self.last_damage_ms = self.last_time_ms = world.time_ms
        self.steps = self.in_range = self.living_steps = self.moves = 0
        self.revisits = self.reversals = self.unique_sum = 0
        self.in_range_ms = self.in_range_damage = self.longest_gap_ms = 0

    def before(self, world):
        self.origin = (world.player.x, world.player.y)
        living = [m for m in world.monsters.values() if m.alive]
        self.has_living = bool(living)
        # Geometric access, not cast eligibility: ignores cooldown, MP and immunity.
        self.has_range = any(in_casting_range(world.player, m) for m in living)

    def after(self, world, info):
        elapsed = world.time_ms - self.last_time_ms
        self.last_time_ms = world.time_ms
        self.steps += 1
        self.living_steps += int(self.has_living)
        self.in_range += int(self.has_range)
        damage = sum(e['damage'] for e in info['damage_events']
                     if e['attacker_id'] == world.player.entity_id and e['target_id'] != world.player.entity_id)
        self.in_range_ms += elapsed * int(self.has_range)
        self.in_range_damage += damage * int(self.has_range)
        self.longest_gap_ms = max(self.longest_gap_ms, world.time_ms - self.last_damage_ms)
        if damage > 0:
            self.last_damage_ms = world.time_ms
        position = (world.player.x, world.player.y)
        delta = (position[0] - self.origin[0], position[1] - self.origin[1])
        moved = position != self.origin
        if moved:
            self.moves += 1
            self.revisits += int(position in self.positions)
            self.reversals += int(self.previous_delta == (-delta[0], -delta[1]))
        self.previous_delta = delta if moved else None
        self.positions.append(position)
        self.unique_sum += len(set(self.positions))
        return {
            'enemy_in_cast_range_fraction': self.in_range / self.steps,
            'outside_cast_range_fraction': (self.living_steps - self.in_range) / max(1, self.living_steps),
            'no_living_enemy_fraction': (self.steps - self.living_steps) / self.steps,
            'seconds_since_damage': (world.time_ms - self.last_damage_ms) / 1000,
            'longest_no_damage_seconds': self.longest_gap_ms / 1000,
            'damage_per_in_range_second': self.in_range_damage / max(.001, self.in_range_ms / 1000),
            'successful_move_revisit_fraction': self.revisits / max(1, self.moves),
            'movement_reversal_fraction': self.reversals / max(1, self.moves),
            'mean_recent_unique_cells': self.unique_sum / self.steps,
        }
