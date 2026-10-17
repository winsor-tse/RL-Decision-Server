"""Training-only whole-area placement; combat and original checkpoint contracts stay intact."""
from dataclasses import dataclass, replace

import gymnasium as gym

from Custom_enviornments.Mystic_Sim.config import EncounterConfig
from Custom_enviornments.Mystic_Sim.encounters import reachable_distances
from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
from Custom_enviornments.Mystic_Sim.scenarios import sample_unoccupied


@dataclass(frozen=True, slots=True)
class NearbyAreasEncounter(EncounterConfig):
    placement: str = 'nearest_spawn_areas_v1'
    spawn_area_count: int = 10

    def __post_init__(self):
        EncounterConfig.__post_init__(self)
        if self.placement not in ('nearest_spawn_areas_v1', 'nearest_spawn_areas_single_v1',
                                  'random_spawn_areas_single_v1', 'all_spawn_areas_density_v1'):
            raise ValueError('Unsupported whole-area placement version')
        if type(self.spawn_area_count) is not int or not 1 <= self.spawn_area_count <= 40:
            raise ValueError('spawn_area_count must be in 1..40')
        per_area = 2 if self.placement == 'nearest_spawn_areas_v1' else 1
        if self.placement == 'all_spawn_areas_density_v1':
            if self.spawn_area_count != 40 or not 40 <= self.enemy_count <= 80:
                raise ValueError('All-area density needs 40 areas and 40..80 enemies')
        elif self.enemy_count != per_area * self.spawn_area_count:
            raise ValueError(f'This placement requires {per_area} Innies per selected area')
        if self.path_distance != (1, 10000):
            raise ValueError('Whole-area placement must not restrict individual spawn cells by distance')


def ranked_spawn_areas(map_definition, start):
    """Rank by shortest terrain path to a legal area cell; break ties by object ID.

    Ranking does not apply Y training penalties or entity occupancy. The NPCs
    spawn normally throughout each chosen original rectangle, not just its edge.
    """
    distances = reachable_distances(map_definition, start)
    ranked = []
    for box in map_definition.spawn_boxes:
        cells = [(x, y) for y in range(box.y, box.y + box.height)
                 for x in range(box.x, box.x + box.width)
                 if (x, y) != start and (x, y) in distances]
        if len(cells) >= box.quantity:
            ranked.append((min(distances[cell] for cell in cells), box.object_id, box))
    ranked.sort(key=lambda row: row[:2])
    return tuple(row[2] for row in ranked)


def nearest_spawn_areas(map_definition, start, count):
    ranked = ranked_spawn_areas(map_definition, start)
    if len(ranked) < count:
        raise ValueError('Not enough reachable spawn areas for this encounter')
    return ranked[:count]


def random_spawn_areas(map_definition, start, count, rng):
    """Uniformly sample distinct reachable original areas, without distance bias."""
    eligible = sorted(ranked_spawn_areas(map_definition, start), key=lambda box: box.object_id)
    if len(eligible) < count:
        raise ValueError('Not enough reachable spawn areas for this encounter')
    indices = sorted(int(i) for i in rng.choice(len(eligible), count, replace=False))
    return tuple(eligible[i] for i in indices)


class NearbyAreasEnv(MysticSimEnv):
    def reset(self, *, seed=None, options=None):
        # Seed once, then let the existing reset construct all state/schedules and
        # normal NPC stats. All temporary configuration is local to this instance.
        gym.Env.reset(self, seed=seed)
        original_map, original_config = self.map_definition, self.config
        encounter = original_config.encounter
        xmin, xmax = original_config.player_spawn_x
        ymin, ymax = original_config.player_spawn_y
        start = sample_unoccupied(self.np_random, xmin, ymin, xmax-xmin+1,
                                  ymax-ymin+1, {}, original_map.blocked_cells)
        placement_map = original_map
        if encounter.placement != 'nearest_spawn_areas_v1':
            # Keep every original rectangle/ID; only its initial population changes.
            placement_map = replace(original_map, spawn_boxes=tuple(
                replace(box, quantity=1) for box in original_map.spawn_boxes))
        boxes = (random_spawn_areas(placement_map, start, encounter.spawn_area_count, self.np_random)
                 if encounter.placement == 'random_spawn_areas_single_v1' else
                 nearest_spawn_areas(placement_map, start, encounter.spawn_area_count))
        if encounter.placement == 'all_spawn_areas_density_v1':
            extras = set(int(i) for i in self.np_random.choice(
                len(boxes), encounter.enemy_count-len(boxes), replace=False))
            boxes = tuple(replace(box, quantity=1+int(i in extras)) for i,box in enumerate(boxes))
        if sum(box.quantity for box in boxes) != encounter.enemy_count:
            raise ValueError('Selected original spawn quantities do not match enemy_count')
        try:
            self.map_definition = replace(original_map, spawn_boxes=boxes)
            self.config = replace(original_config, encounter=None,
                                  player_spawn_x=(start[0], start[0]),
                                  player_spawn_y=(start[1], start[1]))
            obs, info = super().reset(options=options)
        finally:
            self.map_definition, self.config = original_map, original_config
        # Full terrain/map metadata is retained; each monster still owns its
        # selected original spawn box, including normal 50-second respawns.
        self.world.map = original_map
        self.engine.config = original_config
        info['placement'] = encounter.placement
        info['selected_spawn_area_ids'] = [box.object_id for box in boxes]
        return obs, info
