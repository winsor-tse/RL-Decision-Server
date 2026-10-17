"""Seeded curriculum placement on the unchanged map; no training-stage controller."""
from collections import deque
from dataclasses import replace

from .config import EncounterConfig, ScenarioConfig


def scenario_for_encounter(encounter: EncounterConfig) -> ScenarioConfig:
    """Only population, initial placement, and victory goal differ from the full task."""
    if not isinstance(encounter, EncounterConfig):
        raise ValueError("encounter must be an EncounterConfig")
    base = ScenarioConfig()
    return replace(base, encounter=encounter,
                   reward=replace(base.reward, win_kills=encounter.kill_goal))


def reachable_distances(map_definition, start):
    """Cardinal shortest paths through static terrain, ignoring transient entities."""
    distances = {start: 0}
    pending = deque([start])
    while pending:
        x, y = pending.popleft()
        for cell in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
            if (0 <= cell[0] < map_definition.width and 0 <= cell[1] < map_definition.height
                    and cell not in map_definition.blocked_cells and cell not in distances):
                distances[cell] = distances[(x, y)] + 1
                pending.append(cell)
    return distances


def select_encounter(map_definition, config, rng):
    """Finite search over player starts; sample original NPC boxes without replacement.

    The shipped spawn rectangles are disjoint. Reject overlaps explicitly rather
    than quietly violating occupancy or box capacity on an unsupported map.
    """
    encounter = config.encounter
    boxes = sorted(map_definition.spawn_boxes, key=lambda box: box.object_id)
    seen = set()
    box_cells = []
    for box in boxes:
        if box.template_id != config.innie.template_id:
            raise ValueError(f"Unsupported NPC template {box.template_id}")
        cells = [(x, y) for y in range(box.y, box.y + box.height)
                 for x in range(box.x, box.x + box.width)
                 if 0 <= x < map_definition.width and 0 <= y < map_definition.height
                 and (x, y) not in map_definition.blocked_cells]
        if seen.intersection(cells):
            raise ValueError("Encounter selection requires nonoverlapping spawn boxes")
        seen.update(cells)
        box_cells.append(cells)
    starts = [(x, y) for y in range(config.player_spawn_y[0], config.player_spawn_y[1] + 1)
              for x in range(config.player_spawn_x[0], config.player_spawn_x[1] + 1)
              if (x, y) not in map_definition.blocked_cells]
    # Trying each start at most once gives impossible configurations a hard bound.
    for index in rng.permutation(len(starts)):
        start = starts[int(index)]
        distances = reachable_distances(map_definition, start)
        low, high = encounter.path_distance
        candidates = [[cell for cell in cells if low <= distances.get(cell, -1) <= high]
                      for cells in box_cells]
        slots = [(i, member) for i, box in enumerate(boxes)
                 for member in range(min(box.quantity, len(candidates[i])))]
        if len(slots) < encounter.enemy_count:
            continue
        chosen = sorted(int(i) for i in rng.choice(len(slots), encounter.enemy_count, replace=False))
        placements = []
        for slot in chosen:
            box_index, member = slots[slot]
            cells = candidates[box_index]
            cell = cells.pop(int(rng.integers(len(cells))))
            placements.append((boxes[box_index], member, cell))
        return start, placements
    raise ValueError(f"Encounter {encounter.stage_id!r} cannot place {encounter.enemy_count} "
                     f"enemies at path distance {encounter.path_distance} from any legal player start")
