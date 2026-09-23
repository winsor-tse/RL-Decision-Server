"""Seeded reset construction, with explicit RNG consumption order."""
from .state import MonsterState, PlayerState, WorldState


def sample_unoccupied(rng, x, y, width, height, occupancy):
    """Draw X then Y until free. Full rectangles fail before any random draw."""
    if not any((cx, cy) not in occupancy for cy in range(y, y + height)
               for cx in range(x, x + width)):
        raise ValueError(f"Spawn rectangle ({x}, {y}, {width}, {height}) has no free cell")
    while True:
        cell = (int(rng.integers(x, x + width)), int(rng.integers(y, y + height)))
        if cell not in occupancy:
            return cell


def build_scenario(map_definition, config, rng):
    if (map_definition.map_id, map_definition.width, map_definition.height) != (
            config.map_id, config.width, config.height):
        raise ValueError("Map dimensions/ID do not match scenario configuration")
    occupancy = {}
    xmin, xmax = config.player_spawn_x
    ymin, ymax = config.player_spawn_y
    x, y = sample_unoccupied(rng, xmin, ymin, xmax - xmin + 1, ymax - ymin + 1, occupancy)
    player = PlayerState(1, x, y, config.player.max_hp, config.player.max_mp,
                         config.player, next_regen_ms=config.timing.regen_ms)
    occupancy[(x, y)] = player.entity_id
    world = WorldState(map_definition, player, occupancy=occupancy)
    world.enqueue(player.next_regen_ms, "regeneration", player.entity_id)
    entity_id = 2
    for box in sorted(map_definition.spawn_boxes, key=lambda box: box.object_id):
        if box.template_id != config.innie.template_id:
            raise ValueError(f"Unsupported NPC template {box.template_id}")
        for member in range(box.quantity):
            x, y = sample_unoccupied(rng, box.x, box.y, box.width, box.height, occupancy)
            # Exactly one movement draw after placement for each new life.
            movement_ms = int(rng.integers(config.innie.move_ms[0], config.innie.move_ms[1] + 1))
            monster = MonsterState(entity_id, x, y, config.innie.max_hp, config.innie.max_mp,
                                   config.innie, box, member, movement_ms, movement_ms,
                                   config.innie.attack_ms, config.innie.aggro_check_ms,
                                   spawn_x=x, spawn_y=y)
            world.monsters[entity_id] = monster
            occupancy[(x, y)] = entity_id
            # One NPC-update event considers all three deadlines in Phase 2.
            world.enqueue(min(monster.next_move_ms, monster.next_attack_ms,
                              monster.next_aggro_ms), "npc_update", entity_id,
                          monster.life_generation)
            entity_id += 1
    world.events.sort()
    return world
