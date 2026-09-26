"""Literal NPC.cs cardinal candidate scoring and weighted pursuit branches."""
from .state import Direction

# C# order is Left, Right, Up, Down, not the observation enum order.
SERVER_DIRECTIONS = (Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN)
OFFSETS = {Direction.LEFT: (-1, 0), Direction.RIGHT: (1, 0),
           Direction.UP: (0, -1), Direction.DOWN: (0, 1)}


def legal_move(world, x, y):
    return (0 <= x < world.map.width and 0 <= y < world.map.height
            and (x, y) not in world.occupancy
            and (not world.terrain_collision or (x, y) not in world.map.blocked_cells))


def next_step_basic(world, npc, x, y, rng):
    dx, dy = x - npc.x, y - npc.y
    for direction, offset in OFFSETS.items():
        if (dx, dy) == offset:
            return direction
    current = abs(dx) + abs(dy)
    shortest, closest = current, Direction.UP
    distances = {}
    for direction in SERVER_DIRECTIONS:
        ox, oy = OFFSETS[direction]
        nx, ny = npc.x + ox, npc.y + oy
        distance = abs(x - nx) + abs(y - ny) if legal_move(world, nx, ny) else 2147483647
        distances[direction] = distance
        if distance < shortest:
            shortest, closest = distance, direction
    left, right, up, down = (distances[d] for d in SERVER_DIRECTIONS)
    coin = rng.roll(0, 1, "path_coin") == 1
    if shortest == current:
        if dx == 0 and (left == current + 1 or right == current + 1):
            closest = (Direction.LEFT if coin else Direction.RIGHT) if left == right else (
                Direction.LEFT if left == current + 1 else Direction.RIGHT)
        elif dy == 0 and (up == current + 1 or down == current + 1):
            closest = (Direction.UP if coin else Direction.DOWN) if up == down else (
                Direction.UP if up == current + 1 else Direction.DOWN)
        else:
            return SERVER_DIRECTIONS[rng.roll(0, 3, "path_fallback")]
    chance = (abs(dx) + 1)**2 / ((abs(dx) + 1)**2 + (abs(dy) + 1)**2)
    horizontal = rng.chance(chance, "path_axis")
    # Independent ifs intentionally preserve source overwrite ordering.
    for vertical, lateral in ((Direction.UP, Direction.RIGHT), (Direction.UP, Direction.LEFT),
                              (Direction.DOWN, Direction.RIGHT), (Direction.DOWN, Direction.LEFT)):
        if distances[vertical] == shortest and distances[lateral] == shortest:
            closest = lateral if horizontal else vertical
    return closest


def outside_spawn_area(npc):
    box = npc.spawn_box
    # Source OutsideSpawnArea uses > X+W / Y+H (inclusive upper return boundary),
    # whereas spawn sampling uses the half-open tile rectangle.
    return (box.fixed and (npc.x, npc.y) != (npc.spawn_x, npc.spawn_y)
            or npc.x < box.x or npc.x > box.x + box.width
            or npc.y < box.y or npc.y > box.y + box.height)
