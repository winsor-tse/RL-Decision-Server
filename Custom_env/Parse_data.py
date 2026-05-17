from typing import List
import numpy as np
DIRECTION_MAP = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
    "idle": 4  # if needed
}

INV_DIRECTION_MAP = {v: k for k, v in DIRECTION_MAP.items()}
OBS_PLAYER_SIZE = 5
OBS_ENEMY_SIZE = 4
MAX_ENEMIES = 2
OBS_SIZE = OBS_PLAYER_SIZE + OBS_ENEMY_SIZE * MAX_ENEMIES  # 13

# Observation layout:
# [
#   player_mapX,
#   player_mapY,
#   player_direction,
#   player_hp_pct,
#   player_mp_pct,
#
#   enemy1_distance,
#   enemy1_direction_from_player,
#   enemy1_hp_pct,
#   enemy1_mp_pct,
#
#   enemy2_distance,
#   enemy2_direction_from_player,
#   enemy2_hp_pct,
#   enemy2_mp_pct,
# ]

def safe_pct(value: float, max_value: float) -> float:
    if max_value is None or max_value <= 0:
        return 0.0
    return float(np.clip(value / max_value, 0.0, 1.0))


def distance_from_player(player: dict, entity: dict) -> float:
    if "distance" in entity and entity["distance"] is not None:
        return float(entity["distance"])

    dx = entity.get("mapX", 0) - player.get("mapX", 0)
    dy = entity.get("mapY", 0) - player.get("mapY", 0)

    # Grid-world style Manhattan distance
    return float(abs(dx) + abs(dy))


def direction_from_player(player: dict, entity: dict) -> int:
    """
    Returns direction of the entity relative to the player.

    Assumption:
    - Smaller mapY means up.
    - Larger mapY means down.
    - Smaller mapX means left.
    - Larger mapX means right.
    """
    dx = entity.get("mapX", 0) - player.get("mapX", 0)
    dy = entity.get("mapY", 0) - player.get("mapY", 0)

    if dx == 0 and dy == 0:
        return DIRECTION_MAP["idle"]

    if abs(dx) > abs(dy):
        return DIRECTION_MAP["right"] if dx > 0 else DIRECTION_MAP["left"]
    else:
        return DIRECTION_MAP["down"] if dy > 0 else DIRECTION_MAP["up"]


def parse_observation(data: dict, obs_size) -> List[float]:
    """
    Converts the game tick/world state into a fixed 13-float observation.
    Supports either:
    - data["worldState"]
    - or data directly as the world state
    """
    world = data.get("worldState", data)
    player = world["player"]
    entities = world.get("entities", [])
    player_direction = DIRECTION_MAP.get(
        player.get("direction", "idle"),
        DIRECTION_MAP["idle"],
    )
    obs = [
        float(player.get("mapX", 0)),
        float(player.get("mapY", 0)),
        float(player_direction),
        safe_pct(player.get("hp", 0), player.get("maxHp", 0)),
        safe_pct(player.get("mp", 0), player.get("maxMp", 0)),
    ]
    monsters = [
        e for e in entities
        if e.get("type") == "monster" and not e.get("isCurrentPlayer", False)
    ]
    monsters.sort(key=lambda m: distance_from_player(player, m))
    for monster in monsters[:MAX_ENEMIES]:
        obs.extend(
            [
                distance_from_player(player, monster),
                float(direction_from_player(player, monster)),
                safe_pct(monster.get("hp", 0), monster.get("maxHp", 0)),
                safe_pct(monster.get("mp", 0), monster.get("maxMp", 0)),
            ]
        )
    # Pad missing enemy sensor slots with zeros
    while len(obs) < obs_size:
        obs.extend([0.0, 0.0, 0.0, 0.0])
    return np.array(obs, dtype=np.float32)

def get_reward(obs, actions):
    #simple implementation of rewards
    #Implement based on the given OBS Space
    #Attacking give positive rewards
    #Closer Distance from Enemy will give positive rewards, vice versa
    #Killing enemy will give positive rewards (enemy hp turns to 0)
    #Hp percentage below a certain limit is negative rewards, etc.
    #Rest can be zero reward...
    return None