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

def enemy_block(obs: Sequence[float], enemy_index: int):
    """
    enemy_index:
    - 0 for nearest enemy
    - 1 for second nearest enemy
    """
    start = OBS_PLAYER_SIZE + enemy_index * OBS_ENEMY_SIZE

    return {
        "distance": float(obs[start]),
        "direction": int(obs[start + 1]),
        "hp_pct": float(obs[start + 2]),
        "mp_pct": float(obs[start + 3]),
    }


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


#TODO: Prev Observations can create bugs when not cleared during reset??

#Termination: "The game is over forever, so there is no future."
#Future Value: Set to 0.
def get_termination(obs, prev_obs):
    if prev_obs is None or not np.any(prev_obs):
        return False
    player_hp_pct = float(obs[3])
    if player_hp_pct <= 0.0:
        return True
    # Enemy killed
    nearest_enemy = enemy_block(obs, 0)
    prev_nearest_enemy = enemy_block(prev_obs, 0)
    enemy_was_alive = prev_nearest_enemy["hp_pct"] > 0
    enemy_is_dead = nearest_enemy["hp_pct"] <= 0
    if enemy_was_alive and enemy_is_dead:
        return True
    return False

# Truncation: "The game was paused by a timer, so we must guess what the future would have looked like."
# Future Value: Kept (The neural network guesses how many points the agent would have gotten if the timer hadn't stopped the game)
def get_truncated(obs, prev_obs, current_step):
    #simple each epoch has 100 steps
    if current_step == 100:
        return True
    return False


#simple implementation of rewards
#Implement based on the given OBS Space
#Attacking give positive rewards
#Closer Distance from Enemy will give positive rewards, vice versa
#Killing enemy will give positive rewards (enemy hp turns to 0)
#Hp percentage below a certain limit is negative rewards, etc.
#Rest can be zero reward...
def get_reward(obs, actions, prev_obs):
    reward = 0.0
    player_hp_pct = float(obs[3])
    nearest_enemy = enemy_block(obs, 0)
    if player_hp_pct < 0.25:
        reward -= 0.50
    elif player_hp_pct < 0.50:
        reward -= 0.15
        # If no previous observation exists, return stateless reward only
    if prev_obs is None or not np.any(prev_obs):
        return float(reward)
    prev_player_hp_pct = float(prev_obs[3])
    prev_nearest_enemy = enemy_block(prev_obs, 0)
    # Player HP loss penalty
    hp_lost = prev_player_hp_pct - player_hp_pct
    if hp_lost > 0:
        reward -= 2.0 * hp_lost
    # Enemy damage reward
    enemy_hp_lost = prev_nearest_enemy["hp_pct"] - nearest_enemy["hp_pct"]
    if enemy_hp_lost > 0:
        reward += 3.0 * enemy_hp_lost
    # Enemy kill reward
    #TODO: this needs to be changed we are getting positive reward when tp to a certain map
    enemy_was_alive = prev_nearest_enemy["hp_pct"] > 0
    enemy_is_dead = nearest_enemy["hp_pct"] <= 0
    if enemy_was_alive and enemy_is_dead:
        reward += 5.0
    return reward