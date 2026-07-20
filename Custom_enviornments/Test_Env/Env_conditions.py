from pathlib import Path
import sys
from typing import Sequence

import numpy as np

try:
    from Custom_enviornments.Load_env_config import load_env_config
except ModuleNotFoundError:
    env_root = Path(__file__).resolve().parents[1]
    if str(env_root) not in sys.path:
        sys.path.append(str(env_root))
    from Load_env_config import load_env_config


# This file is specific to this directory or class
# New class-specific envs should define their own rewards, termination,
# and truncation rules in their own Env_conditions.py.
CONFIG = load_env_config()
OBS_PLAYER_SIZE = int(CONFIG["OBS_PLAYER_SIZE"])
OBS_ENEMY_SIZE = int(CONFIG["OBS_ENEMY_SIZE"])
MAX_ENEMIES = int(CONFIG["MAX_ENEMIES"])
OBS_SIZE = int(CONFIG["OBS_SIZE"])
MAX_EPISODE_STEPS = int(CONFIG["MAX_EPISODE_STEPS"])

DIRECTION_MAP = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
    "idle": 4,
}

INV_DIRECTION_MAP = {value: key for key, value in DIRECTION_MAP.items()}


def safe_pct(value: float, max_value: float) -> float:
    if max_value is None or max_value <= 0:
        return 0.0
    return float(np.clip(value / max_value, 0.0, 1.0))


def distance_from_player(player: dict, entity: dict) -> float:
    if entity.get("distance") is not None:
        return float(entity["distance"])

    dx = entity.get("mapX", 0) - player.get("mapX", 0)
    dy = entity.get("mapY", 0) - player.get("mapY", 0)
    return float(abs(dx) + abs(dy))


def direction_from_player(player: dict, entity: dict) -> int:
    dx = entity.get("mapX", 0) - player.get("mapX", 0)
    dy = entity.get("mapY", 0) - player.get("mapY", 0)

    if dx == 0 and dy == 0:
        return DIRECTION_MAP["idle"]

    if abs(dx) > abs(dy):
        return DIRECTION_MAP["right"] if dx > 0 else DIRECTION_MAP["left"]

    return DIRECTION_MAP["down"] if dy > 0 else DIRECTION_MAP["up"]


def enemy_block(obs: Sequence[float], enemy_index: int):
    start = OBS_PLAYER_SIZE + enemy_index * OBS_ENEMY_SIZE
    return {
        "distance": float(obs[start]),
        "direction": int(obs[start + 1]),
        "hp_pct": float(obs[start + 2]),
        "mp_pct": float(obs[start + 3]),
    }


def parse_observation(data: dict, obs_size: int = OBS_SIZE):
    """
    Builds the fixed Yugen Saga observation vector used by the RL agent.

    Layout:
    - player map X, map Y, direction, HP pct, MP pct
    - nearest enemy distance, direction, HP pct, MP pct
    - second nearest enemy distance, direction, HP pct, MP pct
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
        entity
        for entity in entities
        if entity.get("type") == "monster" and not entity.get("isCurrentPlayer", False)
    ]
    monsters.sort(key=lambda monster: distance_from_player(player, monster))

    for monster in monsters[:MAX_ENEMIES]:
        obs.extend(
            [
                distance_from_player(player, monster),
                float(direction_from_player(player, monster)),
                safe_pct(monster.get("hp", 0), monster.get("maxHp", 0)),
                safe_pct(monster.get("mp", 0), monster.get("maxMp", 0)),
            ]
        )

    while len(obs) < obs_size:
        obs.extend([0.0] * OBS_ENEMY_SIZE)

    return np.array(obs[:obs_size], dtype=np.float32)


def get_reward(obs, action, prev_obs):
    reward = 0.0
    player_hp_pct = float(obs[3])
    nearest_enemy = enemy_block(obs, 0)

    if player_hp_pct < 0.25:
        reward -= 0.50
    elif player_hp_pct < 0.50:
        reward -= 0.15

    if prev_obs is None or not np.any(prev_obs):
        return float(reward)

    if obs[1] < 30:
        reward -= 0.1 * (30- obs[1])

    prev_player_hp_pct = float(prev_obs[3])
    prev_nearest_enemy = enemy_block(prev_obs, 0)

    hp_lost = prev_player_hp_pct - player_hp_pct
    if hp_lost > 0:
        reward -= 2.0 * hp_lost

    enemy_hp_lost = prev_nearest_enemy["hp_pct"] - nearest_enemy["hp_pct"]
    if enemy_hp_lost > 0:
        reward += 25.0 * enemy_hp_lost

    return reward


def get_episode_outcome(obs, prev_obs):
    if prev_obs is None or not np.any(prev_obs):
        return None

    player_hp_pct = float(obs[3])
    if player_hp_pct <= 0.0:
        return "loss"

    nearest_enemy = enemy_block(obs, 0)
    prev_nearest_enemy = enemy_block(prev_obs, 0)
    enemy_was_alive = prev_nearest_enemy["hp_pct"] > 0
    enemy_is_dead = nearest_enemy["hp_pct"] <= 0
    if enemy_was_alive and enemy_is_dead:
        return "win"

    return None


def get_termination(obs, prev_obs):
    return get_episode_outcome(obs, prev_obs) is not None


def get_truncated(obs, prev_obs, current_step):
    return (current_step >= MAX_EPISODE_STEPS or obs[1] < 25)
