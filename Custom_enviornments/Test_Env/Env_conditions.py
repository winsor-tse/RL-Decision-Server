import json
from pathlib import Path
from typing import Sequence

import numpy as np

from Custom_enviornments.Load_env_config import load_env_config


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
WORLD_STATE_DIR = Path(__file__).resolve().parents[2] / "runs" / "world_state"
MONSTER_IDS_FILE = WORLD_STATE_DIR / "MonsterIds.txt"
WORLD_STATE_FILE = WORLD_STATE_DIR / "world_state.txt"


def ensure_world_state_dir() -> None:
    WORLD_STATE_DIR.mkdir(parents=True, exist_ok=True)


def save_world_state(world: dict) -> None:
    """Replace world_state.txt with the latest complete world state."""
    ensure_world_state_dir()
    WORLD_STATE_FILE.write_text(
        json.dumps(world, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_monster_ids(monsters: Sequence[dict]) -> None:
    """Save observed monster IDs and their latest HP percentages."""
    ensure_world_state_dir()
    monster_hp_percentages = {}

    if MONSTER_IDS_FILE.exists():
        for line in MONSTER_IDS_FILE.read_text(encoding="utf-8").splitlines():
            try:
                monster_id, hp_pct = line.split(":", maxsplit=1)
                monster_hp_percentages[int(monster_id.strip())] = float(hp_pct.strip())
            except (TypeError, ValueError):
                continue

    for monster in monsters:
        try:
            monster_id = int(monster["id"])
        except (KeyError, TypeError, ValueError):
            continue
        monster_hp_percentages[monster_id] = safe_pct(
            monster.get("hp", 0),
            monster.get("maxHp", 0),
        )

    MONSTER_IDS_FILE.write_text(
        "".join(
            f"{monster_id}: {monster_hp_percentages[monster_id]:.6f}\n"
            for monster_id in sorted(monster_hp_percentages)
        ),
        encoding="utf-8",
    )


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
    - player mapID, map X, map Y, direction, HP pct, MP pct
    - nearest enemy distance, direction, HP pct, MP pct
    - second nearest enemy distance, direction, HP pct, MP pct
    """
    world = data.get("worldState", data)
    save_world_state(world)
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
        float(player.get("mapID", 0)[3:]),
    ]

    monsters = [
        entity
        for entity in entities
        if entity.get("type") == "monster" and not entity.get("isCurrentPlayer", False)
    ]
    monsters.sort(key=lambda monster: distance_from_player(player, monster))
    save_monster_ids(monsters)

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


def get_reward_components(obs, action, prev_obs) -> dict[str, float]:
    """Return signed reward contributions for metrics and reward calculation."""
    components = {
        "health_state": 0.0,
        "positioning": 0.0,
        "damage_taken": 0.0,
        "damage_dealt": 0.0,
        "terminal": 0.0,
    }
    player_hp_pct = float(obs[3])
    nearest_enemy = enemy_block(obs, 0)

    if action <= 3 and obs[0] == prev_obs[0] and obs[1] == prev_obs[1]:
        components["positioning"] -= 1000

    if player_hp_pct < 0.25:
        components["health_state"] = -0.50
    elif player_hp_pct < 0.50:
        components["health_state"] = -0.15

    if prev_obs is None or not np.any(prev_obs):
        return components

    if obs[1] < 31:
        components["positioning"] = -100 * (31 - float(obs[1]))

    prev_player_hp_pct = float(prev_obs[3])
    prev_nearest_enemy = enemy_block(prev_obs, 0)

    hp_lost = prev_player_hp_pct - player_hp_pct
    if hp_lost > 0 or player_hp_pct != 0.5:
        components["damage_taken"] = -20.0 * hp_lost

    enemy_hp_lost = prev_nearest_enemy["hp_pct"] - nearest_enemy["hp_pct"]
    if enemy_hp_lost > 0:
        components["damage_dealt"] = 25.0 * enemy_hp_lost

    return components


def get_reward(obs, action, prev_obs) -> float:
    """Return the sum of all non-terminal reward components."""
    return float(sum(get_reward_components(obs, action, prev_obs).values()))


def get_episode_outcome(obs, prev_obs):
    if prev_obs is None or not np.any(prev_obs):
        return None

    map_id = obs[5]
    if map_id != 53:
        return "loss"

    nearest_enemy = enemy_block(obs, 0)
    prev_nearest_enemy = enemy_block(prev_obs, 0)
    enemy_was_alive = prev_nearest_enemy["hp_pct"] > 0
    enemy_is_dead = nearest_enemy["hp_pct"] <= 0
    if enemy_was_alive and enemy_is_dead:
        print("kill")
        return "kill"

    return None


def get_termination(obs, prev_obs):
    return get_episode_outcome(obs, prev_obs) is not None


def get_truncated(obs, prev_obs, current_step):
    return bool(current_step >= MAX_EPISODE_STEPS or obs[1] < 25)
