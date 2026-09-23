"""Headless Gym environment with seeded reset and Phase 2 movement steps."""
import gymnasium as gym
import numpy as np

from .actions import ACTIONS, contract_metadata
from .config import ScenarioConfig
from .map_loader import DEFAULT_MAP_PATH, load_map_definition
from .scenarios import build_scenario
from .observation import encode_observation, nearest_monsters
from .engine import Engine


class MysticSimEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, *, config=None, map_path=DEFAULT_MAP_PATH, trace=False):
        self.config = config if config is not None else ScenarioConfig()
        if not isinstance(self.config, ScenarioConfig):
            raise ValueError("config must be a ScenarioConfig")
        # All map I/O happens here, never during reset or observation encoding.
        self.map_definition = load_map_definition(map_path)
        self.world = None
        self.engine = None
        self.trace_enabled = bool(trace)
        self.episode_done = False
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        max_distance = self.config.width + self.config.height - 2
        high = [self.config.width - 1, self.config.height - 1, 4, 1, 1,
                self.config.map_id] + [max_distance, 4, 1, 1] * 5
        self.observation_space = gym.spaces.Box(
            low=np.zeros(26, dtype=np.float32), high=np.array(high, dtype=np.float32))
        self.contract = contract_metadata()

    def reset(self, *, seed=None, options=None):
        if options is not None:
            if not isinstance(options, dict):
                raise ValueError("reset options must be a dictionary")
            if set(options) - {"profile"}:
                raise ValueError("Only the profile reset option is supported")
            if options.get("profile", self.config.profile) != self.config.profile:
                raise ValueError("Requested profile does not match environment configuration")
        super().reset(seed=seed)
        self.world = build_scenario(self.map_definition, self.config, self.np_random)
        self.engine = Engine(self.world, self.config, self.np_random, trace=self.trace_enabled)
        self.episode_done = False
        info = {
            "seed": int(self.np_random_seed),
            "profile": self.config.profile,
            "simulation_time_ms": self.world.time_ms,
            "current_step": self.world.step_count,
            "kills": self.world.kills,
            "player_spawn": (self.world.player.x, self.world.player.y),
            "npc_spawns": {entity_id: (m.x, m.y) for entity_id, m in self.world.monsters.items()},
            "npc_move_intervals_ms": {entity_id: m.move_interval_ms for entity_id, m in self.world.monsters.items()},
            "selected_entity_ids": [m.entity_id for m in nearest_monsters(self.world)],
            "parsed_balance_cap": self.map_definition.parsed_balance_cap,
            "npc_effective_level_override": self.config.innie.effective_level,
            "scheduled_event_count": len(self.world.events),
            "contract": contract_metadata(),
        }
        return encode_observation(self.world), info

    def step(self, action):
        if self.engine is None or self.episode_done:
            raise gym.error.ResetNeeded("Call reset before starting or continuing a finished episode")
        applied, reason = self.engine.advance(action)
        terminated = not self.world.player.alive or self.world.kills >= self.config.reward.win_kills
        truncated = self.world.step_count >= self.config.reward.max_episode_steps
        self.episode_done = terminated or truncated
        info = {"current_step": self.world.step_count, "simulation_time_ms": self.world.time_ms,
                "profile": self.config.profile, "action_applied": applied,
                "action_failure_reason": reason, "kills": self.world.kills,
                "selected_entity_ids": [m.entity_id for m in nearest_monsters(self.world)],
                "combat_implemented": False, "reward_implemented": False,
                "episode_outcome": "loss" if not self.world.player.alive else
                    "win" if terminated else "truncated" if truncated else None}
        if self.trace_enabled:
            info["trace"] = list(self.engine.trace)
        return encode_observation(self.world), 0.0, terminated, truncated, info
