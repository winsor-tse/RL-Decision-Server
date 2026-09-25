"""Thin Gymnasium wrapper around the clock, reward profiles, and diagnostics."""
import gymnasium as gym
import numpy as np
from dataclasses import replace

from .actions import ACTIONS, contract_metadata
from .config import ScenarioConfig
from .map_loader import DEFAULT_MAP_PATH, load_map_definition
from .scenarios import build_scenario
from .observation import encode_observation
from .engine import Engine
from . import rewards
from .diagnostics import build_info


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
        self.reward_config = self.config.reward
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
            if set(options) - {"profile", "reward_profile"}:
                raise ValueError("Only profile and reward_profile reset options are supported")
            if options.get("profile", self.config.profile) != self.config.profile:
                raise ValueError("Requested profile does not match environment configuration")
        reward_config = replace(self.config.reward, profile=(options or {}).get("reward_profile", self.config.reward.profile))
        super().reset(seed=seed)
        self.reward_config = reward_config
        self.world = build_scenario(self.map_definition, self.config, self.np_random)
        self.engine = Engine(self.world, self.config, self.np_random, trace=self.trace_enabled)
        self.episode_done = False
        info = {**build_info(self, components=rewards.empty_components()),
            "player_spawn": (self.world.player.x, self.world.player.y),
            "npc_spawns": {entity_id: (m.x, m.y) for entity_id, m in self.world.monsters.items()},
            "npc_move_intervals_ms": {entity_id: m.move_interval_ms for entity_id, m in self.world.monsters.items()},
            "parsed_balance_cap": self.map_definition.parsed_balance_cap,
            "npc_effective_level_override": self.config.innie.effective_level,
            "scheduled_event_count": len(self.world.events),
            "contract": contract_metadata(),
            "regeneration": {"enabled": self.config.timing.regen_enabled,
                             "interval_ms": self.config.timing.regen_ms,
                             "hp_per_second_override": self.config.player.hp_regen_override,
                             "mp_per_second_override": self.config.player.mp_regen_override},
        }
        return encode_observation(self.world), info

    def step(self, action):
        if self.engine is None or self.episode_done:
            raise gym.error.ResetNeeded("Call reset before starting or continuing a finished episode")
        previous_obs = encode_observation(self.world)
        applied, reason = self.engine.advance(action, self.config.timing.step_ms)
        obs = encode_observation(self.world)
        terminated, truncated, outcome, end_reason = rewards.episode_status(self.world, self.reward_config)
        self.episode_done = terminated or truncated
        components = rewards.calculate(self.world, self.reward_config, previous_obs, obs, action,
                                       self.engine.damage_events, self.engine.death_events, outcome)
        info = build_info(self, action=action, applied=applied, reason=reason,
                          components=components, outcome=outcome, end_reason=end_reason)
        return obs, float(sum(components.values())), terminated, truncated, info
