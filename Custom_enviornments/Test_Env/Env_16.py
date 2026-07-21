from pathlib import Path
import sys

import numpy as np

from Custom_enviornments.BaseEnv import BaseEnv
from Custom_enviornments.Load_env_config import load_env_config
from Custom_enviornments.Test_Env import Env_conditions


ACTIONS_16 = [
    "up",
    "down",
    "left",
    "right",
    "direction:up",
    "direction:down",
    "direction:left",
    "direction:right",
    "attack",
    "castSpell:1",
    "castSpell:2",
    "castSpell:3",
    "castSpell:4",
    "castSpell:5",
    "castSpell:6",
    "castSpell:7",
]


# This action space is specific to Test_Env.
# A different game class should define its own env file and action list.
class Env16(BaseEnv):
    """Yugen Saga environment with a 16-action discrete action space."""

    def __init__(self):
        super().__init__(actions=ACTIONS_16, config=load_env_config())

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        message = self.socket.recv_json()
        print("RESET Received:", message)

        response = self._build_response(
            message=message,
            move="direction:up",
            reset=True,
        )
        self.socket.send_json(response)

        world_state = message.get("worldState", {})
        self.next_state = Env_conditions.parse_observation(
            world_state,
            int(self.config["OBS_SIZE"]),
        )
        self.current_step = 0
        print("RESET")
        return self.next_state, self._get_info()

    def step(self, action):
        action_idx = self._normalize_action(action)
        if action_idx < 0 or action_idx >= len(self.Actions):
            raise ValueError(f"Action index {action_idx} is outside Env_16.")

        message = self.socket.recv_json()
        world_state = message.get("worldState", {})

        response = self._build_response(
            message=message,
            move=self.Actions[action_idx],
            reset=False,
        )
        self.socket.send_json(response)

        self.current_step += 1
        real_next_state = Env_conditions.parse_observation(
            world_state,
            int(self.config["OBS_SIZE"]),
        )
        reward = Env_conditions.get_reward(real_next_state, action_idx, self.next_state)
        episode_outcome = Env_conditions.get_episode_outcome(
            real_next_state,
            self.next_state,
        )
        terminated = episode_outcome is not None
        truncated = Env_conditions.get_truncated(
            real_next_state,
            self.next_state,
            self.current_step,
        )

        self.next_state = real_next_state

        #Define termination reward
        if terminated == "loss":
            reward -= 500
        elif terminated == "win":
            reward += 500

        info = self._get_info()
        info["is_win"] = episode_outcome == "win"
        info["episode_outcome"] = (
            episode_outcome
            if terminated
            else "truncated" if truncated else None
        )
        return self.next_state, reward, terminated, truncated, info

    def close(self):
        self.socket.close(linger=0)
