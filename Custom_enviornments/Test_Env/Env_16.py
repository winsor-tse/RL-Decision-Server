import logging

import numpy as np

from Custom_enviornments.BaseEnv import BaseEnv
from Custom_enviornments.Load_env_config import load_env_config
from Custom_enviornments.Test_Env import Env_conditions

LOGGER = logging.getLogger(__name__)


ACTIONS_15 = [
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
    #"castSpell:4",
    "castSpell:5",
    "castSpell:6",
    "castSpell:7",
]


# This action space is specific to Test_Env.
# A different game class should define its own env file and action list.
class Env16(BaseEnv):
    """Yugen Saga environment with the current 15-action discrete action space."""

    def __init__(self):
        super().__init__(actions=ACTIONS_15, config=load_env_config())
        self.kill_counter = 0
        self.next_Ent_state = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        message = self.socket.recv_json()
        LOGGER.debug("Reset message received: %s", message)
        self.kill_counter = 0

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
        self.next_Ent_state = Env_conditions.parse_entity_state(world_state)
        self.current_step = 0
        LOGGER.info("Environment reset")
        return self.next_state, self._get_info()

    def step(self, action):
        action_idx = self._normalize_action(action)
        if action_idx < 0 or action_idx >= len(self.Actions):
            raise ValueError(f"Action index {action_idx} is outside Env_16.")

        message = self.socket.recv_json()
        world_state = message.get("worldState", {})
        LOGGER.debug("World state received: %s", world_state)

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
        true_next_ent_state = Env_conditions.parse_entity_state(world_state)
        reward_components = Env_conditions.get_reward_components(
            real_next_state,
            action_idx,
            self.next_state,
            self.next_Ent_state,
            true_next_ent_state,
        )
        is_loss = Env_conditions.is_episode_loss(
            real_next_state,
            self.next_state,
        )
        # print(f"{episode_outcome}")
        truncated = Env_conditions.get_truncated(
            real_next_state,
            self.next_state,
            self.current_step,
        )

        self.next_state = real_next_state
        self.next_Ent_state = true_next_ent_state

        #Define termination reward
        terminated = False
        
        if is_loss:
            reward_components["terminal"] -= 100
            terminated = True
        elif reward_components["killed"] > 0:
            self.kill_counter += (reward_components["killed"] / 10)

        if self.kill_counter >= 5:
            terminated = True

        info = self._get_info()
        reward = float(sum(reward_components.values()))
        info["reward_components"] = reward_components
        info["is_win"] = self.kill_counter >= 5
        
        if is_loss and terminated:
            info["episode_outcome"] = "loss"
        elif info["is_win"] and terminated: 
            info["episode_outcome"] = "win"
        elif truncated:
            info["episode_outcome"] = "truncated"
        else:
            info["episode_outcome"] = None
        
        return self.next_state, reward, terminated, truncated, info

    def close(self):
        self.socket.close(linger=0)
