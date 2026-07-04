import math
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from Custom_enviornments.Load_env_config import load_env_config


class BaseEnv(gym.Env):
    """Example bare-bones Gym/ZMQ environment shared by concrete game envs."""

    def __init__(self, actions, config=None):
        super().__init__()
        self.config = config or load_env_config()
        self.Actions = list(actions)

        obs_size = int(self.config["OBS_SIZE"])
        self.single_action_space = spaces.Discrete(len(self.Actions))
        self.single_observation_space = spaces.Box(
            low=0,
            high=math.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )
        self.action_space = self.single_action_space
        self.observation_space = self.single_observation_space

        self.next_state = np.zeros(obs_size, dtype=np.float32)
        self.current_step = 0

        import zmq

        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(self.config["ZMQ_BIND_URL"])

    def _get_obs(self):
        return self.next_state

    def _get_info(self):
        return {
            "current_step": self.current_step,
            "next_state": self.next_state,
        }

    def _normalize_action(self, action):
        return int(np.asarray(action).item())

    def _build_response(self, message, move, reset):
        message_type = message.get("type")
        if message_type == "ai_tick":
            return {
                "type": "ai_result",
                "requestId": message.get("requestId"),
                "move": move,
                "reset": reset,
                "serverTime": time.time(),
            }

        return {
            "type": "error",
            "requestId": message.get("requestId"),
            "error": f"Unknown message type: {message_type}",
        }

    def close(self):
        self.socket.close(linger=0)
