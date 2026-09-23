"""Gym registration scaffold; deliberately cannot produce fake transitions."""
import gymnasium as gym
import numpy as np

from .actions import ACTIONS, contract_metadata


class MysticSimEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(26,), dtype=np.float32)
        self.contract = contract_metadata()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        raise NotImplementedError("MysticSim Phase 0 defines contracts only; reset is Phase 1")

    def step(self, action):
        raise NotImplementedError("MysticSim engine and transitions are not implemented yet")
