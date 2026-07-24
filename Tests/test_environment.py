import unittest

import numpy as np

from Custom_enviornments.Load_env_config import load_env_config
from Custom_enviornments.Test_Env import Env_conditions
from Custom_enviornments.Test_Env.Env_16 import ACTIONS_15, Env16


def make_world_state() -> dict:
    return {
        "player": {
            "mapX": 10,
            "mapY": 30,
            "direction": "up",
            "hp": 100,
            "maxHp": 100,
            "mp": 50,
            "maxMp": 100,
            "mapID": "map53",
        },
        "entities": [
            {
                "type": "monster",
                "mapX": 11,
                "mapY": 30,
                "hp": 100,
                "maxHp": 100,
                "mp": 0,
                "maxMp": 0,
            }
        ],
    }


class FakeSocket:
    def __init__(self, message: dict):
        self.message = message
        self.response = None

    def recv_json(self) -> dict:
        return self.message

    def send_json(self, response: dict) -> None:
        self.response = response


class EnvironmentCorrectnessTests(unittest.TestCase):
    def test_action_list_contains_fifteen_actions(self):
        self.assertEqual(len(ACTIONS_15), 15)

    def test_normal_step_returns_boolean_termination_flags(self):
        world_state = make_world_state()
        env = Env16.__new__(Env16)
        env.Actions = list(ACTIONS_15)
        env.config = load_env_config()
        env.socket = FakeSocket(
            {
                "type": "ai_tick",
                "requestId": "test-step",
                "worldState": world_state,
            }
        )
        env.next_state = Env_conditions.parse_observation(
            world_state,
            int(env.config["OBS_SIZE"]),
        )
        env.current_step = 0
        env.kill_counter = 0

        _, _, terminated, truncated, _ = env.step(np.array([0]))

        self.assertIs(terminated, False)
        self.assertIsInstance(truncated, bool)
        self.assertEqual(env.socket.response["move"], "up")


if __name__ == "__main__":
    unittest.main()
