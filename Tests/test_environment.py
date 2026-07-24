import unittest

import numpy as np

from Custom_enviornments.Load_env_config import load_env_config
from Custom_enviornments.Test_Env import Env_conditions
from Custom_enviornments.Test_Env.Env_16 import ACTIONS_15, Env16


def make_world_state(*, player_hp: int = 100, enemy_hp: int = 100) -> dict:
    return {
        "player": {
            "mapX": 10,
            "mapY": 30,
            "direction": "up",
            "hp": player_hp,
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
                "hp": enemy_hp,
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

        _, _, terminated, truncated, info = env.step(np.array([0]))

        self.assertIs(terminated, False)
        self.assertIsInstance(truncated, bool)
        self.assertEqual(env.socket.response["move"], "up")
        self.assertIn("reward_components", info)

    def test_reward_components_sum_to_reward(self):
        previous = Env_conditions.parse_observation(make_world_state())
        current = Env_conditions.parse_observation(
            make_world_state(player_hp=80, enemy_hp=50)
        )

        components = Env_conditions.get_reward_components(current, 0, previous)
        reward = Env_conditions.get_reward(current, 0, previous)

        self.assertAlmostEqual(components["damage_taken"], -0.4)
        self.assertAlmostEqual(components["damage_dealt"], 12.5)
        self.assertAlmostEqual(reward, sum(components.values()))


if __name__ == "__main__":
    unittest.main()
