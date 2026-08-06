import unittest

import numpy as np

from Custom_enviornments.Load_env_config import load_env_config
from Custom_enviornments.Test_Env import Env_conditions
from Custom_enviornments.Test_Env.Env_16 import ACTIONS_11, Env16


def make_world_state(
    *,
    player_hp: int = 100,
    enemy_hp: int = 100,
    enemy_id: int = 1,
    enemy_distance: float | None = None,
) -> dict:
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
                "id": enemy_id,
                "type": "monster",
                "mapX": 11,
                "mapY": 30,
                "distance": enemy_distance,
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
        self.assertEqual(len(ACTIONS_11), 11)

    def test_normal_step_returns_boolean_termination_flags(self):
        world_state = make_world_state()
        env = Env16.__new__(Env16)
        env.Actions = list(ACTIONS_11)
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
        env.next_Ent_state = Env_conditions.parse_entity_state(world_state)
        env.current_step = 0
        env.kill_counter = 0

        _, _, terminated, truncated, info = env.step(np.array([0]))

        self.assertIs(terminated, False)
        self.assertIsInstance(truncated, bool)
        self.assertEqual(env.socket.response["move"], "up")
        self.assertIn("reward_components", info)

    def test_reward_components_sum_to_reward(self):
        previous = Env_conditions.parse_observation(make_world_state())
        current_world = make_world_state(player_hp=98, enemy_hp=50)
        current = Env_conditions.parse_observation(current_world)
        previous_entities = Env_conditions.parse_entity_state(
            make_world_state(enemy_hp=100)
        )
        current_entities = Env_conditions.parse_entity_state(current_world)

        components = Env_conditions.get_reward_components(
            current,
            0,
            previous,
            previous_entities,
            current_entities,
        )
        reward = Env_conditions.get_reward(
            current,
            0,
            previous,
            previous_entities,
            current_entities,
        )

        self.assertAlmostEqual(components["damage_taken"], -0.4, places=5)
        self.assertAlmostEqual(components["damage_dealt"], 12.5)
        self.assertAlmostEqual(reward, sum(components.values()))

    def test_entity_state_excludes_player_and_tracks_required_fields(self):
        world_state = make_world_state(enemy_hp=75, enemy_distance=7.5)
        world_state["entities"].append(
            {
                "id": 99,
                "type": "player",
                "isCurrentPlayer": True,
                "hp": 100,
                "maxHp": 100,
                "distance": 0,
            }
        )

        entity_state = Env_conditions.parse_entity_state(world_state)

        self.assertEqual(
            entity_state,
            {1: {"hp_pct": 0.75, "distance": 7.5}},
        )

    def test_damage_reward_matches_enemy_ids_not_observation_order(self):
        previous_world = make_world_state(enemy_id=1, enemy_hp=100)
        previous_world["entities"].append(
            {
                "id": 2,
                "type": "monster",
                "mapX": 12,
                "mapY": 30,
                "hp": 80,
                "maxHp": 100,
            }
        )
        current_world = make_world_state(enemy_id=1, enemy_hp=70)
        current_world["entities"].append(
            {
                "id": 2,
                "type": "monster",
                "mapX": 10,
                "mapY": 30,
                "hp": 60,
                "maxHp": 100,
            }
        )
        previous = Env_conditions.parse_observation(previous_world)
        current = Env_conditions.parse_observation(current_world)

        components = Env_conditions.get_reward_components(
            current,
            4,
            previous,
            Env_conditions.parse_entity_state(previous_world),
            Env_conditions.parse_entity_state(current_world),
        )

        self.assertAlmostEqual(components["damage_dealt"], 12.5)

    def test_nearby_missing_enemy_is_rewarded_but_distant_enemy_is_not(self):
        previous_world = make_world_state(
            enemy_id=1,
            enemy_hp=40,
            enemy_distance=14.9,
        )
        previous_world["entities"].append(
            {
                "id": 2,
                "type": "monster",
                "hp": 100,
                "maxHp": 100,
                "distance": 15,
            }
        )
        current_world = make_world_state()
        current_world["entities"] = []
        previous = Env_conditions.parse_observation(previous_world)
        current = Env_conditions.parse_observation(current_world)

        components = Env_conditions.get_reward_components(
            current,
            4,
            previous,
            Env_conditions.parse_entity_state(previous_world),
            Env_conditions.parse_entity_state(current_world),
        )

        self.assertAlmostEqual(components["damage_dealt"], 10.0)

    def test_step_stores_the_true_next_entity_state(self):
        previous_world = make_world_state(enemy_id=1, enemy_hp=100)
        current_world = make_world_state(enemy_id=1, enemy_hp=25)
        env = Env16.__new__(Env16)
        env.Actions = list(ACTIONS_11)
        env.config = load_env_config()
        env.socket = FakeSocket(
            {
                "type": "ai_tick",
                "requestId": "test-step",
                "worldState": current_world,
            }
        )
        env.next_state = Env_conditions.parse_observation(previous_world)
        env.next_Ent_state = Env_conditions.parse_entity_state(previous_world)
        env.current_step = 0
        env.kill_counter = 0

        env.step(np.array([4]))

        self.assertEqual(
            env.next_Ent_state,
            Env_conditions.parse_entity_state(current_world),
        )


if __name__ == "__main__":
    unittest.main()
