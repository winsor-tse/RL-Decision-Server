import os
import tempfile
import unittest
from unittest import mock

import minari

from Custom_enviornments.Test_Env.Env_16_BC import (
    BC_ACTIONS_11,
    KEY_TO_ACTION_INDEX,
    Env16BC,
    action_from_input,
)
from Offline.record_minari import record_minari_dataset


def make_world_state(*, player_hp=100, enemy_hp=100):
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
                "id": 1,
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


def tick(request_id, **world_state_kwargs):
    return {
        "type": "ai_tick",
        "requestId": request_id,
        "worldState": make_world_state(**world_state_kwargs),
    }


def input_snapshot(key, *, event=True):
    key_data = {"key": key, "vk": 0}
    return {
        "capture_available": True,
        "keys_down": [key_data],
        "events": (
            [{**key_data, "event": "key_down", "timestamp_unix_ns": 1}]
            if event
            else []
        ),
    }


class FakeInputCapture:
    def __init__(self, snapshots):
        self.snapshots = list(snapshots)
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def snapshot(self):
        return self.snapshots.pop(0)


class FakeSocket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.responses = []

    def recv_json(self):
        return self.messages.pop(0)

    def send_json(self, response):
        self.responses.append(response)


class Env16BCMappingTests(unittest.TestCase):
    def test_all_requested_keys_map_to_their_bc_indices(self):
        expected_labels = {
            "W": "up",
            "A": "left",
            "D": "right",
            "S": "down",
            "SPACE": "attack",
            "1": "castSpell:1",
            "2": "castSpell:2",
            "3": "castSpell:3",
            "5": "castSpell:5",
            "6": "castSpell:6",
            "7": "castSpell:7",
        }

        for key, action_idx in KEY_TO_ACTION_INDEX.items():
            with self.subTest(key=key):
                self.assertEqual(
                    action_from_input(input_snapshot(key)),
                    (action_idx, key),
                )
                self.assertEqual(BC_ACTIONS_11[action_idx], expected_labels[key])

    def test_invalid_and_key_up_inputs_do_not_map(self):
        self.assertIsNone(action_from_input(input_snapshot("Q")))
        self.assertIsNone(
            action_from_input(
                {
                    "keys_down": [],
                    "events": [{"key": "W", "event": "key_up"}],
                }
            )
        )

    def test_new_key_event_takes_precedence_over_a_held_key(self):
        snapshot = {
            "keys_down": [{"key": "W"}, {"key": "SPACE"}],
            "events": [{"key": "SPACE", "event": "key_down"}],
        }
        self.assertEqual(action_from_input(snapshot), (4, "SPACE"))

    def test_two_simultaneous_mapped_keys_are_ambiguous(self):
        snapshot = {
            "keys_down": [{"key": "W"}, {"key": "A"}],
            "events": [
                {"key": "W", "event": "key_down"},
                {"key": "A", "event": "key_down"},
            ],
        }
        self.assertIsNone(action_from_input(snapshot))


class Env16BCCollectionTests(unittest.TestCase):
    def make_env(self, messages, snapshots):
        socket = FakeSocket(messages)
        capture = FakeInputCapture(snapshots)
        env = Env16BC(socket=socket, input_capture=capture)
        return env, socket, capture

    def test_invalid_ticks_get_no_op_but_do_not_advance_the_environment(self):
        env, socket, capture = self.make_env(
            [tick("reset"), tick("invalid"), tick("valid", enemy_hp=75)],
            [
                input_snapshot("W"),
                input_snapshot("Q"),
                input_snapshot("SPACE"),
            ],
        )

        _, reset_info = env.reset(options={"minari_autoseed": False})
        self.assertEqual(env.next_action(), 0)
        _, _, terminated, truncated, info = env.step(env.next_action())

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(env.current_step, 1)
        self.assertEqual(env.ignored_ticks, 1)
        self.assertEqual(env.next_action(), 4)
        self.assertEqual(capture.reset_count, 1)
        self.assertEqual(reset_info.keys(), info.keys())
        self.assertTrue(all(response["move"] == "NoOp" for response in socket.responses))
        self.assertTrue(socket.responses[0]["reset"])
        self.assertFalse(socket.responses[1]["reset"])
        self.assertFalse(socket.responses[2]["reset"])

    def test_minari_records_every_required_episode_field(self):
        env, _, _ = self.make_env(
            [tick("reset"), tick("step", player_hp=98, enemy_hp=75)],
            [input_snapshot("W"), input_snapshot("SPACE")],
        )

        with tempfile.TemporaryDirectory() as datasets_path:
            with mock.patch.dict(
                os.environ,
                {"MINARI_DATASETS_PATH": datasets_path},
            ):
                collector = minari.DataCollector(env, record_infos=True)
                try:
                    collector.reset(options={"minari_autoseed": False})
                    collector.step(env.next_action())
                    dataset = collector.create_dataset(
                        "env16/BC-v0",
                        algorithm_name="human",
                        author="test",
                        description="Env16BC integration test",
                    )

                    self.assertEqual(dataset.total_episodes, 1)
                    self.assertEqual(dataset.total_steps, 1)
                    self.assertEqual(dataset.env_spec.id, "YugenSaga/Env16BC-v0")
                    episode = next(dataset.iterate_episodes())
                    self.assertEqual(episode.observations.shape, (2, 26))
                    self.assertEqual(episode.actions.tolist(), [0])
                    self.assertEqual(len(episode.rewards), 1)
                    self.assertEqual(len(episode.terminations), 1)
                    self.assertEqual(len(episode.truncations), 1)
                    self.assertTrue(episode.truncations[-1])
                    self.assertEqual(
                        set(episode.infos),
                        {
                            "current_step",
                            "next_state",
                            "reward_components",
                            "is_win",
                            "episode_outcome",
                        },
                    )
                    self.assertEqual(len(episode.infos["current_step"]), 2)
                finally:
                    collector.close()

    def test_recording_entrypoint_creates_a_versioned_dataset(self):
        env, socket, _ = self.make_env(
            [tick("reset"), tick("step", enemy_hp=50)],
            [input_snapshot("W"), input_snapshot("SPACE")],
        )

        with tempfile.TemporaryDirectory() as datasets_path:
            with mock.patch.dict(
                os.environ,
                {"MINARI_DATASETS_PATH": datasets_path},
            ):
                dataset = record_minari_dataset(
                    dataset_id="env16/entrypoint-v0",
                    max_steps=1,
                    author="test",
                    raw_env=env,
                )

                self.assertIsNotNone(dataset)
                self.assertEqual(dataset.total_steps, 1)
                self.assertEqual(dataset.total_episodes, 1)
                self.assertTrue(all(reply["move"] == "NoOp" for reply in socket.responses))


if __name__ == "__main__":
    unittest.main()
