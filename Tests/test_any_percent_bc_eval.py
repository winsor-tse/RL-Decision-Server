import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from Inference.any_percent_bc_eval import (
    Actor,
    PreparedDataset,
    discrete_bc_actions,
    env16_action_from_bc_index,
    evaluate_dataset,
    load_actor,
    prepare_dataset,
)


class FakeDataset:
    def __init__(self, episodes):
        self.episodes = episodes

    def iterate_episodes(self):
        return iter(self.episodes)


class AnyPercentBCEvaluationTests(unittest.TestCase):
    def test_predictions_round_and_clip_to_bc_action_space(self):
        predictions = np.array([-2.0, 0.49, 1.51, 20.0])

        actions = discrete_bc_actions(predictions)

        np.testing.assert_array_equal(actions, [0, 0, 2, 10])

    def test_bc_movement_order_translates_to_env16_order(self):
        env = SimpleNamespace(
            Actions=[
                "up",
                "down",
                "left",
                "right",
                "attack",
                "castSpell:1",
                "castSpell:2",
                "castSpell:3",
                "castSpell:5",
                "castSpell:6",
                "castSpell:7",
            ]
        )

        self.assertEqual(env16_action_from_bc_index(env, 1), 2)
        self.assertEqual(env16_action_from_bc_index(env, 2), 3)
        self.assertEqual(env16_action_from_bc_index(env, 3), 1)

    def test_prepare_dataset_selects_top_return_episodes(self):
        low_return = SimpleNamespace(
            observations=np.array([[1.0, 1.0], [2.0, 2.0]]),
            actions=np.array([1]),
            rewards=np.array([1.0]),
        )
        high_return = SimpleNamespace(
            observations=np.array([[3.0, 3.0], [4.0, 4.0]]),
            actions=np.array([7]),
            rewards=np.array([10.0]),
        )

        with mock.patch(
            "Inference.any_percent_bc_eval.minari.load_dataset",
            return_value=FakeDataset([low_return, high_return]),
        ):
            prepared = prepare_dataset(
                "env16/test-v0",
                top_fraction=0.5,
                gamma=0.99,
                normalize_state=False,
            )

        np.testing.assert_array_equal(prepared.observations, [[3.0, 3.0]])
        np.testing.assert_array_equal(prepared.actions, [7.0])
        self.assertEqual(prepared.max_action, 7.0)

    def test_checkpoint_load_and_dataset_evaluation_write_csv(self):
        prepared = PreparedDataset(
            observations=np.zeros((3, 2), dtype=np.float32),
            actions=np.zeros(3, dtype=np.float32),
            state_mean=0.0,
            state_std=1.0,
            max_action=1.0,
        )
        source_actor = Actor(state_dim=2, max_action=1.0)
        for parameter in source_actor.parameters():
            torch.nn.init.zeros_(parameter)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir, "BC_model.pt")
            output_csv = Path(temp_dir, "predictions.csv")
            torch.save({"actor": source_actor.state_dict()}, checkpoint)

            actor = load_actor(checkpoint, prepared, torch.device("cpu"))
            mse, accuracy = evaluate_dataset(
                actor,
                prepared,
                torch.device("cpu"),
                output_csv,
            )

            self.assertEqual(mse, 0.0)
            self.assertEqual(accuracy, 1.0)
            self.assertTrue(output_csv.is_file())
            self.assertEqual(
                output_csv.read_text(encoding="utf-8").splitlines()[0],
                "predicted_value,predicted_action,recorded_action",
            )


if __name__ == "__main__":
    unittest.main()
