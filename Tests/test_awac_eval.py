import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from Custom_enviornments.Test_Env.Env_16_BC import BC_ACTIONS_11
from Inference.awac_eval import evaluate_live, load_actor, select_action
from Offline.awac import Actor


class FakeEnv:
    def __init__(self):
        self.actions = []

    def reset(self, seed=None):
        return np.array([3.0, 5.0], dtype=np.float32), {}

    def step(self, action):
        self.actions.append(action)
        return (
            np.array([3.0, 5.0], dtype=np.float32),
            2.5,
            True,
            False,
            {"is_win": True, "episode_outcome": "win"},
        )


class AWACLiveEvaluationTests(unittest.TestCase):
    def actor(self):
        actor = Actor(2, len(BC_ACTIONS_11), 8)
        for parameter in actor.parameters():
            torch.nn.init.zeros_(parameter)
        actor._mlp[-1].bias.data[3] = 5.0
        actor.eval()
        return actor

    def test_checkpoint_restores_actor_and_normalization(self):
        source = self.actor()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "AWAC_model.pt")
            torch.save(
                {
                    "actor": source.state_dict(),
                    "state_mean": torch.tensor([1.0, 2.0]),
                    "state_std": torch.tensor([2.0, 3.0]),
                    "state_dim": 2,
                    "action_dim": len(BC_ACTIONS_11),
                    "hidden_dim": 8,
                    "actions": list(BC_ACTIONS_11),
                },
                path,
            )
            actor, mean, std, actions = load_actor(path, torch.device("cpu"))

        np.testing.assert_array_equal(mean, [1.0, 2.0])
        np.testing.assert_array_equal(std, [2.0, 3.0])
        self.assertEqual(actions, BC_ACTIONS_11)
        self.assertEqual(
            select_action(
                actor, np.array([3.0, 5.0]), mean, std,
                torch.device("cpu"), deterministic=True,
            ),
            3,
        )

    def test_live_evaluation_uses_policy_action_and_reports_wins(self):
        env = FakeEnv()
        returns, wins = evaluate_live(
            env,
            self.actor(),
            np.array([1.0, 2.0]),
            np.array([2.0, 3.0]),
            eval_episodes=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(returns, [2.5, 2.5])
        self.assertEqual(wins, 2)
        self.assertEqual(env.actions, [3, 3])

    def test_incomplete_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "AWAC_model.pt")
            torch.save({"actor": {}}, path)
            with self.assertRaisesRegex(ValueError, "missing required fields"):
                load_actor(path, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
