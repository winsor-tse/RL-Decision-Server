import unittest

import numpy as np
import torch
import torch.nn as nn

from Inference.ppo_eval import evaluate, select_action


class FixedActor(nn.Module):
    def forward(self, observation):
        return torch.tensor(
            [[-1.0, 2.0]],
            dtype=torch.float32,
            device=observation.device,
        ).expand(observation.shape[0], -1)


class FakePPOAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = FixedActor()

    def get_action_and_value(self, observation):
        action = torch.ones(
            observation.shape[0],
            dtype=torch.long,
            device=observation.device,
        )
        zeros = torch.zeros_like(action, dtype=torch.float32)
        values = torch.zeros(
            (observation.shape[0], 1),
            dtype=torch.float32,
            device=observation.device,
        )
        return action, zeros, zeros, values


class FakeEnv:
    def __init__(self):
        self.actions = []

    def reset(self):
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        action_index = int(np.asarray(action).item())
        self.actions.append(action_index)
        return (
            np.ones(2, dtype=np.float32),
            2.5,
            True,
            False,
            {"is_win": True, "episode_outcome": "win"},
        )


class PPOEvaluationTests(unittest.TestCase):
    def test_deterministic_action_uses_highest_policy_logit(self):
        action = select_action(
            FakePPOAgent(),
            np.zeros(2, dtype=np.float32),
            torch.device("cpu"),
            deterministic=True,
        )

        self.assertEqual(action, 1)

    def test_evaluate_returns_episode_results_and_wins(self):
        env = FakeEnv()

        episodic_returns, wins = evaluate(
            env,
            FakePPOAgent(),
            eval_episodes=2,
            deterministic=True,
        )

        self.assertEqual(episodic_returns, [2.5, 2.5])
        self.assertEqual(wins, 2)
        self.assertEqual(env.actions, [1, 1])


if __name__ == "__main__":
    unittest.main()
