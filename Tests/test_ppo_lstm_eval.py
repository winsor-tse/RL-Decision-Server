import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from Inference.ppo_lstm_eval import EvalArgs, evaluate, select_action


class FixedActor(nn.Module):
    def forward(self, embedding):
        return torch.tensor(
            [[-1.0, 2.0]],
            dtype=torch.float32,
            device=embedding.device,
        ).expand(embedding.shape[0], -1)


class FakeRecurrentAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = SimpleNamespace(num_layers=1, hidden_size=2)
        self.actor = FixedActor()
        self.seen_hidden_states = []

    def get_states(self, observation, lstm_state, _done):
        self.seen_hidden_states.append(lstm_state[0].detach().clone())
        next_lstm_state = (
            lstm_state[0] + 1.0,
            lstm_state[1] + 1.0,
        )
        return observation, next_lstm_state


class FakeEnv16:
    def __init__(self):
        self.actions = []
        self.episode_step = 0

    def reset(self):
        self.episode_step = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.episode_step += 1
        action_index = int(np.asarray(action).item())
        self.actions.append(action_index)
        done = self.episode_step == 2
        return (
            np.ones(2, dtype=np.float32),
            1.25,
            done,
            False,
            {
                "is_win": done,
                "episode_outcome": "win" if done else None,
            },
        )


class PPOLSTMEvaluationTests(unittest.TestCase):
    def test_default_checkpoint_selects_latest_lstm_run(self):
        self.assertIsNone(EvalArgs().model_path)

    def test_deterministic_action_advances_recurrent_state(self):
        model = FakeRecurrentAgent()
        state = (
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2),
        )

        action, next_state = select_action(
            model,
            np.zeros(2, dtype=np.float32),
            state,
            torch.device("cpu"),
            deterministic=True,
        )

        self.assertEqual(action, 1)
        self.assertTrue(torch.equal(next_state[0], torch.ones(1, 1, 2)))

    def test_evaluate_resets_lstm_state_between_env16_episodes(self):
        env = FakeEnv16()
        model = FakeRecurrentAgent()

        episodic_returns, wins = evaluate(
            env,
            model,
            eval_episodes=2,
            deterministic=True,
        )

        self.assertEqual(episodic_returns, [2.5, 2.5])
        self.assertEqual(wins, 2)
        self.assertEqual(env.actions, [1, 1, 1, 1])
        hidden_state_means = [
            state.mean().item() for state in model.seen_hidden_states
        ]
        self.assertEqual(hidden_state_means, [0.0, 1.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
