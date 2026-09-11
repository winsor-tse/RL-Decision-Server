import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gymnasium as gym
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from Offline import awac


class AWACTests(unittest.TestCase):
    def dataset(self):
        episode = SimpleNamespace(
            observations=np.arange(78, dtype=np.float32).reshape(3, 26),
            actions=np.array([1, 3]),
            rewards=np.array([1.0, 2.0]),
            terminations=np.array([False, False]),
            truncations=np.array([False, True]),
        )
        return SimpleNamespace(
            action_space=gym.spaces.Discrete(11),
            iterate_episodes=lambda: iter([episode]),
        )

    def test_full_buffer_wraps_and_truncation_bootstraps(self):
        data = awac.qlearning_dataset(self.dataset())
        self.assertEqual(data['terminals'].tolist(), [0, 0])
        np.testing.assert_array_equal(data['next_observations'][0], data['observations'][1])
        replay = awac.ReplayBuffer(26, 1, 2)
        replay.load_dataset(data)
        replay.add_transition(np.zeros(26), 10, 7, np.ones(26), True)
        self.assertEqual(replay._size, 2)
        self.assertEqual(replay._actions[0].item(), 10)
        self.assertEqual(replay._dones[0].item(), 1)

    def test_offline_training_logs_and_saves_without_live_env(self):
        with tempfile.TemporaryDirectory() as directory:
            config = awac.TrainConfig(
                checkpoints_path=directory, offline_iterations=2,
                batch_size=2, buffer_size=2, hidden_dim=8, device='cpu',
            )
            with mock.patch.object(awac.minari, 'load_dataset', return_value=self.dataset()) as load, mock.patch.object(awac, 'Env16') as env:
                awac.train(config)
            load.assert_called_once_with('env16/BC-v0', download=False)
            env.assert_not_called()
            checkpoint = torch.load(Path(config.checkpoints_path, awac.AWAC_MODEL_FILENAME), weights_only=True)
            self.assertEqual(checkpoint['actions'], awac.BC_ACTIONS_11)
            self.assertEqual(checkpoint['completed_steps'], 2)
            actor = awac.Actor(26, 11, 8)
            actor.load_state_dict(checkpoint['actor'])
            actor.eval()
            self.assertIn(actor.act(np.zeros(26)), range(11))
            events = EventAccumulator(config.checkpoints_path).Reload()
            for tag in ('train/actor_loss', 'train/critic_loss'):
                values = events.Scalars(tag)
                self.assertEqual(len(values), 2)
                self.assertTrue(all(np.isfinite(event.value) for event in values))

    def test_online_uses_bc_order_and_gymnasium_episode_api(self):
        with tempfile.TemporaryDirectory() as directory:
            config = awac.TrainConfig(
                checkpoints_path=directory, offline_iterations=1,
                online_iterations=2, batch_size=2, buffer_size=2,
                hidden_dim=8, device='cpu',
            )
            env = mock.Mock()
            env.reset.return_value = (np.zeros(26), {})
            env.step.return_value = (np.ones(26), 1.0, False, True, {})
            with mock.patch.object(awac.minari, 'load_dataset', return_value=self.dataset()), mock.patch.object(awac, 'Env16', return_value=env) as factory:
                awac.train(config)
            factory.assert_called_once_with(actions=awac.BC_ACTIONS_11)
            self.assertEqual(env.step.call_count, 2)
            for call in env.step.call_args_list:
                self.assertIn(call.args[0], range(11))
            env.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
