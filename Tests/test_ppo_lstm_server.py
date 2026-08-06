import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from Training import PPO_lstm_server
from Training.PPO_lstm_server import (
    Agent,
    Args,
    recurrent_minibatches,
    save_agent,
)
from Training.PPO_server import Args as PPOArgs


class FakeWriter:
    instances = []

    def __init__(self, log_directory):
        self.log_directory = log_directory
        self.scalar_tags = []
        self.histogram_tags = []
        self.closed = False
        self.__class__.instances.append(self)

    def add_text(self, _tag, _text):
        pass

    def add_scalar(self, tag, _value, _step):
        self.scalar_tags.append(tag)

    def add_histogram(self, tag, _values, _step, bins):
        self.histogram_tags.append(tag)

    def close(self):
        self.closed = True


class FakeEnv16:
    Actions = ["up", "castSpell:1"]

    def __init__(self):
        self.single_observation_space = SimpleNamespace(shape=(10,))
        self.single_action_space = SimpleNamespace(n=len(self.Actions))
        self.config = {"OBS_PLAYER_SIZE": 6}
        self.current_step = 0
        self.closed = False

    def _observation(self):
        observation = np.zeros(10, dtype=np.float32)
        observation[3] = 0.75
        observation[8] = 0.5
        return observation

    def reset(self, seed=None):
        self.current_step = 0
        return self._observation(), {}

    def step(self, _action):
        self.current_step += 1
        done = self.current_step == 2
        return (
            self._observation(),
            1.0,
            done,
            False,
            {
                "reward_components": {
                    "damage_dealt": 1.0,
                    "terminal": 0.0,
                },
                "episode_outcome": "win" if done else None,
            },
        )

    def close(self):
        self.closed = True


class PPOLSTMTests(unittest.TestCase):
    def setUp(self):
        FakeWriter.instances.clear()

    def test_live_environment_defaults_to_one_instance(self):
        self.assertEqual(Args().num_envs, 1)
        self.assertEqual(Args().metrics_frequency, 1)

    def test_algorithm_defaults_match_feedforward_ppo(self):
        lstm_args = Args()
        ppo_args = PPOArgs()
        algorithm_fields = (
            "total_timesteps",
            "learning_rate",
            "num_envs",
            "num_steps",
            "metrics_frequency",
            "anneal_lr",
            "gamma",
            "gae_lambda",
            "num_minibatches",
            "update_epochs",
            "norm_adv",
            "clip_coef",
            "clip_vloss",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "target_kl",
            "batch_size",
            "minibatch_size",
            "num_iterations",
        )

        for field_name in algorithm_fields:
            self.assertEqual(
                getattr(lstm_args, field_name),
                getattr(ppo_args, field_name),
                field_name,
            )

    def test_main_parses_args_and_starts_training(self):
        parsed_args = Args(total_timesteps=128)
        train_mock = Mock()

        with (
            patch.object(PPO_lstm_server.tyro, "cli", return_value=parsed_args),
            patch.object(PPO_lstm_server, "train", train_mock),
        ):
            PPO_lstm_server.main()

        train_mock.assert_called_once_with(parsed_args)

    def test_recurrent_minibatches_cover_contiguous_rollout_sequences(self):
        minibatches = recurrent_minibatches(8, 4)

        self.assertEqual(len(minibatches), 4)
        self.assertEqual(
            sorted(np.concatenate(minibatches).tolist()),
            list(range(8)),
        )
        for minibatch in minibatches:
            np.testing.assert_array_equal(np.diff(minibatch), np.ones(1))

    def test_recurrent_minibatches_reject_uneven_sequences(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            recurrent_minibatches(5, 2)

    def test_saved_agent_loads_into_same_recurrent_architecture(self):
        source_agent = Agent(FakeEnv16())

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = save_agent(
                source_agent,
                str(Path(directory) / "ppo_lstm.pt"),
            )
            loaded_agent = Agent(FakeEnv16())
            loaded_agent.load_state_dict(
                torch.load(checkpoint_path, weights_only=True)
            )

        for source_parameter, loaded_parameter in zip(
            source_agent.parameters(),
            loaded_agent.parameters(),
        ):
            self.assertTrue(torch.equal(source_parameter, loaded_parameter))

    @patch.object(PPO_lstm_server, "SummaryWriter", FakeWriter)
    @patch.object(PPO_lstm_server, "Env16", FakeEnv16)
    def test_training_uses_env16_episode_and_dashboard_metrics(self):
        args = Args(
            total_timesteps=4,
            num_steps=4,
            num_minibatches=2,
            update_epochs=1,
            cuda=False,
            save_model=False,
        )

        PPO_lstm_server.train(args)

        writer = FakeWriter.instances[-1]
        expected_scalar_tags = {
            "charts/step_reward",
            "charts/episodic_return",
            "charts/episode_length",
            "charts/win_rate",
            "environment/player_hp",
            "environment/enemy_hp",
            "environment/action_frequency/up",
            "environment/action_frequency/castSpell_1",
            "rewards/damage_dealt",
            "rewards/episode_damage_dealt",
            "lstm/hidden_state_norm",
            "lstm/cell_state_norm",
        }
        self.assertTrue(expected_scalar_tags.issubset(writer.scalar_tags))
        self.assertIn("environment/action_frequency", writer.histogram_tags)
        self.assertTrue(writer.closed)

    @patch.object(PPO_lstm_server, "save_agent")
    @patch.object(PPO_lstm_server, "SummaryWriter", FakeWriter)
    @patch.object(PPO_lstm_server, "Env16", FakeEnv16)
    def test_default_model_is_saved_inside_tensorboard_run(
        self,
        save_agent_mock,
    ):
        args = Args(
            total_timesteps=4,
            num_steps=4,
            num_minibatches=2,
            update_epochs=1,
            cuda=False,
            save_model=True,
        )

        PPO_lstm_server.train(args)

        run_directory = Path(FakeWriter.instances[-1].log_directory)
        expected_checkpoint = run_directory / "PPO_lstm_server.pt"
        self.assertEqual(Path(args.model_path), expected_checkpoint)
        self.assertEqual(
            Path(save_agent_mock.call_args.args[1]),
            expected_checkpoint,
        )


if __name__ == "__main__":
    unittest.main()
