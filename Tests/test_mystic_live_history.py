"""Opt-in 71-input training, loading, simulation/live parity and snapshot decisions."""
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
from Custom_enviornments.Mystic_Sim.observation import encode_observation
from Inference.Mystic_Live.inputs import LivePolicyInputs
from Inference.Mystic_Live.commands import LiveCommands, COMMAND_SCHEMA
from Inference.Mystic_Live.lifecycle import SingleRun
from Inference.Mystic_Live.policy import FrozenPolicy
from Training.Mystic_Sim.history import LiveObservableHistory, LIVE_HISTORY_SCHEMA
from Training.Mystic_Sim.core import PPOConfig, SimBatch
from Training.Mystic_Sim.evaluate import evaluate, load_evaluation_agent
from Training.Mystic_Sim.train import train
from Tests.test_mystic_sim_ppo import short_env
from Tests.test_mystic_live_lifecycle import request


def project(world):
    p = world.player
    return dict(timestamp=world.time_ms, player=dict(id=p.entity_id, mapID='map53', mapX=p.x,
                mapY=p.y, direction=p.facing.name.lower(), hp=p.hp, maxHp=p.max_hp,
                mp=p.mp, maxMp=p.max_mp), entities=[dict(id=m.entity_id, type='monster',
                isCurrentPlayer=False, mapX=m.x, mapY=m.y, hp=m.hp, maxHp=m.max_hp,
                mp=m.mp, maxMp=m.max_mp) for m in world.monsters.values()])


class LiveHistoryTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_sim_live_tensor_parity_and_failed_choices(self):
        env = MysticSimEnv()
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=7)
        sim = LiveObservableHistory.from_env(env)
        live = LivePolicyInputs(71)
        for action in (4, 4, 5, 6, 0, 3):
            np.testing.assert_array_equal(sim.encode(obs), live.encode(project(env.world))['values'])
            old = obs
            obs, _, _, _, info = env.step(action)
            sim.update_from_info(old, action, info)
            live.record_action(action)
            self.assertEqual(sim.features[action], 1)
        sim.reset()
        self.assertFalse(sim.features.any())
        self.assertFalse(live.requires_execution_feedback)
        self.assertEqual(live.schema, LIVE_HISTORY_SCHEMA)

    def test_batch_reset_and_configuration_restrictions(self):
        batch = SimBatch(1, 7, factory=short_env, recurrent=True, history_mode='live')
        self.addCleanup(batch.close)
        self.assertEqual(batch.observations.shape, (1, 71))
        for _ in range(8):
            final, *_ = batch.step([4])
        self.assertEqual(final[0, 60], 1)
        self.assertFalse(batch.observations[0, 56:63].any())
        for config in (PPOConfig(history_mode='live'), PPOConfig(recurrent=True, history_mode='live')):
            with self.assertRaises(ValueError):
                config.validate()

    def test_real_training_checkpoint_live_replay_and_watch(self):
        config = PPOConfig(recurrent=True, recurrent_encoder='flat', history_mode='live',
                           total_timesteps=80, num_steps=8, num_minibatches=2,
                           update_epochs=1, eval_episodes=1)
        def quick_evaluate(agent, seeds, **kwargs):
            kwargs['factory'] = short_env
            return evaluate(agent, seeds, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch('Training.Mystic_Sim.train.evaluate', quick_evaluate):
            root = train(config, run_dir=Path(directory)/'smoke', device='cpu',
                         record_gameplay=False, stop_after_checkpoint=1)
            path = root/'checkpoints/checkpoint_01.pt'
            actor = FrozenPolicy(path)
            self.assertEqual(actor.input_size, 71)
            reference, _, meta = load_evaluation_agent(path)
            self.assertEqual(meta['history_schema'], LIVE_HISTORY_SCHEMA)
            run = SingleRun(actor, input_adapter=actor.input_adapter, progression='snapshot',
                            command_adapter=LiveCommands(actor.input_adapter.encoder))
            state = reference.initial_state(1)
            for index in range(3):
                message = request(index)
                message['capabilities'] = [COMMAND_SCHEMA]
                result = run.handle(message)
                self.assertIsNotNone(result['actionId'])
                observation = torch.tensor(run.last_decision['policy_input']['values']).reshape(1, 1, 71)
                with torch.inference_mode():
                    logits, _, state, _ = reference.sequence(observation, state, torch.tensor([[index == 0]]))
                self.assertEqual(run.last_decision['action'], int(logits[0, 0].argmax()))
                for actual, expected in zip(run.memory, state[:2]):
                    torch.testing.assert_close(actual, expected)
            from Training.Mystic_Sim.watch import PolicySession
            session = PolicySession(reference, short_env, mode='greedy')
            try:
                session.step()
                self.assertEqual(len(session.snapshot.observation), 71)
            finally:
                session.close()
            # Resume must retain the new history contract without caller overrides.
            train(PPOConfig(), resume=path, device='cpu', record_gameplay=False, stop_after_checkpoint=2)
            resumed = FrozenPolicy(root/'checkpoints/checkpoint_02.pt')
            self.assertEqual(resumed.input_size, 71)
