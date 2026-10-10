"""Sequence identity, episode isolation and real recurrent training gates."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from Training.Mystic_Sim.core import PPOConfig, SimBatch, make_env, metadata, load_checkpoint
from Training.Mystic_Sim.history import ObservableHistory, observation_high, terrain_sensors
from Training.Mystic_Sim.recurrent import RecurrentAgent, recurrent_minibatches, optimize_recurrent
from Training.Mystic_Sim.evaluate import evaluate
from Training.Mystic_Sim.train import train
from Tests.test_mystic_sim_ppo import short_env


class RecurrentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def agent(self, device='cpu'):
        env = make_env()
        self.addCleanup(env.close)
        return RecurrentAgent(observation_high(env.observation_space.high)).to(device)

    def test_history_is_causal_and_resets(self):
        env = make_env()
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=1)
        h = ObservableHistory.from_env(env)
        h.update(obs, 4, applied=True, blocked=False, cooldown_rejected=False)
        self.assertAlmostEqual(float(h.features[10]), .02)
        self.assertEqual(h.features[13], 1)
        h.update(obs, 4, applied=False, blocked=False, cooldown_rejected=True)
        self.assertAlmostEqual(float(h.features[10]), .04)  # rejection does not restart cast age
        self.assertEqual(h.features[9], 1)
        h.update(obs, 0, applied=False, blocked=True, cooldown_rejected=False)
        self.assertEqual(h.features[8], 1)
        self.assertEqual(h.encode(obs).shape, (50,))
        self.assertTrue(env.observation_space.contains(h.encode(obs)[:26]))
        h.reset()
        self.assertFalse(h.features.any())

    def test_contiguous_blocks_cover_all_times(self):
        blocks = recurrent_minibatches(128, 4)
        times = np.concatenate([np.arange(128)[ix] for ix in blocks])
        np.testing.assert_array_equal(np.sort(times), np.arange(128))
        for ix in blocks:
            self.assertEqual(ix.stop-ix.start, 32)
        with self.assertRaises(ValueError):
            replace(PPOConfig(), recurrent=True, num_steps=6, num_envs=4, num_minibatches=4).validate()

    def test_sequence_replay_matches_collection_and_resets_independently(self):
        agent = self.agent()
        obs = torch.rand(12, 3, 50) * agent.observation_scale
        starts = torch.zeros(12, 3, dtype=torch.bool)
        starts[0] = True
        starts[5, 1] = True
        state = agent.initial_state(3)
        saved, logits, values = [], [], []
        with torch.no_grad():
            for t in range(12):
                saved.append(tuple(s.clone() for s in state))
                logit, value, state, _ = agent.sequence(obs[t:t+1], state, starts[t:t+1])
                logits.append(logit); values.append(value)
            for ix in recurrent_minibatches(12, 3):
                logit, value, _, _ = agent.sequence(obs[ix], saved[ix.start], starts[ix])
                torch.testing.assert_close(logit, torch.cat(logits)[ix])
                torch.testing.assert_close(value, torch.cat(values)[ix])
            # A reset world gives exactly the same outputs as a fresh world,
            # even while the other two worlds retain memory.
            fresh, _, _, _ = agent.sequence(obs[5:6, 1:2], agent.initial_state(1), starts[5:6, 1:2])
            torch.testing.assert_close(fresh[0, 0], logits[5][0, 1])
            # No reset at a rollout boundary: split and whole sequence agree.
            full, _, _, _ = agent.sequence(obs, saved[0], starts)
            torch.testing.assert_close(full, torch.cat(logits))

    def test_bptt_reaches_prior_observations_but_not_across_reset(self):
        agent = self.agent()
        obs = torch.rand(6, 1, 50, requires_grad=True)
        starts = torch.zeros(6, 1, dtype=torch.bool)
        starts[3] = True
        logits, _, _, _ = agent.sequence(obs, agent.initial_state(1), starts)
        logits[-1].sum().backward()
        self.assertEqual(float(obs.grad[:3].abs().sum()), 0.)
        self.assertGreater(float(obs.grad[3:5].abs().sum()), 0.)

    def test_history_final_observation_kept_before_autoreset(self):
        batch = SimBatch(2, 4, factory=short_env, recurrent=True)
        self.addCleanup(batch.close)
        batch.envs[0].world.step_count = 7
        final, _, _, truncated, _ = batch.step([4, 4])
        self.assertTrue(truncated[0])
        self.assertEqual(final[0, 26+4], 1)
        self.assertFalse(batch.observations[0, 26:42].any())
        # Sensors are recomputed at both the final and reset positions.
        for i in range(2):
            for encoded in (final[i], batch.observations[i]):
                expected = terrain_sensors(int(encoded[0]), int(encoded[1]), width=100,
                    height=100, blocked_cells=batch.envs[i].map_definition.blocked_cells)
                np.testing.assert_array_equal(encoded[42:], expected)
        self.assertEqual(batch.observations[1, 26+4], 1)

    def test_sensor_edges_range_and_first_blocker(self):
        sensed = terrain_sensors(10, 10, width=30, height=30,
            blocked_cells={(10, 9), (10, 7), (10, 14), (5, 10), (12, 10)})
        # Adjacent up, farthest visible down, left just beyond range, right two away.
        np.testing.assert_array_equal(sensed, [0, 1, .75, 1, 1, 0, .25, 1])
        edge = terrain_sensors(0, 0, width=30, height=30, blocked_cells=())
        np.testing.assert_array_equal(edge, [0, 1, 1, 0, 0, 1, 1, 0])
        edge = terrain_sensors(29, 29, width=30, height=30, blocked_cells=())
        np.testing.assert_array_equal(edge, [1, 0, 0, 1, 1, 0, 0, 1])

    def test_sensors_translate_with_local_geometry_and_recompute_after_movement(self):
        cells = {(10, 9), (10, 13), (8, 10)}
        a = terrain_sensors(10, 10, width=50, height=50, blocked_cells=cells)
        b = terrain_sensors(20, 20, width=50, height=50,
                            blocked_cells={(x+10, y+10) for x, y in cells})
        np.testing.assert_array_equal(a, b)
        high = np.ones(26, dtype=np.float32)
        high[:2] = 49
        h = ObservableHistory(high, blocked_cells=cells)
        obs = np.zeros(26, dtype=np.float32)
        obs[:2] = 10, 10
        np.testing.assert_array_equal(h.encode(obs)[42:], a)
        obs[:2] = 10, 11
        self.assertEqual(h.encode(obs)[42], .25)

    def test_sensors_ignore_entities(self):
        env = make_env()
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=11)
        history = ObservableHistory.from_env(env)
        before = history.encode(obs)[42:]
        monster = next(iter(env.world.monsters.values()))
        monster.x, monster.y = int(obs[0]), int(obs[1])-1
        np.testing.assert_array_equal(history.encode(obs)[42:], before)

    def test_previous_recurrent_sensor_contract_is_rejected(self):
        env = make_env()
        self.addCleanup(env.close)
        contract = metadata(env, recurrent=True)
        self.assertEqual(contract['contract']['observation_size'], 50)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'old.pt'
            torch.save({'format': 'mystic_sim_ppo_v2', 'metadata': {
                **contract, 'history_schema': 'mystic-observable-history-46-v1'}}, path)
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                load_checkpoint(path, contract)

    def test_seven_outputs_no_idle_and_history_action_alignment(self):
        from Training.Mystic_Sim.core import Agent
        env = make_env()
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=3)
        self.assertEqual(env.action_space.n, 7)
        for invalid in (7, None):
            before = env.world.time_ms
            with self.assertRaises(ValueError):
                env.step(invalid)
            self.assertEqual(env.world.time_ms, before)
        mlp = Agent(env.observation_space.high)
        self.assertEqual(mlp.actor(torch.tensor(obs)).shape[-1], 7)
        recurrent = self.agent()
        h = ObservableHistory.from_env(env)
        for action in range(7):
            h.update(obs, action, applied=True, blocked=False, cooldown_rejected=False)
            self.assertEqual(h.features[:7].sum(), 1)
            self.assertEqual(h.features[action], 1)
        logits, _, _, _ = recurrent.sequence(torch.tensor(h.encode(obs)).view(1,1,-1),
            recurrent.initial_state(1), torch.ones(1,1,dtype=torch.bool))
        self.assertEqual(logits.shape, (1,1,7))

    def test_evaluation_replay_and_rng_isolation(self):
        agent = self.agent()
        rng = torch.get_rng_state().clone()
        a = evaluate(agent, [20, 21], stochastic=True, factory=short_env)
        b = evaluate(agent, [20, 21], stochastic=True, factory=short_env)
        self.assertEqual(a['episodes'], b['episodes'])
        torch.testing.assert_close(rng, torch.get_rng_state())

    def test_train_checkpoint_resume_and_contract(self):
        config = PPOConfig(recurrent=True, total_timesteps=160, num_envs=2,
                           num_steps=8, num_minibatches=2, update_epochs=1, eval_episodes=1)
        # Real training and evaluation, with short evaluation episodes only.
        def quick_evaluate(agent, seeds, **kwargs):
            return evaluate(agent, seeds, factory=short_env, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch('Training.Mystic_Sim.train.evaluate', quick_evaluate):
            path = train(config, run_dir=Path(directory)/'run', device='cpu',
                         record_gameplay=False, stop_after_checkpoint=1)
            first = path/'checkpoints/checkpoint_01.pt'
            env = make_env()
            self.addCleanup(env.close)
            payload = load_checkpoint(first, metadata(env, recurrent=True))
            with self.assertRaises(ValueError):
                load_checkpoint(first, metadata(env))
            train(PPOConfig(), resume=first, device='cpu', record_gameplay=False)
            last = load_checkpoint(path/'checkpoints/checkpoint_10.pt', metadata(env, recurrent=True))
            self.assertEqual(last['global_step'], 160)
            for name in ('actor.lstm.weight_hh_l0', 'critic.lstm.weight_hh_l0'):
                self.assertFalse(torch.equal(payload['agent'][name], last['agent'][name]))
            self.assertEqual(len(list((path/'evaluation').glob('*.json'))), 10)


if __name__ == '__main__':
    unittest.main()
