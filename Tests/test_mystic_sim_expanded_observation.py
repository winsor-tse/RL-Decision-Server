"""Geometry, slot alignment, padding and policy input contract regression tests."""
import copy
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
from Custom_enviornments.Mystic_Sim.observation import encode_observation, nearest_monsters
from Training.Mystic_Sim.core import Agent, load_checkpoint, metadata
from Training.Mystic_Sim.history import ObservableHistory, observation_high
from Training.Mystic_Sim.recurrent import RecurrentAgent


class ExpandedObservationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.env = MysticSimEnv()
        self.addCleanup(self.env.close)
        self.env.reset(seed=42)
        self.world = self.env.world
        self.world.player.x = self.world.player.y = 50
        self.template = copy.deepcopy(next(iter(self.world.monsters.values())))

    def place(self, offsets):
        self.world.monsters.clear()
        for i, (dx, dy) in enumerate(offsets):
            npc = copy.deepcopy(self.template)
            npc.entity_id = i + 2
            npc.x, npc.y = 50 + dx, 50 + dy
            self.world.monsters[npc.entity_id] = npc

    def test_ten_slots_and_five_aligned_offsets_with_ties_and_dead_exclusion(self):
        self.place([(3, 2), (3, -2), (-1, 0), (0, -1), (0, 2), (2, 2),
                    (4, 3), (5, 3), (6, 3), (7, 3), (8, 3), (0, 0)])
        self.world.monsters[13].hp = 0
        self.world.monsters = dict(reversed(list(self.world.monsters.items())))
        selected = nearest_monsters(self.world)
        self.assertEqual([m.entity_id for m in selected], [4, 5, 6, 7, 2, 3, 8, 9, 10, 11])
        obs = encode_observation(self.world)
        self.assertEqual(obs.shape, (56,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(self.env.observation_space.contains(obs))
        np.testing.assert_array_equal(obs[46:], [-1, 0, 0, -1, 0, 2, 2, 2, 3, 2])
        # These two geometries have identical old positional encoding.
        np.testing.assert_array_equal(obs[22:24], obs[26:28])
        self.assertEqual(obs[42], 10)

    def test_signed_geometry_disambiguates_and_missing_slots_are_zero(self):
        self.place([(3, 2), (3, -2)])
        obs = encode_observation(self.world)
        np.testing.assert_array_equal(obs[6:8], obs[10:12])
        np.testing.assert_array_equal(obs[46:50], [3, 2, 3, -2])
        self.assertFalse(obs[14:46].any())
        self.assertFalse(obs[50:].any())
        self.world.monsters.clear()
        self.assertFalse(encode_observation(self.world)[6:].any())

    def test_bounds_normalization_and_both_policy_paths(self):
        self.place([(-50, -50), (49, 49)])
        obs = encode_observation(self.world)
        self.assertTrue(self.env.observation_space.contains(obs))
        np.testing.assert_array_equal(self.env.observation_space.low[46:], [-99] * 10)
        agent = Agent(self.env.observation_space.high)
        normalized = agent.normalized(torch.tensor(obs))
        self.assertLess(float(normalized[48]), 0)
        self.assertTrue(torch.all(normalized.abs() <= 1))
        action, _, _, value = agent.action_value(torch.tensor(obs))
        self.assertTrue(torch.isfinite(value))
        history = ObservableHistory.from_env(self.env)
        encoded = history.encode(obs)
        self.assertEqual(encoded.shape, (80,))
        recurrent = RecurrentAgent(observation_high(self.env.observation_space.high))
        with torch.no_grad():
            logits, values, _, _ = recurrent.sequence(
                torch.tensor(encoded).reshape(1, 1, 80), recurrent.initial_state(1),
                torch.ones((1, 1), dtype=torch.bool))
        self.assertEqual(tuple(logits.shape), (1, 1, 7))
        self.assertTrue(torch.isfinite(values).all())

    def test_old_contract_is_rejected(self):
        current = metadata(self.env, recurrent=True)
        old = copy.deepcopy(current)
        old['contract']['observation_size'] = 50
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.pt'
            torch.save({'format': 'mystic_sim_ppo_v2', 'metadata': old}, path)
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                load_checkpoint(path, current)
