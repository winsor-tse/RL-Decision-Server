from dataclasses import FrozenInstanceError, asdict, replace
import hashlib
import unittest

import numpy as np

from Custom_enviornments.Mystic_Sim.config import EncounterConfig, ScenarioConfig
from Custom_enviornments.Mystic_Sim.encounters import reachable_distances, scenario_for_encounter
from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
from Training.Mystic_Sim.curriculum import EncounterFactory, make_full_evaluation_env
from Training.Mystic_Sim.core import SimBatch


class EncounterTests(unittest.TestCase):
    def test_full_reset_prechange_fingerprint(self):
        env = make_full_evaluation_env()
        env.reset(seed=42)
        w = env.world
        digest = hashlib.sha256(repr((w.player, w.monsters, w.events,
                                     env.np_random.bit_generator.state)).encode()).hexdigest()
        self.assertEqual(digest, 'eed60bb4710771f8638a09a37b08eb4452edff6412826e325a71b945fedb8fd8')
        self.assertEqual(len(w.monsters), 80)
        self.assertEqual(env.config.reward.win_kills, 5)
        self.assertEqual(env.config.reward.max_episode_steps, 1024)

    def test_contract_is_immutable_and_only_encounter_changes(self):
        encounter = EncounterConfig('one', 1, 1, (2, 12))
        with self.assertRaises(FrozenInstanceError):
            encounter.enemy_count = 2
        actual = asdict(scenario_for_encounter(encounter))
        expected = asdict(ScenarioConfig())
        actual.pop('encounter'); expected.pop('encounter')
        actual['reward']['win_kills'] = 5
        self.assertEqual(actual, expected)
        for args in [('full', 1, 1), ('', 1, 1), ('x', 0, 1), ('x', 81, 1),
                     ('x', 1, 2), ('x', True, 1), ('x', 1, 1, (0, 2)),
                     ('x', 1, 1, (4, 2))]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                EncounterConfig(*args)
        with self.assertRaises(ValueError):
            replace(ScenarioConfig(), encounter=encounter)

    def test_legal_seeded_encounters_and_isolation(self):
        factory = EncounterFactory(EncounterConfig('three', 3, 3, (2, 12)))
        a, b = factory(), factory()
        full = make_full_evaluation_env()
        signatures = set()
        for seed in range(12):
            oa, ia = a.reset(seed=seed)
            ob, ib = b.reset(seed=seed)
            np.testing.assert_array_equal(oa, ob)
            self.assertEqual(ia, ib)
            self.assertEqual(a.map_definition, full.map_definition)
            w = a.world
            distances = reachable_distances(w.map, (w.player.x, w.player.y))
            self.assertEqual(len(w.monsters), 3)
            self.assertEqual(len(w.occupancy), 4)
            self.assertEqual(len(w.events), 4)
            self.assertEqual({e.entity_id for e in w.events}, {1, 2, 3, 4})
            for m in w.monsters.values():
                self.assertTrue(m.spawn_box.contains(m.x, m.y))
                self.assertTrue(2 <= distances[(m.x, m.y)] <= 12)
                self.assertNotIn((m.x, m.y), w.map.blocked_cells)
            signatures.add(tuple((m.x, m.y) for m in w.monsters.values()))
            ia['encounter']['enemy_count'] = 99
            self.assertEqual(a.config.encounter.enemy_count, 3)
            for action in [4, 0, 1, 5, 2, 6]:
                ra, rb = a.step(action), b.step(action)
                np.testing.assert_array_equal(ra[0], rb[0])
                self.assertEqual(ra[1:], rb[1:])
                if ra[2] or ra[3]:
                    break
        self.assertGreater(len(signatures), 1)
        self.assertEqual(full.config, ScenarioConfig())

    def test_impossible_encounter_fails_clearly(self):
        env = EncounterFactory(EncounterConfig('impossible', 80, 5, (1, 1)))()
        with self.assertRaisesRegex(ValueError, 'cannot place 80'):
            env.reset(seed=1)

    def test_existing_recurrent_batch_accepts_factory(self):
        batch = SimBatch(2, seed=42,
                         factory=EncounterFactory(EncounterConfig('two', 2, 2)),
                         recurrent=True)
        try:
            self.assertEqual(batch.observations.shape, (2, 80))
            self.assertIsNot(batch.envs[0].world, batch.envs[1].world)
            batch.step(np.array([4, 0]))
            self.assertTrue(all(len(e.world.monsters) == 2 for e in batch.envs))
        finally:
            batch.close()

    def test_selected_npc_uses_normal_respawn_and_stage_win(self):
        env = EncounterFactory(EncounterConfig('one', 1, 1, (2, 12)))()
        env.reset(seed=42)
        w = env.world
        npc = w.monsters[2]
        env.engine.apply_damage(w.player, npc, npc.hp)
        self.assertEqual(npc.respawn_at_ms, 50000)
        _, _, terminated, truncated, info = env.advance_idle()
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['episode_outcome'], 'win')
        # Direct scheduler exercise past termination verifies shared respawn mechanics.
        env.engine.scheduler.advance(50000, env.engine.dispatch)
        self.assertTrue(npc.alive)
        self.assertEqual(len(w.monsters), 1)
        self.assertTrue(npc.spawn_box.contains(npc.x, npc.y))
        self.assertEqual(npc.hp, npc.max_hp)
        self.assertTrue(900 <= npc.move_interval_ms <= 1100)
        env.reset(seed=42)
        self.assertEqual(env.world.kills, 0)
        self.assertEqual(env.world.step_count, 0)
        self.assertEqual(len(env.world.monsters), 1)


if __name__ == '__main__':
    unittest.main()
