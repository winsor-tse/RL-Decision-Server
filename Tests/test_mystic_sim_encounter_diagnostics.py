from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
import numpy as np

from Training.Mystic_Sim.encounter_diagnostics import EncounterDiagnostics
from Training.Mystic_Sim.curriculum_run import load_run_config
from Training.Mystic_Sim.curriculum import encounter_factory
from Training.Mystic_Sim.core import SimBatch
from Custom_enviornments.Mystic_Sim.encounters import reachable_distances
from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
from Custom_enviornments.Mystic_Sim.config import ScenarioConfig


class DiagnosticTests(unittest.TestCase):
    def test_batch_autoreset_preserves_final_metrics_and_clears_tracker(self):
        config = ScenarioConfig()
        config = replace(config, reward=replace(config.reward, max_episode_steps=1))
        batch = SimBatch(1, 42, factory=lambda: MysticSimEnv(config=config))
        self.addCleanup(batch.close)
        for _ in range(2):
            _, _, _, truncated, infos = batch.step([0])
            self.assertTrue(truncated[0])
            self.assertIn('combat_navigation', infos[0])
            self.assertEqual(batch.encounter_diagnostics[0].steps, 0)
            self.assertEqual(batch.envs[0].world.step_count, 0)

    def test_range_damage_reversal_and_empty_world(self):
        w = NS(player=NS(x=0, y=0, entity_id=1), time_ms=0,
               monsters={2: NS(x=16, y=10, alive=True)})
        tracker = EncounterDiagnostics(w)
        tracker.before(w); w.player.x=1; w.time_ms=200
        row = tracker.after(w, {'damage_events': []})
        self.assertEqual(row['enemy_in_cast_range_fraction'], 1)
        tracker.before(w); w.player.x=0; w.time_ms=400
        row = tracker.after(w, {'damage_events': [{'attacker_id': 1, 'target_id': 2, 'damage': 100}]})
        self.assertEqual(row['movement_reversal_fraction'], .5)
        self.assertEqual(row['successful_move_revisit_fraction'], .5)
        self.assertEqual(row['damage_per_in_range_second'], 250)
        self.assertEqual(row['seconds_since_damage'], 0)
        self.assertEqual(row['longest_no_damage_seconds'], .4)
        w.monsters[2].x = 17
        tracker.before(w); w.time_ms=600
        row = tracker.after(w, {'damage_events': []})
        self.assertAlmostEqual(row['outside_cast_range_fraction'], 1/3)
        w.monsters[2].alive=False
        tracker.before(w); w.time_ms=800
        row = tracker.after(w, {'damage_events': []})
        self.assertEqual(row['no_living_enemy_fraction'], .25)
        self.assertAlmostEqual(row['outside_cast_range_fraction'], 1/3)

    def test_tight_placement_and_diagnostics_do_not_change_replay(self):
        run=load_run_config('Training/Mystic_Sim/configs/three_innies_tight.json')
        factory=encounter_factory(run.encounter)
        env=factory(); self.addCleanup(env.close)
        for seed in range(30):
            env.reset(seed=seed)
            w=env.world; p=(w.player.x,w.player.y)
            distances=reachable_distances(w.map,p)
            self.assertEqual(len(w.monsters),3)
            for m in w.monsters.values():
                self.assertTrue(3 <= distances[(m.x,m.y)] <= 8)
            # Open cardinal neighbours allow immediate repositioning at reset.
            self.assertGreaterEqual(sum((p[0]+dx,p[1]+dy) not in w.map.blocked_cells
                                        for dx,dy in ((0,1),(0,-1),(1,0),(-1,0))),2)
        batch=SimBatch(1,42,factory=factory)
        self.addCleanup(batch.close)
        env.reset(seed=42)
        for action in (0,3,4,1,6,2,5):
            expected=env.step(action)
            actual=batch.step([action])
            np.testing.assert_array_equal(actual[0][0],expected[0])
            self.assertAlmostEqual(float(actual[1][0]),expected[1],places=5)
            self.assertEqual(actual[4][0]['damage_events'],expected[4]['damage_events'])
            self.assertIn('combat_navigation',actual[4][0])
            self.assertEqual(env.np_random.bit_generator.state,batch.envs[0].np_random.bit_generator.state)


if __name__ == '__main__':
    unittest.main()
