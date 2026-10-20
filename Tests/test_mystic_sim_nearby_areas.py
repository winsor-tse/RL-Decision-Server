"""Whole-area curriculum placement tests without training."""
from collections import Counter
from dataclasses import asdict, replace
import unittest

import numpy as np

from Custom_enviornments.Mystic_Sim.encounters import reachable_distances
from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
from Custom_enviornments.Mystic_Sim.scaled_calcs import scale_innie
from Training.Mystic_Sim.core import metadata
from Training.Mystic_Sim.curriculum import encounter_factory, checkpoint_factory
from Training.Mystic_Sim.curriculum_run import load_run_config


class NearbyAreasTests(unittest.TestCase):
    def test_random_areas_are_distinct_mapwide_and_reproducible(self):
        from Training.Mystic_Sim.nearby_areas import random_spawn_areas, nearest_spawn_areas
        encounter=replace(self.run.encounter, placement='random_spawn_areas_single_v1',
                          spawn_area_count=20, enemy_count=20)
        env=encounter_factory(encounter)();self.addCleanup(env.close)
        original=env.map_definition
        placement_map=replace(original,spawn_boxes=tuple(replace(b,quantity=1) for b in original.spawn_boxes))
        start=(50,45)
        nearest={b.object_id for b in nearest_spawn_areas(placement_map,start,20)}
        selections=[]
        for seed in range(10):
            selected=random_spawn_areas(placement_map,start,20,np.random.default_rng(seed))
            ids={b.object_id for b in selected}
            self.assertEqual(len(ids),20)
            self.assertTrue(ids.issubset({b.object_id for b in original.spawn_boxes}))
            self.assertEqual(selected,random_spawn_areas(placement_map,start,20,np.random.default_rng(seed)))
            selections.append(ids)
        self.assertTrue(any(ids!=nearest for ids in selections))
        self.assertGreater(len({frozenset(ids) for ids in selections}),1)
        obs,info=env.reset(seed=28000)
        counts=Counter(m.spawn_box.object_id for m in env.world.monsters.values())
        self.assertEqual(len(counts),20)
        self.assertEqual(set(counts.values()),{1})
        for m in env.world.monsters.values():
            self.assertEqual(m.stats,scale_innie(env.config.innie))
        npc=next(iter(env.world.monsters.values()));box=npc.spawn_box
        env.engine.handle_death(npc,env.world.player.entity_id)
        self.assertEqual(npc.respawn_at_ms,50000)
        env.world.time_ms=50000;env.engine.respawn(npc)
        self.assertEqual(npc.spawn_box,box)
        self.assertEqual(len(env.world.monsters),20)
        restored=checkpoint_factory({'metadata':metadata(env,recurrent=True)})()
        self.addCleanup(restored.close)
        replay,replay_info=restored.reset(seed=28000)
        np.testing.assert_array_equal(obs,replay)
        self.assertEqual(info['npc_spawns'],replay_info['npc_spawns'])
        self.assertEqual(info['selected_spawn_area_ids'],replay_info['selected_spawn_area_ids'])

    def setUp(self):
        self.run = load_run_config('Training/Mystic_Sim/configs/timeout25_20v1_nearest10_areas.json')
        self.env = encounter_factory(self.run.encounter)()
        self.addCleanup(self.env.close)

    def test_nearest_complete_areas_and_unchanged_stats(self):
        env = self.env
        original_map, original_config = env.map_definition, env.config
        normal = scale_innie(env.config.innie)
        for seed in (42, 20000, 26000):
            obs, info = env.reset(seed=seed)
            p = env.world.player
            distances = reachable_distances(original_map, (p.x,p.y))
            ranked = []
            for box in original_map.spawn_boxes:
                eligible = [distances[(x,y)] for y in range(box.y,box.y+box.height)
                            for x in range(box.x,box.x+box.width)
                            if (x,y) in distances and (x,y)!=(p.x,p.y)]
                if len(eligible) >= box.quantity:
                    ranked.append((min(eligible),box.object_id))
            expected = [i for _,i in sorted(ranked)[:10]]
            self.assertEqual(info['selected_spawn_area_ids'],expected)
            counts = Counter(m.spawn_box.object_id for m in env.world.monsters.values())
            self.assertEqual(counts,Counter({i:2 for i in expected}))
            self.assertEqual(len(env.world.occupancy),21)
            self.assertIs(env.config,original_config)
            self.assertIs(env.engine.config,original_config)
            self.assertIs(env.world.map,original_map)
            for m in env.world.monsters.values():
                self.assertEqual(m.stats,normal)
                b=m.spawn_box
                self.assertTrue(b.x<=m.x<b.x+b.width and b.y<=m.y<b.y+b.height)
                self.assertNotIn((m.x,m.y),original_map.blocked_cells)
            env.step(0)

    def test_reset_replay_checkpoint_reconstruction_and_isolation(self):
        first, info = self.env.reset(seed=42)
        self.env.step(0)
        again, info_again = self.env.reset(seed=42)
        np.testing.assert_array_equal(first,again)
        self.assertEqual(info['npc_spawns'],info_again['npc_spawns'])
        self.assertEqual(info['npc_move_intervals_ms'],info_again['npc_move_intervals_ms'])
        payload={'metadata':metadata(self.env,recurrent=True)}
        restored=checkpoint_factory(payload)()
        self.addCleanup(restored.close)
        other, other_info=restored.reset(seed=42)
        np.testing.assert_array_equal(first,other)
        self.assertEqual(info['npc_spawns'],other_info['npc_spawns'])
        full=MysticSimEnv();self.addCleanup(full.close);full.reset(seed=42)
        self.assertEqual(len(full.world.monsters),80)
        self.assertEqual(len(full.map_definition.spawn_boxes),40)

    def test_respawn_keeps_original_area_and_fifty_seconds(self):
        env=self.env;env.reset(seed=42)
        npc=next(iter(env.world.monsters.values()));box=npc.spawn_box
        env.engine.handle_death(npc,env.world.player.entity_id)
        self.assertEqual(npc.respawn_at_ms,50000)
        env.world.time_ms=50000
        env.engine.respawn(npc)
        self.assertEqual(npc.spawn_box,box)
        self.assertTrue(box.x<=npc.x<box.x+box.width and box.y<=npc.y<box.y+box.height)
        self.assertEqual(npc.hp,npc.max_hp)

    def test_rejects_partial_areas_and_cell_distance_cap(self):
        for change in ({'enemy_count':19},{'path_distance':(3,12)}, {'placement':'unknown'}):
            with self.assertRaises(ValueError):replace(self.run.encounter,**change)

    def test_single_population_coverage_replay_and_respawn(self):
        for count in (20,30,40):
            encounter=replace(self.run.encounter, placement='nearest_spawn_areas_single_v1',
                              spawn_area_count=count, enemy_count=count)
            env=encounter_factory(encounter)();self.addCleanup(env.close)
            obs,info=env.reset(seed=26000)
            counts=Counter(m.spawn_box.object_id for m in env.world.monsters.values())
            self.assertEqual(len(counts),count)
            self.assertEqual(set(counts.values()),{1})
            if count==40:
                self.assertEqual(set(counts),{b.object_id for b in env.map_definition.spawn_boxes})
            self.assertTrue(all(b.quantity==2 for b in env.map_definition.spawn_boxes))
            npc=next(iter(env.world.monsters.values()));box=npc.spawn_box
            env.engine.handle_death(npc,env.world.player.entity_id)
            self.assertEqual(npc.respawn_at_ms,50000)
            env.world.time_ms=50000;env.engine.respawn(npc)
            self.assertEqual(len(env.world.monsters),count)
            self.assertEqual(npc.spawn_box,box)
            self.assertTrue(box.x<=npc.x<box.x+box.width and box.y<=npc.y<box.y+box.height)
            again,again_info=env.reset(seed=26000)
            np.testing.assert_array_equal(obs,again)
            self.assertEqual(info['npc_spawns'],again_info['npc_spawns'])
            restored=checkpoint_factory({'metadata':metadata(env,recurrent=True)})()
            self.addCleanup(restored.close)
            _,restored_info=restored.reset(seed=26000)
            self.assertEqual(info['npc_spawns'],restored_info['npc_spawns'])


if __name__ == '__main__':
    unittest.main()
