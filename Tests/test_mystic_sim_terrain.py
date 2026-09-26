from dataclasses import replace
import unittest

from Custom_enviornments.Mystic_Sim.config import ScenarioConfig
from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
from Custom_enviornments.Mystic_Sim.movement import OFFSETS, legal_move, next_step_basic
from Custom_enviornments.Mystic_Sim.state import Direction


class TerrainTests(unittest.TestCase):
    def scene(self, seed=42):
        env = MysticSimEnv(config=ScenarioConfig(profile='map53_blocked_v2',terrain_collision=True))
        env.reset(seed=seed)
        self.addCleanup(env.close)
        return env

    def test_reset_and_episode_never_enter_blocked_tiles(self):
        for seed in range(8):
            env=self.scene(seed)
            for step in range(100):
                world=env.world
                for entity in (world.player,*world.monsters.values()):
                    self.assertNotIn((entity.x,entity.y),world.map.blocked_cells)
                    self.assertTrue(0 <= entity.x < 100 and 0 <= entity.y < 100)
                env.step(step%8)
                if env.episode_done:
                    break

    def test_players_and_npcs_cannot_cross_terrain_edges(self):
        env=self.scene()
        w=env.world
        for direction,(dx,dy) in OFFSETS.items():
            start=next((x,y) for y in range(1,99) for x in range(1,99)
                       if (x,y) not in w.map.blocked_cells and (x+dx,y+dy) in w.map.blocked_cells)
            for entity in (w.player,w.monsters[2]):
                entity.x,entity.y=start
                w.occupancy={start:entity.entity_id}
                before=dict(w.occupancy)
                self.assertFalse(env.engine.move(entity,direction,npc=entity is not w.player))
                self.assertEqual((entity.x,entity.y),start)
                self.assertEqual(w.occupancy,before)
                self.assertFalse(legal_move(w,start[0]+dx,start[1]+dy))

    def test_pathfinding_and_spawn_return_respect_walls(self):
        env=self.scene()
        w=env.world
        npc=w.monsters[2]
        npc.x,npc.y=50,50
        w.occupancy={(50,50):2}
        w.map=replace(w.map,blocked_cells=frozenset({(51,50),(10,10)}))
        direction=next_step_basic(w,npc,52,50,env.engine.rng)
        self.assertNotEqual(direction,Direction.RIGHT)
        npc.spawn_x,npc.spawn_y=10,10
        npc.aggro_target=None
        npc.last_aggro_update_ms=0
        w.time_ms=300000
        env.engine.npc_movement(npc)
        self.assertEqual((npc.x,npc.y),(50,50))

    def test_respawn_waits_for_walkable_cell(self):
        env=self.scene()
        w=env.world
        npc=w.monsters[2]
        env.engine.apply_damage(w.player,npc,npc.hp)
        box=npc.spawn_box
        cells={(x,y) for y in range(box.y,box.y+box.height) for x in range(box.x,box.x+box.width)}
        free=next(cell for cell in cells if cell not in w.occupancy and cell not in w.map.blocked_cells)
        w.map=replace(w.map,blocked_cells=w.map.blocked_cells | cells)
        w.events.clear()
        w.time_ms=50000
        env.engine.respawn(npc)
        self.assertFalse(npc.alive)
        self.assertEqual(npc.respawn_at_ms,50200)
        w.map=replace(w.map,blocked_cells=w.map.blocked_cells - {free})
        env.engine.scheduler.advance(50200,env.engine.dispatch)
        self.assertTrue(npc.alive)
        self.assertEqual((npc.x,npc.y),free)

    def test_old_open_profile_retains_terrain_behavior(self):
        env=MysticSimEnv()
        self.addCleanup(env.close)
        env.reset(seed=42)
        cell=next(c for c in env.world.map.blocked_cells if c not in env.world.occupancy)
        self.assertTrue(legal_move(env.world,*cell))
        with self.assertRaises(ValueError):
            ScenarioConfig(profile='map53_blocked_v2',terrain_collision=False)


if __name__=='__main__':
    unittest.main()
