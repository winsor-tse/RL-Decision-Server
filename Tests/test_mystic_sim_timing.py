from dataclasses import replace
import copy
import unittest
from unittest.mock import patch
from pathlib import Path

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
from numpy.testing import assert_array_equal

from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
from Custom_enviornments.Mystic_Sim.config import ScenarioConfig, PlayerConfig
from Custom_enviornments.Mystic_Sim.engine import Engine, TracedRng
from Custom_enviornments.Mystic_Sim.movement import next_step_basic, outside_spawn_area
from Custom_enviornments.Mystic_Sim.scheduler import Scheduler
from Custom_enviornments.Mystic_Sim.state import Direction, TimedEffect


class TimingTests(unittest.TestCase):
    def scene(self):
        env = MysticSimEnv(trace=True)
        env.reset(seed=123)
        world = env.world
        npc = world.monsters[2]
        world.monsters = {2: npc}
        world.events.clear()
        world.player.x, world.player.y = 50, 50
        npc.x, npc.y = 45, 50
        npc.spawn_x, npc.spawn_y = 45, 50
        npc.spawn_box = replace(npc.spawn_box, x=40, y=40)
        world.occupancy = {(50,50):1, (45,50):2}
        env.engine = Engine(world, env.config, env.np_random, trace=True)
        return env, world, npc

    def test_heap_boundaries_equal_times_nested_and_cancel(self):
        env, w, npc = self.scene()
        scheduler = Scheduler(w)
        seen = []
        scheduler.schedule(175, "first", 1)
        token = scheduler.schedule(175, "cancelled", 1)
        scheduler.schedule(175, "second", 1)
        scheduler.schedule(201, "later", 1)
        scheduler.cancel(token)
        def handler(event):
            seen.append((w.time_ms, event.kind))
            if event.kind == "first":
                scheduler.schedule(175, "nested", 1)
        scheduler.advance(174, handler)
        self.assertEqual(seen, [])
        scheduler.advance(200, handler)
        self.assertEqual(seen, [(175,"first"),(175,"second"),(175,"nested")])
        self.assertEqual(w.time_ms, 200)
        scheduler.advance(201, handler)
        self.assertEqual(seen[-1], (201,"later"))
        with self.assertRaises(ValueError):
            scheduler.schedule(200,"past",1)
        with self.assertRaises(ValueError):
            scheduler.advance(200, handler)

    def test_same_timestamp_loop_guard(self):
        env, w, npc = self.scene()
        scheduler = Scheduler(w, max_events_at_timestamp=5)
        scheduler.schedule(1, "loop", 1)
        with self.assertRaisesRegex(RuntimeError, "rescheduling loop"):
            scheduler.advance(200, lambda e: scheduler.schedule(e.due_ms, "loop", 1))

    def test_player_movement_occupancy_bounds_facing(self):
        env, w, npc = self.scene()
        w.occupancy[(49,50)] = 999
        obs, reward, term, trunc, info = env.step(2)
        self.assertFalse(info["action_applied"])
        self.assertEqual(w.player.facing, Direction.LEFT)
        self.assertEqual((w.player.x,w.player.y),(50,50))
        env.step(0)
        self.assertEqual((w.player.x,w.player.y),(50,49))
        self.assertNotIn((50,50),w.occupancy)
        self.assertEqual(w.occupancy[(50,49)],1)
        w.occupancy.pop((50,49))
        w.player.x,w.player.y=0,0
        w.occupancy[(0,0)]=1
        self.assertFalse(env.step(2)[4]["action_applied"])
        self.assertFalse(env.step(0)[4]["action_applied"])
        self.assertEqual(w.time_ms,800)

    def test_source_pursuit_roll_order_and_obstacle_branches(self):
        env,w,npc=self.scene()
        class Draws:
            def __init__(self, coin=1, axis=True): self.calls=[]; self.coin=coin; self.axis=axis
            def roll(self,low,high,purpose): self.calls.append((purpose,low,high)); return self.coin
            def chance(self,p,purpose): self.calls.append((purpose,p)); return self.axis
        rng=Draws()
        self.assertEqual(next_step_basic(w,npc,46,50,rng),Direction.RIGHT)
        self.assertEqual(rng.calls,[])
        self.assertEqual(next_step_basic(w,npc,48,52,rng),Direction.RIGHT)
        self.assertEqual(rng.calls,[('path_coin',0,1),('path_axis',16/25)])
        rng=Draws(axis=False)
        self.assertEqual(next_step_basic(w,npc,48,52,rng),Direction.DOWN)
        w.occupancy[(46,50)]=99
        rng=Draws()
        self.assertEqual(next_step_basic(w,npc,48,50,rng),Direction.UP)
        self.assertEqual(len(rng.calls),2)  # weighted draw is still consumed
        for cell in [(44,50),(45,49),(45,51)]: w.occupancy[cell]=99
        rng=Draws(coin=0)
        self.assertEqual(next_step_basic(w,npc,48,50,rng),Direction.LEFT)
        self.assertEqual(rng.calls,[('path_coin',0,1),('path_fallback',0,3)])

    def test_rollchance_inclusive_endpoints(self):
        records=[]
        class Stub:
            def integers(self,low,high):
                self.range=(low,high)
                return self.value
        stub=Stub(); rng=TracedRng(stub, lambda *a,**kw: records.append((a,kw)))
        stub.value=0
        self.assertTrue(rng.chance(0,"zero"))
        self.assertEqual(stub.range,(0,1000000001))
        stub.value=1000000000
        self.assertTrue(rng.chance(1,"one"))
        self.assertFalse(rng.chance(.999,"below"))

    def test_move_timers_blocked_and_adjacent_face(self):
        env,w,npc=self.scene(); engine=env.engine
        npc.aggro_target=1
        npc.move_interval_ms=937
        w.time_ms=1000
        engine.move(npc,Direction.RIGHT,npc=True)
        self.assertEqual((npc.next_move_ms,npc.next_attack_ms),(1937,2500))
        self.assertEqual(npc.move_interval_ms,937)
        w.time_ms=1100
        w.occupancy[(47,50)]=99
        engine.move(npc,Direction.RIGHT,npc=True)
        self.assertEqual((npc.x,npc.y,npc.next_move_ms,npc.next_attack_ms),(46,50,2037,2500))
        w.player.x,w.player.y=47,50
        engine.npc_movement(npc)
        self.assertEqual(npc.next_move_ms,2037)  # FaceEntity does not reset timer
        self.assertEqual(npc.facing,Direction.RIGHT)

    def test_aggro_boundaries_drop_damage_and_player_move(self):
        env,w,npc=self.scene()
        npc.next_move_ms=99999
        w.player.x=49  # distance exactly 4
        env.engine.scheduler.schedule(1500,"npc_update",2)
        env.engine.scheduler.advance(1499,env.engine.dispatch)
        self.assertIsNone(npc.aggro_target)
        env.engine.scheduler.advance(1500,env.engine.dispatch)
        self.assertEqual(npc.aggro_target,1)
        self.assertEqual(npc.next_attack_ms,2500)
        w.player.x=63 # exactly 18 retained
        env.engine.update_npc(npc)
        self.assertEqual(npc.aggro_target,1)
        w.player.x=64
        env.engine.update_npc(npc)
        self.assertIsNone(npc.aggro_target)
        env.engine.add_damage_aggro(2,1)
        self.assertEqual(npc.aggro_target,1)
        w.player.hp=0
        env.engine.update_npc(npc)
        self.assertIsNone(npc.aggro_target)
        env,w,npc=self.scene()
        env.step(2) # from x50 to49 triggers OnEntityMoved acquisition immediately
        self.assertEqual(npc.aggro_target,1)
        self.assertEqual(next(e['time_ms'] for e in env.engine.trace if e['kind']=='aggro_acquired'),0)

    def test_attack_trace_cardinal_only_and_damage(self):
        env,w,npc=self.scene()
        npc.aggro_target=1; npc.next_attack_ms=0; npc.next_move_ms=99999
        w.player.x,w.player.y=46,51
        env.engine.update_npc(npc)
        self.assertFalse(any(e['kind']=='npc_attack' for e in env.engine.trace))
        w.player.x,w.player.y=46,50
        env.engine.update_npc(npc)
        self.assertEqual(npc.next_attack_ms,1000)
        self.assertTrue(any(e['kind']=='npc_attack' for e in env.engine.trace))
        self.assertLess(w.player.hp,w.player.max_hp)

    def test_idle_random_and_source_spawn_boundary(self):
        env,w,npc=self.scene()
        npc.x,npc.y=50,50 # X+W remains inside under C# OutsideSpawnArea
        self.assertFalse(outside_spawn_area(npc))
        npc.x=51
        self.assertTrue(outside_spawn_area(npc))
        w.player.x,w.player.y=60,60
        w.occupancy={(npc.x,npc.y):2,(w.player.x,w.player.y):1}
        env.engine.npc_movement(npc)
        self.assertEqual(npc.x,50)
        self.assertFalse(any(e.get('purpose')=='idle_direction' for e in env.engine.trace))
        env,w,npc=self.scene()
        env.engine.npc_movement(npc)
        self.assertEqual([e['purpose'] for e in env.engine.trace if e['kind']=='rng'],['idle_direction'])

    def test_stale_life_events_and_trace_only_timing_hooks(self):
        env,w,npc=self.scene()
        env.engine.scheduler.schedule(1,'npc_update',2,99)
        env.engine.scheduler.advance(1,env.engine.dispatch)
        self.assertEqual((npc.x,npc.y),(45,50))
        env.engine.scheduler.schedule(2000,'regeneration',1)
        effect=TimedEffect('acid',1,2,1000,3000,1000)
        env.engine.schedule_effect(effect)
        npc.hp=0
        env.engine.schedule_respawn(2)
        env.engine.scheduler.advance(3000,env.engine.dispatch)
        ticks=[e['time_ms'] for e in env.engine.trace if e['kind']=='effect_tick']
        self.assertEqual(ticks,[1000,2000])
        self.assertEqual(w.effects,[])
        self.assertEqual(w.player.next_regen_ms,4000)
        env.engine.scheduler.advance(50000,env.engine.dispatch)
        self.assertFalse(any(e['kind']=='respawn_due' for e in env.engine.trace))
        env.engine.scheduler.advance(50001,env.engine.dispatch)
        self.assertTrue(any(e['kind']=='respawn' for e in env.engine.trace))
        self.assertTrue(npc.alive)

    def test_equal_time_occupancy_first_wins(self):
        env,w,npc=self.scene()
        other=copy.deepcopy(npc); other.entity_id=3; other.x=47
        w.monsters[3]=other; w.occupancy[(47,50)]=3
        env.engine.scheduler.schedule(100,'a',2)
        env.engine.scheduler.schedule(100,'b',3)
        def dispatch(e):
            env.engine.move(w.monsters[e.entity_id],Direction.RIGHT if e.entity_id==2 else Direction.LEFT,npc=True)
        env.engine.scheduler.advance(100,dispatch)
        self.assertEqual((npc.x,other.x),(46,47))
        self.assertEqual(w.occupancy[(46,50)],2)

    def test_movement_deadlines_are_not_rounded_to_decisions(self):
        for interval in (900,937,1100):
            env,w,npc=self.scene()
            npc.move_interval_ms=npc.next_move_ms=interval
            npc.next_aggro_ms=99999
            env.engine.queue_npc(npc)
            env.engine.scheduler.advance(interval-1,env.engine.dispatch)
            self.assertFalse(any(e['kind']=='movement' for e in env.engine.trace))
            env.engine.scheduler.advance(interval,env.engine.dispatch)
            moves=[e for e in env.engine.trace if e['kind']=='movement']
            self.assertEqual(len(moves),1)
            self.assertEqual(moves[0]['time_ms'],interval)
            self.assertEqual(npc.next_move_ms,interval*2)

    def test_euclidean_aggro_rejects_outside_radius(self):
        env,w,npc=self.scene()
        npc.next_move_ms=99999
        w.player.x,w.player.y=npc.x+3,npc.y+3
        w.time_ms=1500
        env.engine.update_npc(npc)
        self.assertIsNone(npc.aggro_target)
        self.assertEqual(npc.next_aggro_ms,3000)

    def test_idle_return_jump_respects_occupancy(self):
        env,w,npc=self.scene()
        w.occupancy.pop((npc.x,npc.y))
        npc.x,npc.y=10,10
        w.occupancy[(10,10)]=2
        w.time_ms=300000
        w.occupancy[(npc.spawn_x,npc.spawn_y)]=99
        env.engine.npc_movement(npc)
        self.assertEqual((npc.x,npc.y),(10,10))
        w.occupancy.pop((npc.spawn_x,npc.spawn_y))
        env.engine.npc_movement(npc)
        self.assertEqual((npc.x,npc.y),(45,50))
        self.assertEqual(w.occupancy[(45,50)],2)
        self.assertNotIn((10,10),w.occupancy)

    def test_replay_no_io_and_intervals_unchanged(self):
        config=ScenarioConfig(player=replace(PlayerConfig(),max_hp=10000000))
        a,b=MysticSimEnv(trace=True,config=config),MysticSimEnv(trace=True,config=config)
        a.reset(seed=7); b.reset(seed=7)
        intervals={i:m.move_interval_ms for i,m in a.world.monsters.items()}
        with patch.object(Path,'open',side_effect=AssertionError('file I/O')), \
             patch('builtins.open',side_effect=AssertionError('file I/O')), \
             patch('socket.socket',side_effect=AssertionError('socket')), \
             patch('time.sleep',side_effect=AssertionError('sleep')):
            for i in range(200):
                # This movement-only test requires all NPCs to remain alive.
                action=[0,2,1,3,4][i%5]
                ra,rb=a.step(action),b.step(action)
                assert_array_equal(ra[0],rb[0])
                self.assertEqual(ra[1:],rb[1:])
                self.assertEqual(len(a.world.occupancy),81)
                for m in a.world.monsters.values():
                    self.assertEqual(a.world.occupancy[(m.x,m.y)],m.entity_id)
            self.assertEqual(a.world.time_ms,40000)
        self.assertEqual(intervals,{i:m.move_interval_ms for i,m in a.world.monsters.items()})

    def test_gym_api_limits_invalid_actions_and_reset(self):
        env=MysticSimEnv(config=ScenarioConfig(player=replace(PlayerConfig(),max_hp=10000000)))
        with self.assertRaises(gym.error.ResetNeeded): env.step(0)
        check_env(env,skip_render_check=True)
        env.reset(seed=1)
        for invalid in [-1,8,True,1.5,'0']:
            with self.assertRaises(ValueError): env.step(invalid)
        self.assertEqual(env.world.time_ms,0)
        for _ in range(256): result=env.step(4)
        self.assertEqual(result[1:4],(0.0,False,True))
        with self.assertRaises(gym.error.ResetNeeded): env.step(0)
        env.reset(seed=1)
        self.assertEqual(env.world.time_ms,0)


if __name__=='__main__': unittest.main()
