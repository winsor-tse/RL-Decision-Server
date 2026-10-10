"""Playable adapter tests; graphical dependency remains optional."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy

from Custom_enviornments.Mystic_Sim.viewer import PlaySession, Viewer
from Custom_enviornments.Mystic_Sim.state import TimedEffect


class ViewerTests(unittest.TestCase):
    def test_pause_idle_and_replay(self):
        a,b=PlaySession(42),PlaySession(42)
        self.addCleanup(a.close);self.addCleanup(b.close)
        self.assertIsNone(a.step(4))
        self.assertEqual(a.env.world.time_ms,0)
        a.paused=b.paused=False
        for action in (None,0,3,4,5,6,None):
            self.assertEqual(a.step(action),b.step(action))
        self.assertEqual(a.env.world,b.env.world)
        self.assertEqual(a.env.world.time_ms,1400)

    def test_free_play_and_training_rules(self):
        for training in (False,True):
            session=PlaySession(training_rules=training)
            self.addCleanup(session.close)
            session.paused=False
            session.env.world.events.clear()
            session.env.world.step_count=1023
            result=session.step()
            self.assertEqual(session.env.episode_done,training)
            self.assertEqual(result['episode_outcome'],'truncated' if training else None)
            if training:
                self.assertIsNone(session.step())
            session.reset()
            self.assertEqual(session.env.world.step_count,0)
            self.assertFalse(session.env.episode_done)

    def test_reset_seed_and_idle_does_not_cast(self):
        session=PlaySession(12)
        self.addCleanup(session.close)
        original=session.info['npc_spawns']
        session.paused=False
        result=session.step()
        self.assertEqual(result['cast_events'],[])
        self.assertEqual(session.env.world.player.mp,session.env.world.player.max_mp)
        session.reset()
        self.assertEqual(original,session.info['npc_spawns'])
        session.reset(new_seed=True)
        self.assertEqual(session.seed,13)
        self.assertNotEqual(original,session.info['npc_spawns'])
        self.assertEqual(session.total_reward,0)

    def test_default_viewer_truncates_at_environment_limit_and_stays_stopped(self):
        session=PlaySession()
        self.addCleanup(session.close)
        self.assertTrue(session.training_rules)
        self.assertEqual(session.env.reward_config.max_episode_steps,1024)
        session.env.world.events.clear()  # Isolate the time limit from combat.
        self.assertIsNone(session.step())  # Paused frames do not count.
        session.paused=False
        for step in range(1,1025):
            info=session.step()
            self.assertEqual(info['current_step'],step)
            self.assertEqual(session.env.episode_done,step==1024)
        self.assertEqual(info['episode_outcome'],'truncated')
        self.assertEqual(info['episode_end_reason'],'step_limit')
        self.assertIn('step limit',session.log[-1])
        for _ in range(3):
            self.assertIsNone(session.step())
        self.assertEqual(session.env.world.step_count,1024)
        self.assertEqual(session.env.world.time_ms,204800)
        session.reset()
        self.assertEqual(session.info['current_step'],0)
        self.assertIsNone(session.info['episode_end_reason'])
        self.assertTrue(session.paused)
        session.paused=False
        self.assertEqual(session.step()['current_step'],1)

    def test_viewer_restart_restores_entire_initial_world_and_clears_ui(self):
        session=PlaySession(42)
        self.addCleanup(session.close)
        initial=deepcopy(session.env.world)
        session.paused=False
        for action in (0,3,4,5,6):
            session.step(action)
        old_world, old_engine=session.env.world,session.env.engine
        npc=next(iter(old_world.monsters.values()))
        old_engine.apply_damage(old_world.player,npc,npc.hp)
        old_world.player.hp=1
        old_world.player.mp=0
        old_world.player.cooldowns.slots[1]=99999
        old_world.effects.append(TimedEffect('test',1,3,5000,9000,1000))
        old_world.player_collision_streak=3
        session.total_reward=-30
        session.env.episode_done=True
        # Exercise the keyboard's restart path without opening a display.
        viewer=Viewer.__new__(Viewer)
        viewer.session=session
        viewer.pending=3
        viewer.accumulator=.19
        viewer.flashes=[['old spell']]
        viewer.floating=[['old damage']]
        viewer.restart()
        self.assertIsNot(session.env.world,old_world)
        self.assertIsNot(session.env.engine,old_engine)
        self.assertEqual(session.env.world,initial)
        self.assertEqual(session.info['player_spawn'],(initial.player.x,initial.player.y))
        self.assertEqual(session.info['npc_spawns'],
                         {key:(npc.x,npc.y) for key,npc in initial.monsters.items()})
        self.assertEqual(session.total_reward,0)
        self.assertFalse(session.env.episode_done)
        self.assertTrue(session.paused)
        self.assertIsNone(session.step())
        self.assertIsNone(viewer.pending)
        self.assertEqual(viewer.accumulator,0)
        self.assertEqual(viewer.flashes,[])
        self.assertEqual(viewer.floating,[])
        self.assertEqual(session.env.engine.damage_events,[])
        self.assertEqual(session.env.engine.death_events,[])

    @unittest.skipUnless(importlib.util.find_spec('pygame'), 'Optional pygame viewer dependency not installed')
    def test_dummy_driver_renders_real_simulation(self):
        with tempfile.TemporaryDirectory() as temp:
            screenshot=Path(temp)/'frame.png'
            result=subprocess.run([sys.executable,'-m','Custom_enviornments.Mystic_Sim.viewer',
                                   '--smoke-test','--screenshot',str(screenshot)],
                                  capture_output=True,text=True,timeout=30,
                                  cwd=Path(__file__).resolve().parents[1])
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('40 steps, 8000 ms',result.stdout)
            self.assertEqual(screenshot.read_bytes()[:8],b'\x89PNG\r\n\x1a\n')


if __name__=='__main__':
    unittest.main()
