"""Playable adapter tests; graphical dependency remains optional."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from Custom_enviornments.Mystic_Sim.viewer import PlaySession


class ViewerTests(unittest.TestCase):
    def test_pause_idle_and_replay(self):
        a,b=PlaySession(42),PlaySession(42)
        self.addCleanup(a.close);self.addCleanup(b.close)
        self.assertIsNone(a.step(5))
        self.assertEqual(a.env.world.time_ms,0)
        a.paused=b.paused=False
        for action in (None,0,3,5,6,7,None):
            self.assertEqual(a.step(action),b.step(action))
        self.assertEqual(a.env.world,b.env.world)
        self.assertEqual(a.env.world.time_ms,1400)

    def test_free_play_and_training_rules(self):
        for training in (False,True):
            session=PlaySession(training_rules=training)
            self.addCleanup(session.close)
            session.paused=False
            session.env.world.events.clear()
            session.env.world.step_count=255
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
