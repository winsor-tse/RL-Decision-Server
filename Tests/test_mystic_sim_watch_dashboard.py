"""Watch telemetry must stay aligned with inference and never affect playback."""
from copy import deepcopy
import importlib.util
import os
import unittest
from unittest.mock import patch

import numpy as np
import torch

from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
from Training.Mystic_Sim.core import Agent, metadata
from Training.Mystic_Sim.recurrent import RecurrentAgent
from Training.Mystic_Sim.watch import PolicySession, PolicyViewer


class WatchDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        with torch.random.fork_rng():
            torch.manual_seed(17)
            env = MysticSimEnv()
            try:
                cls.recurrent = RecurrentAgent(metadata(env, recurrent=True)['observation_scale']).eval()
                cls.feedforward = Agent(env.observation_space.high).eval()
            finally:
                env.close()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def session(self, recurrent=True):
        session = PolicySession(self.recurrent if recurrent else self.feedforward,
                                MysticSimEnv, mode='greedy')
        self.addCleanup(session.close)
        return session

    def test_exact_distribution_and_pre_action_state_are_retained(self):
        for recurrent in (True, False):
            with self.subTest(recurrent=recurrent):
                session = self.session(recurrent)
                before = session.snapshot
                tensor = torch.tensor(before.observation).unsqueeze(0)
                with torch.no_grad():
                    if recurrent:
                        logits, _, _, _ = session.agent.sequence(
                            tensor.unsqueeze(0), session.state, torch.tensor([[True]]))
                        expected = logits[0, 0].softmax(-1).numpy()
                    else:
                        expected = session.agent.actor(session.agent.normalized(tensor))[0].softmax(-1).numpy()
                session.paused = False
                session.step()
                self.assertIs(session.decision.input_snapshot, before)
                np.testing.assert_allclose(session.decision.probabilities, expected)
                self.assertAlmostEqual(sum(session.decision.probabilities), 1, places=6)
                self.assertEqual(session.decision.action, int(expected.argmax()))
                self.assertEqual(session.snapshot.step, before.step + 1)
                self.assertEqual(session.snapshot.time_ms, before.time_ms + 200)
                self.assertEqual(len(before.observation), 80 if recurrent else 56)
                np.testing.assert_allclose(before.normalized_observation,
                                           np.array(before.observation) / session.agent.observation_scale.numpy())
                self.assertTrue(0 <= session.decision.entropy_bits <= np.log2(7) + 1e-6)

    def test_outcomes_history_and_reset_do_not_leak_previous_decision(self):
        session = self.session()
        self.assertIsNone(session.decision)
        session.paused = False
        info = session.step()
        current = session.snapshot
        self.assertEqual(current.last_action, session.decision.action)
        self.assertEqual(current.action_applied, info['action_applied'])
        self.assertEqual(current.history[session.last_action], 1)
        self.assertEqual(current.history[7], bool(info['action_applied']))
        self.assertEqual(current.observation[56:], current.history)
        self.assertEqual(current.history[7:10],
                         (float(bool(info['action_applied'])), float(info['collision_kind'] is not None),
                          float(info['action_failure_reason'] in ('cooldown', 'family_cooldown', 'global_cooldown'))))
        # Retained inspection data is detached from the world and future resets.
        old_hp = current.player['hp']
        current.player['hp'] = 0
        self.assertEqual(session.env.world.player.hp, old_hp)
        session.reset()
        self.assertIsNone(session.decision)
        self.assertIsNone(session.last_action)
        self.assertEqual(session.snapshot.step, 0)
        self.assertEqual(session.snapshot.history[:16], (0,) * 16)

    @unittest.skipUnless(importlib.util.find_spec('pygame'), 'Optional pygame dependency missing')
    def test_render_navigation_and_resize_do_not_advance_policy_or_rng(self):
        with patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy', 'PYGAME_HIDE_SUPPORT_PROMPT': '1'}):
            import pygame
            session = self.session()
            viewer = PolicyViewer(pygame, session)
            try:
                viewer.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n, mod=0))
                self.assertTrue(session.paused)
                self.assertEqual(session.snapshot.step, 1)
                world = deepcopy(session.env.world)
                memory = tuple(s.clone() for s in session.state)
                sampling_rng = deepcopy(session.rng.bit_generator.state)
                torch_rng = torch.get_rng_state().clone()
                decision = session.decision
                for size in ((1480, 920), (1000, 760)):
                    viewer.handle_event(pygame.event.Event(pygame.VIDEORESIZE, w=size[0], h=size[1]))
                    self.assertFalse(viewer.view.colliderect(viewer.inspector))
                    self.assertFalse(viewer.view.colliderect(viewer.probability_panel))
                    for name, rect in viewer.tab_rects():
                        viewer.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rect.center))
                        self.assertEqual(viewer.tab, name)
                        for state_view, _, state_rect in viewer.state_rects():
                            viewer.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                                                  button=1, pos=state_rect.center))
                            self.assertEqual(viewer.state_view, state_view)
                            viewer.draw()
                            viewer.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_PAGEDOWN, mod=0))
                            viewer.draw()
                            self.assertLessEqual(viewer.scroll, viewer.max_scroll)
                old_zoom = viewer.zoom
                with patch.object(pygame.mouse, 'get_pos', return_value=viewer.inspector.center):
                    viewer.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, y=-1))
                self.assertEqual(viewer.zoom, old_zoom)
                self.assertEqual(session.env.world, world)
                self.assertIs(session.decision, decision)
                self.assertEqual(session.rng.bit_generator.state, sampling_rng)
                self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
                for old, new in zip(memory, session.state):
                    self.assertTrue(torch.equal(old, new))
                # The dashboard reads neither the model nor engine in its panels.
                with patch.object(session.agent, 'sequence', side_effect=AssertionError('render inferred')):
                    viewer.draw()
                viewer.restart()
                viewer.draw()
                self.assertIsNone(session.decision)
            finally:
                pygame.quit()


if __name__ == '__main__':
    unittest.main()
