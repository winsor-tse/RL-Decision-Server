"""Estimated 80-input history vs simulator outcomes, with no server acknowledgements."""
from copy import deepcopy
from dataclasses import replace
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
from Custom_enviornments.Mystic_Sim.config import ScenarioConfig
from Training.Mystic_Sim.history import ObservableHistory
from Inference.Mystic_Live.simulated_feedback import SimulatedFeedbackInputs
from Inference.Mystic_Live.commands import LiveCommands, COMMAND_SCHEMA
from Inference.Mystic_Live.lifecycle import SingleRun
from Inference.Mystic_Live.policy import FrozenPolicy
from Inference.Mystic_Live import server
from Tests.test_mystic_live_history import project
from Tests.test_mystic_live_lifecycle import Clock, StubActor, request
from Tests import test_mystic_live_policy as policy_tests


class SimulatedFeedbackTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_full_tensor_matches_simulator_sequence(self):
        env = MysticSimEnv()
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=7)
        estimated = SimulatedFeedbackInputs(env.config)
        actual = ObservableHistory.from_env(env)
        np.testing.assert_array_equal(estimated.encode(project(env.world))['values'], actual.encode(obs))
        for action in (4, 5, 4, 6, 6, 0, 2, 3, 1, 5, 4):
            estimated.record_action(action)
            prediction = estimated.pending_estimate
            previous = obs
            obs, _, terminated, truncated, info = env.step(action)
            self.assertEqual(prediction['applied'], info['action_applied'])
            self.assertEqual(prediction['collision_kind'], info['collision_kind'])
            self.assertEqual(prediction['failure_reason'], info['action_failure_reason'])
            actual.update_from_info(previous, action, info, env.config.timing.step_ms)
            np.testing.assert_array_equal(estimated.encode(project(env.world))['values'], actual.encode(obs))
            if terminated or truncated:
                break

    def world(self, *, entities=None):
        world = deepcopy(request(0)['worldState'])
        world['timestamp'] = 10000
        world['player'].update(mapX=50, mapY=50)
        world['entities'] = [] if entities is None else entities
        return world

    def test_cooldown_order_elapsed_time_and_resynchronization(self):
        model = SimulatedFeedbackInputs(ScenarioConfig())
        world = self.world()
        model.encode(world)
        model.record_action(5)  # Self-target Acid, enough resources.
        self.assertTrue(model.pending_estimate['applied'])
        world['timestamp'] += 175
        observation = model.encode(world)['values']
        self.assertAlmostEqual(observation[67], .0175, places=7)
        self.assertEqual(observation[70], 1)
        model.record_action(6)
        self.assertEqual(model.pending_estimate['failure_reason'], 'global_cooldown')
        world['timestamp'] += 125
        observation = model.encode(world)['values']
        self.assertEqual(observation[65], 1)
        self.assertAlmostEqual(observation[67], .03, places=7)
        world['player']['mp'] = 0
        # The next input supplies the actual observed resource, not predicted consumption.
        model.record_action(6)
        self.assertTrue(model.pending_estimate['applied'])
        world['timestamp'] += 100000
        observation = model.encode(world)['values']
        self.assertEqual(observation[4], 0)
        self.assertEqual(observation[67], 1)
        model.record_action(5)
        self.assertEqual(model.pending_estimate['failure_reason'], 'insufficient_mp')

    def test_no_target_boundary_and_visible_occupancy(self):
        model = SimulatedFeedbackInputs(ScenarioConfig())
        world = self.world()
        model.encode(world)
        model.record_action(4)
        self.assertEqual(model.pending_estimate['failure_reason'], 'no_target')
        world['timestamp'] += 200
        world['player'].update(mapX=0, mapY=0)
        model.encode(world)
        model.record_action(2)
        self.assertEqual(model.pending_estimate['collision_kind'], 'map_boundary')
        world['timestamp'] += 200
        world['player'].update(mapX=50, mapY=50)
        # Select a terrain-free neighbor to isolate entity collision.
        x, y = next((x,y) for y in range(20,80) for x in range(20,80)
                    if (x+1,y) not in model.encoder.map.blocked_cells)
        world['player'].update(mapX=x, mapY=y)
        world['entities'] = [dict(type='player', id='other', hp=10, isCurrentPlayer=False,
                                  mapX=x+1, mapY=y)]
        model.encode(world)
        model.record_action(3)
        self.assertEqual(model.pending_estimate['collision_kind'], 'entity')

    def test_family_cooldown_and_terrain_reuse(self):
        base = ScenarioConfig()
        # Shorter slot deadline isolates the simulator's family-cooldown branch.
        config = replace(base, spells=(replace(base.spells[0], cooldown_ms=100), *base.spells[1:]))
        model = SimulatedFeedbackInputs(config)
        world = self.world(entities=[dict(type='monster', id=99, isCurrentPlayer=False,
                           mapX=51, mapY=50, hp=10, maxHp=10, mp=0, maxMp=0)])
        model.encode(world)
        model.record_action(4)
        world['timestamp'] += 200
        model.encode(world)
        model.record_action(4)
        self.assertEqual(model.pending_estimate['failure_reason'], 'family_cooldown')
        x, y = next((x,y) for x,y in model.encoder.map.blocked_cells if x > 0)
        world['timestamp'] += 200
        world['player'].update(mapX=x-1, mapY=y)
        model.encode(world)
        model.record_action(3)
        self.assertEqual(model.pending_estimate['collision_kind'], 'terrain')

    def test_duplicates_and_death_do_not_fabricate_confirmed_outcomes(self):
        clock = Clock()
        model = SimulatedFeedbackInputs(ScenarioConfig())
        actor = StubActor(actions=(5, 6))
        run = SingleRun(actor, input_adapter=model, progression='snapshot', monotonic=clock,
                        command_adapter=LiveCommands(model.encoder))
        first = request(0, clock=clock)
        first['capabilities'] = [COMMAND_SCHEMA]
        run.handle(first)
        state = model.diagnostics()
        run.handle(first)
        self.assertEqual(model.diagnostics(), state)
        clock.advance()
        second = request(1, clock=clock)
        second['capabilities'] = [COMMAND_SCHEMA]
        run.handle(second)
        self.assertEqual(run.action_count, 2)
        view = run.snapshot()
        self.assertEqual(view['current_policy_input']['history_source'], 'simulator_estimated')
        self.assertEqual(view['completed_actions'], [])
        self.assertIsNone(view['history']['action_applied'])
        self.assertIsNone(view['issued_commands'][0]['execution_outcome'])
        self.assertEqual(view['issued_commands'][0]['simulator_estimate']['source'], 'simulator_estimated')
        run.handle(request(2, hp=0))
        self.assertEqual(run.stop_reason, 'player_death')
        self.assertEqual(run.action_count, 2)

    def test_checkpoint_cli_opt_in_and_mutually_exclusive_modes(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = policy_tests.FrozenPolicyTests().checkpoint(folder, recurrent=True)
            actor = FrozenPolicy(path, assume_sim_parity=True)
            self.assertFalse(actor.input_adapter.requires_execution_feedback)
            self.assertEqual(actor.input_size, 80)
            with patch.object(server.zmq, 'Context'), \
                 patch.object(server, 'serve', side_effect=lambda run, socket, **kw: run.stop()) as serve:
                server.main(['--checkpoint', str(path), '--assume-sim-parity',
                             '--snapshot-output', str(Path(folder)/'run.json')])
                self.assertEqual(serve.call_args.args[0].progression, 'snapshot')
            with patch.object(server.zmq, 'Context') as context:
                with self.assertRaises(SystemExit):
                    server.main(['--checkpoint', str(path), '--assume-sim-parity', '--confirmed-feedback'])
                context.assert_not_called()
            path, _ = policy_tests.FrozenPolicyTests().checkpoint(folder)
            with self.assertRaisesRegex(ValueError, 'only for existing 80-input'):
                FrozenPolicy(path, assume_sim_parity=True)
